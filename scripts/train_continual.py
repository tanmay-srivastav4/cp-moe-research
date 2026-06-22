from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cpmoe.benchmark import build_summary, evaluate_task_generation
from cpmoe.config import asdict_shallow, load_config
from cpmoe.data import iter_token_budget, load_tasks
from cpmoe.modeling import moe_auxiliary_loss, replace_target_linears
from cpmoe.train_utils import ensure_dir, resolve_dtype, set_seed
from cpmoe.transient import (
    expert_regularization,
    run_transient_probe,
    snapshot_experts,
    zero_importance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.experiment.seed)
    output_dir = ensure_dir(config.experiment.output_dir)
    (output_dir / "config.json").write_text(
        json.dumps(asdict_shallow(config), indent=2),
        encoding="utf-8",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        torch_dtype=resolve_dtype(config.model.torch_dtype),
        trust_remote_code=config.model.trust_remote_code,
    )
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)

    layers = replace_target_linears(
        model,
        target_modules=config.cp_moe.target_modules,
        num_experts=config.cp_moe.num_experts,
        rank=config.cp_moe.rank,
        top_k=config.cp_moe.top_k,
        lora_alpha=config.cp_moe.lora_alpha,
        lora_dropout=config.cp_moe.lora_dropout,
    )
    model.to(device)

    tasks = load_tasks(config.data)
    all_eval_tasks = load_tasks(config.data, include_zero_shot=True)
    zero_shot_tasks = {task.name: task for task in all_eval_tasks if task.name in (config.data.zero_shot or [])}
    train_task_names = [task.name for task in tasks]
    importance = zero_importance(layers)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.training.lr,
    )

    metrics = []
    score_matrix: list[dict[str, float]] = []
    for task_index, task in enumerate(tasks, start=1):
        print(f"\n=== Task {task_index}/{len(tasks)}: {task.name} ===")
        old_experts = snapshot_experts(layers)
        warmup_rows = iter_token_budget(task.train, tokenizer, config.transient.warmup_tokens)
        warmup_dataset = EncodedRows(warmup_rows, tokenizer, config.training.max_seq_length)
        warmup_loader = DataLoader(
            warmup_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            collate_fn=warmup_dataset.collate,
        )
        probe_stats = run_transient_probe(
            model=model,
            layers=layers,
            warmup_loader=warmup_loader,
            importance=importance,
            max_steps=config.transient.max_steps,
            lr=config.transient.lr,
            damping_xi=config.cp_moe.damping_xi,
            cp_bias_alpha=config.cp_moe.cp_bias_alpha,
            device=device,
        )
        print(
            "probe "
            f"steps={probe_stats.warmup_steps} "
            f"mean_cka={probe_stats.mean_cka:.4f} "
            f"mean_importance={probe_stats.mean_importance:.4f}"
        )

        dataset = EncodedRows(_maybe_limit(task.train, config.training.train_max_samples), tokenizer, config.training.max_seq_length)
        loader = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            collate_fn=dataset.collate,
        )

        for epoch in range(config.training.epochs_per_task):
            progress = tqdm(loader, desc=f"{task.name} epoch {epoch + 1}")
            for batch in progress:
                batch = {key: value.to(device) for key, value in batch.items()}
                out = model(**batch)
                task_loss = out.loss
                aux_loss = moe_auxiliary_loss(model, device)
                reg_loss = expert_regularization(layers, old_experts, importance)
                loss = (
                    task_loss
                    + config.cp_moe.aux_gamma * aux_loss
                    + config.cp_moe.reg_lambda * reg_loss
                )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                progress.set_postfix(
                    task=f"{task_loss.item():.4f}",
                    aux=f"{aux_loss.item():.4f}",
                    reg=f"{reg_loss.item():.4f}",
                )

        seen_scores = {}
        if config.training.evaluate_seen_after_each_task:
            for eval_task in tasks[:task_index]:
                task_score = evaluate_task_generation(
                    model,
                    tokenizer,
                    eval_task,
                    max_seq_length=config.training.max_seq_length,
                    max_new_tokens=config.training.max_new_tokens,
                    batch_size=config.training.eval_batch_size,
                    device=device,
                    max_samples=config.training.eval_max_samples,
                )
                seen_scores[eval_task.name] = task_score.score
                print(f"eval {eval_task.name} {task_score.metric}={task_score.score:.2f}")
        score_matrix.append(seen_scores)

        task_metrics = evaluate_loss(model, task.eval, tokenizer, config.training.max_seq_length, device)
        metrics.append(
            {
                "task": task.name,
                "probe_steps": probe_stats.warmup_steps,
                "probe_mean_cka": probe_stats.mean_cka,
                "probe_mean_importance": probe_stats.mean_importance,
                **task_metrics,
                "seen_scores": seen_scores,
            }
        )
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (output_dir / "score_matrix.json").write_text(json.dumps(score_matrix, indent=2), encoding="utf-8")

        if config.training.save_after_each_task:
            task_dir = ensure_dir(output_dir / f"task_{task_index:02d}_{task.name}")
            torch.save({"model": model.state_dict()}, task_dir / "checkpoint.pt")

    zero_shot_scores = {}
    for task_name, eval_task in zero_shot_tasks.items():
        task_score = evaluate_task_generation(
            model,
            tokenizer,
            eval_task,
            max_seq_length=config.training.max_seq_length,
            max_new_tokens=config.training.max_new_tokens,
            batch_size=config.training.eval_batch_size,
            device=device,
            max_samples=config.training.eval_max_samples,
        )
        zero_shot_scores[task_name] = task_score.score
        print(f"zero-shot {task_name} {task_score.metric}={task_score.score:.2f}")

    summary = build_summary(train_task_names, score_matrix, zero_shot_scores)
    (output_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nBenchmark summary")
    print(json.dumps(summary, indent=2))


class EncodedRows(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        prompt_ids = self.tokenizer(row["input"], add_special_tokens=False).input_ids
        full = row["input"] + row["output"]
        encoded = self.tokenizer(full, truncation=True, max_length=self.max_length, add_special_tokens=True)
        input_ids = encoded.input_ids
        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        return {"input_ids": input_ids, "attention_mask": encoded.attention_mask, "labels": labels}

    def collate(self, rows):
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(row["input_ids"]) for row in rows)
        input_ids = []
        attention_mask = []
        labels = []
        for row in rows:
            pad = max_len - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [pad_id] * pad)
            attention_mask.append(row["attention_mask"] + [0] * pad)
            labels.append(row["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@torch.no_grad()
def evaluate_loss(model, rows, tokenizer, max_length: int, device: torch.device) -> dict[str, float]:
    model.eval()
    dataset = EncodedRows(rows, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate)
    losses = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        losses.append(float(model(**batch).loss.detach().cpu()))
    model.train()
    return {"eval_loss": sum(losses) / max(len(losses), 1)}


def _maybe_limit(rows: list[dict[str, str]], limit: int | None) -> list[dict[str, str]]:
    if limit is None:
        return rows
    return rows[:limit]


if __name__ == "__main__":
    main()