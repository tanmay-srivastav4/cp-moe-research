from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
        json.dumps(asdict_shallow(config), indent=2), encoding="utf-8"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}  {props.total_memory/1e9:.1f} GB")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        print("WARNING: HF_TOKEN not set — gated models like LLaMA-2 will fail")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        trust_remote_code=config.model.trust_remote_code,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # ── Model loading ──────────────────────────────────────────────────────
    # device_map="auto" with bfloat16 is the right choice for Kaggle dual-GPU.
    # The MoE params are stored as CPU Parameters and moved to the right device
    # inside forward() via _to_x(), so multi-GPU splits work transparently.
    if config.model.load_in_4bit:
        print("Loading in 4-bit (bitsandbytes NF4)...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path,
            quantization_config=bnb_cfg,
            device_map=config.model.device_map,
            trust_remote_code=config.model.trust_remote_code,
            token=hf_token,
            attn_implementation="sdpa",
        )
    else:
        print(f"Loading in {config.model.torch_dtype} with device_map=auto...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path,
            torch_dtype=resolve_dtype(config.model.torch_dtype),
            device_map=config.model.device_map,
            trust_remote_code=config.model.trust_remote_code,
            token=hf_token,
            attn_implementation="sdpa",
        )

    for param in model.parameters():
        param.requires_grad = False

    # ── Inject MoE-LoRA ────────────────────────────────────────────────────
    layers = replace_target_linears(
        model,
        target_modules=config.cp_moe.target_modules,
        num_experts=config.cp_moe.num_experts,
        rank=config.cp_moe.rank,
        top_k=config.cp_moe.top_k,
        lora_alpha=config.cp_moe.lora_alpha,
        lora_dropout=config.cp_moe.lora_dropout,
    )
    # MoE params (lora_a, lora_b, router_weight) live on CPU intentionally.
    # They are moved to the active GPU inside forward() via _to_x().

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    # ── Data ───────────────────────────────────────────────────────────────
    tasks = load_tasks(config.data)
    all_tasks = load_tasks(config.data, include_zero_shot=True)
    zero_shot_tasks = {
        t.name: t for t in all_tasks if t.name in (config.data.zero_shot or [])
    }
    train_task_names = [t.name for t in tasks]

    # ── Continual state ────────────────────────────────────────────────────
    importance = zero_importance(layers)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.training.lr,
        weight_decay=0.01,
    )
    grad_accum = max(1, config.training.gradient_accumulation_steps)

    metrics: list[dict] = []
    score_matrix: list[dict[str, float]] = []

    # ── Task loop ──────────────────────────────────────────────────────────
    for task_idx, task in enumerate(tasks, start=1):
        print(f"\n{'='*60}\nTask {task_idx}/{len(tasks)}: {task.name}\n{'='*60}")

        old_experts = snapshot_experts(layers)

        # Transient probe
        warmup_rows = iter_token_budget(
            task.train, tokenizer, config.transient.warmup_tokens
        )
        warmup_ds = EncodedRows(warmup_rows, tokenizer, config.training.max_seq_length)
        warmup_loader = DataLoader(
            warmup_ds, batch_size=config.training.batch_size,
            shuffle=True, collate_fn=warmup_ds.collate,
        )
        probe = run_transient_probe(
            model=model, layers=layers, warmup_loader=warmup_loader,
            importance=importance, max_steps=config.transient.max_steps,
            lr=config.transient.lr, damping_xi=config.cp_moe.damping_xi,
            cp_bias_alpha=config.cp_moe.cp_bias_alpha, device=device,
        )
        print(f"Probe: steps={probe.warmup_steps}  mean_cka={probe.mean_cka:.4f}  "
              f"mean_importance={probe.mean_importance:.4f}")

        # Main training
        train_ds = EncodedRows(
            _maybe_limit(task.train, config.training.train_max_samples),
            tokenizer, config.training.max_seq_length,
        )
        loader = DataLoader(
            train_ds, batch_size=config.training.batch_size,
            shuffle=True, collate_fn=train_ds.collate,
        )

        model.train()
        for epoch in range(config.training.epochs_per_task):
            optimizer.zero_grad(set_to_none=True)
            bar = tqdm(loader, desc=f"{task.name} ep{epoch+1}")
            for step, batch in enumerate(bar, start=1):
                batch = {k: v.to(device) for k, v in batch.items()}
                task_loss = model(**batch).loss
                aux_loss = moe_auxiliary_loss(model, device)
                # reg_loss computed on CPU (lora params are CPU tensors)
                reg_loss = expert_regularization(layers, old_experts, importance)

                total_loss = (
                    task_loss
                    + config.cp_moe.aux_gamma * aux_loss
                    + config.cp_moe.reg_lambda * reg_loss.to(device)
                ) / grad_accum
                total_loss.backward()

                if step % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                bar.set_postfix(
                    task=f"{task_loss.item():.4f}",
                    aux=f"{aux_loss.item():.4f}",
                    reg=f"{reg_loss.item():.4f}",
                )

            # flush remaining grads at epoch end
            if len(loader) % grad_accum != 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # Evaluate seen tasks
        seen_scores: dict[str, float] = {}
        if config.training.evaluate_seen_after_each_task:
            for eval_task in tasks[:task_idx]:
                score = evaluate_task_generation(
                    model, tokenizer, eval_task,
                    max_seq_length=config.training.max_seq_length,
                    max_new_tokens=config.training.max_new_tokens,
                    batch_size=config.training.eval_batch_size,
                    device=device,
                    max_samples=config.training.eval_max_samples,
                )
                seen_scores[eval_task.name] = score.score
                print(f"  eval {eval_task.name}: {score.metric}={score.score:.2f}")
        score_matrix.append(seen_scores)

        eval_loss = _evaluate_loss(
            model, task.eval, tokenizer, config.training.max_seq_length, device
        )
        metrics.append({
            "task": task.name,
            "probe_steps": probe.warmup_steps,
            "probe_mean_cka": probe.mean_cka,
            "probe_mean_importance": probe.mean_importance,
            **eval_loss,
            "seen_scores": seen_scores,
        })
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        (output_dir / "score_matrix.json").write_text(
            json.dumps(score_matrix, indent=2), encoding="utf-8"
        )

        if config.training.save_after_each_task:
            ckpt_dir = ensure_dir(output_dir / f"task_{task_idx:02d}_{task.name}")
            # Save only LoRA params (much smaller than full model)
            lora_state = {
                n: p for n, p in model.named_parameters() if p.requires_grad
            }
            torch.save(lora_state, ckpt_dir / "lora_checkpoint.pt")
            print(f"  LoRA checkpoint → {ckpt_dir}")

    # ── Zero-shot evaluation ───────────────────────────────────────────────
    print(f"\n{'='*60}\nZero-shot evaluation\n{'='*60}")
    zs_scores: dict[str, float] = {}
    for name, zt in zero_shot_tasks.items():
        score = evaluate_task_generation(
            model, tokenizer, zt,
            max_seq_length=config.training.max_seq_length,
            max_new_tokens=config.training.max_new_tokens,
            batch_size=config.training.eval_batch_size,
            device=device,
            max_samples=config.training.eval_max_samples,
        )
        zs_scores[name] = score.score
        print(f"  zero-shot {name}: {score.metric}={score.score:.2f}")

    summary = build_summary(train_task_names, score_matrix, zs_scores)
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nFinal summary:")
    print(json.dumps(summary, indent=2))


# ── Dataset ────────────────────────────────────────────────────────────────

class EncodedRows(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        prompt_len = len(
            self.tokenizer(row["input"], add_special_tokens=False).input_ids
        )
        full = row["input"] + row["output"]
        enc = self.tokenizer(
            full, truncation=True, max_length=self.max_length, add_special_tokens=True
        )
        ids = enc.input_ids
        labels = list(ids)
        mask_len = min(prompt_len + 1, len(labels))  # +1 for BOS
        labels[:mask_len] = [-100] * mask_len
        return {"input_ids": ids, "attention_mask": enc.attention_mask, "labels": labels}

    def collate(self, rows):
        pad = self.tokenizer.pad_token_id
        L = max(len(r["input_ids"]) for r in rows)
        ids, mask, labs = [], [], []
        for r in rows:
            p = L - len(r["input_ids"])
            ids.append(r["input_ids"] + [pad] * p)
            mask.append(r["attention_mask"] + [0] * p)
            labs.append(r["labels"] + [-100] * p)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(labs, dtype=torch.long),
        }


@torch.no_grad()
def _evaluate_loss(model, rows, tokenizer, max_length, device) -> dict[str, float]:
    model.eval()
    ds = EncodedRows(rows, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=ds.collate)
    losses = [
        float(model(**{k: v.to(device) for k, v in b.items()}).loss.cpu())
        for b in loader
    ]
    model.train()
    return {"eval_loss": sum(losses) / max(len(losses), 1)}


def _maybe_limit(rows, limit):
    return rows if limit is None else rows[:limit]


if __name__ == "__main__":
    main()