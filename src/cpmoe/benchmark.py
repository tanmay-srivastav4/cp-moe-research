from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from .data import TaskDataset

try:
    from rouge_score import rouge_scorer
except ImportError:  # pragma: no cover
    rouge_scorer = None


@dataclass
class TaskScore:
    task: str
    metric: str
    score: float
    num_examples: int


class PromptRows(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict[str, str]], tokenizer, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        encoded = self.tokenizer(
            row["input"],
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        return {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "target": row["output"],
        }

    def collate(self, rows):
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(row["input_ids"]) for row in rows)
        input_ids = []
        attention_mask = []
        targets = []
        for row in rows:
            pad = max_len - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [pad_id] * pad)
            attention_mask.append(row["attention_mask"] + [0] * pad)
            targets.append(row["target"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "targets": targets,
        }


@torch.no_grad()
def evaluate_task_generation(
    model,
    tokenizer,
    task: TaskDataset,
    *,
    max_seq_length: int,
    max_new_tokens: int,
    batch_size: int,
    device: torch.device,
    max_samples: int | None = None,
) -> TaskScore:
    model.eval()
    rows = task.eval[:max_samples] if max_samples is not None else task.eval
    dataset = PromptRows(rows, tokenizer, max_seq_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)
    predictions: list[str] = []
    targets: list[str] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        for output_ids, prompt_len in zip(generated, prompt_lengths, strict=True):
            new_tokens = output_ids[int(prompt_len) :]
            predictions.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        targets.extend(batch["targets"])

    score = score_predictions(predictions, targets, task.metric)
    model.train()
    return TaskScore(task=task.name, metric=task.metric, score=score, num_examples=len(targets))


def score_predictions(predictions: list[str], targets: list[str], metric: str) -> float:
    if not predictions:
        return 0.0
    if metric == "accuracy":
        correct = 0
        for prediction, target in zip(predictions, targets, strict=True):
            correct += int(_normalise_label(prediction) == _normalise_label(target))
        return 100.0 * correct / len(predictions)
    if metric == "rouge_l":
        if rouge_scorer is None:
            raise ImportError("rouge-score is required for ROUGE-L evaluation")
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = []
        for prediction, target in zip(predictions, targets, strict=True):
            scores.append(scorer.score(target, prediction)["rougeL"].fmeasure)
        return 100.0 * sum(scores) / len(scores)
    raise ValueError(f"Unsupported metric: {metric}")


def build_summary(
    train_task_names: list[str],
    score_matrix: list[dict[str, float]],
    zero_shot_scores: dict[str, float],
) -> dict[str, float | dict[str, float]]:
    final_seen = score_matrix[-1] if score_matrix else {}
    final_scores = {task: final_seen.get(task, 0.0) for task in train_task_names}
    ap = _mean(final_scores.values())
    af = _average_forgetting(train_task_names, score_matrix)
    zst = _mean(zero_shot_scores.values()) if zero_shot_scores else 0.0
    return {
        "AP": ap,
        "AF": af,
        "ZST": zst,
        "final_seen_scores": final_scores,
        "zero_shot_scores": zero_shot_scores,
    }


def _average_forgetting(task_names: list[str], score_matrix: list[dict[str, float]]) -> float:
    if len(score_matrix) < 2:
        return 0.0
    forget_values = []
    final = score_matrix[-1]
    for task in task_names[:-1]:
        observed = [row[task] for row in score_matrix if task in row]
        if not observed:
            continue
        best_before_final = max(observed[:-1] or observed)
        forget_values.append(best_before_final - final.get(task, 0.0))
    return _mean(forget_values)


def _mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _normalise_label(text: str) -> str:
    text = text.strip().lower()
    text = text.splitlines()[0] if text else text
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text