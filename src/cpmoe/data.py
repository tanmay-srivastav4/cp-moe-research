from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import json

from .config import DataConfig


SUPERNI_TASKS = {
    "task1572": {"dataset": "samsum_summary", "type": "question_answering", "metric": "rouge_l"},
    "task363": {"dataset": "sst2_polarity_classification", "type": "sentiment", "metric": "accuracy"},
    "task1290": {"dataset": "xsum_summarization", "type": "question_answering", "metric": "rouge_l"},
    "task181": {"dataset": "outcome_extraction", "type": "info_extraction", "metric": "rouge_l"},
    "task002": {"dataset": "quoref_answer_generation", "type": "dialogue_generation", "metric": "rouge_l"},
    "task1510": {"dataset": "evaluation_relation_extraction", "type": "info_extraction", "metric": "rouge_l"},
    "task639": {"dataset": "multi_woz_user_utterance_generation", "type": "summarization", "metric": "rouge_l"},
    "task1729": {"dataset": "personachat_generate_next", "type": "summarization", "metric": "rouge_l"},
    "task073": {"dataset": "commonsenseqa_answer_generation", "type": "dialogue_generation", "metric": "rouge_l"},
    "task1590": {"dataset": "diplomacy_text_generation", "type": "summarization", "metric": "rouge_l"},
    "task748": {"dataset": "glucose_reverse_cause_event_detection", "type": "info_extraction", "metric": "rouge_l"},
    "task511": {"dataset": "reddit_tifu_long_text_summarization", "type": "question_answering", "metric": "rouge_l"},
    "task591": {"dataset": "sciq_answer_generation", "type": "dialogue_generation", "metric": "rouge_l"},
    "task084": {"dataset": "sentiment140_classification", "type": "sentiment", "metric": "accuracy"},
    "task875": {"dataset": "emotion_classification", "type": "sentiment", "metric": "accuracy"},
}


@dataclass
class TaskDataset:
    name: str
    train: list[dict[str, str]]
    eval: list[dict[str, str]]
    metric: str = "rouge_l"
    dataset_name: str | None = None
    task_type: str | None = None


def load_tasks(config: DataConfig, include_zero_shot: bool = False) -> list[TaskDataset]:
    if config.mode == "toy":
        return _load_toy_tasks(config)
    if config.mode == "superni":
        return _load_superni_tasks(config, include_zero_shot=include_zero_shot)
    raise ValueError(f"Unsupported data mode: {config.mode}")


def _load_toy_tasks(config: DataConfig) -> list[TaskDataset]:
    tasks = []
    for task in config.tasks or []:
        rows = [{"input": row["input"], "output": row["output"]} for row in task["train"]]
        tasks.append(
            TaskDataset(
                name=task["name"],
                train=rows,
                eval=rows,
                metric=task.get("metric", "accuracy"),
                dataset_name=task["name"],
                task_type="toy",
            )
        )
    return tasks


def _load_superni_tasks(config: DataConfig, include_zero_shot: bool) -> list[TaskDataset]:
    if not config.data_dir:
        raise ValueError("data.data_dir is required for SuperNI mode")

    root = Path(config.data_dir)
    task_names = list(config.order or [])
    if include_zero_shot:
        task_names.extend(config.zero_shot or [])

    tasks = []
    for task_name in task_names:
        task_path = root / f"{task_name}.json"
        if not task_path.exists():
            raise FileNotFoundError(
                f"Missing {task_path}. Prepare SuperNI task JSON files before the real run."
            )
        rows = _read_superni_task(task_path)
        train_rows, eval_rows = _split_rows(
            rows,
            eval_fraction=config.eval_fraction,
            train_max_samples=config.train_max_samples,
            eval_max_samples=config.eval_max_samples,
        )
        meta = SUPERNI_TASKS.get(task_name, {})
        tasks.append(
            TaskDataset(
                name=task_name,
                train=train_rows,
                eval=eval_rows,
                metric=meta.get("metric", "rouge_l"),
                dataset_name=meta.get("dataset", task_name),
                task_type=meta.get("type"),
            )
        )
    return tasks


def _read_superni_task(path: Path) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    definition = " ".join(raw.get("Definition", []))
    rows = []
    for instance in raw.get("Instances", []):
        input_text = instance.get("input", "")
        outputs = instance.get("output", [])
        output = outputs[0] if isinstance(outputs, list) and outputs else str(outputs)
        rows.append(
            {
                "input": f"Definition: {definition}\nInput: {input_text}\nOutput:",
                "output": str(output),
            }
        )
    if not rows:
        raise ValueError(f"No instances found in {path}")
    return rows


def _split_rows(
    rows: list[dict[str, str]],
    *,
    eval_fraction: float,
    train_max_samples: int | None,
    eval_max_samples: int | None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    eval_count = max(1, int(len(rows) * eval_fraction))
    eval_rows = rows[-eval_count:]
    train_rows = rows[:-eval_count] or rows
    if train_max_samples is not None:
        train_rows = train_rows[:train_max_samples]
    if eval_max_samples is not None:
        eval_rows = eval_rows[:eval_max_samples]
    return train_rows, eval_rows


def iter_token_budget(rows: Iterable[dict[str, str]], tokenizer, budget: int) -> list[dict[str, str]]:
    selected = []
    total = 0
    for row in rows:
        selected.append(row)
        total += len(tokenizer(row["input"] + row["output"], add_special_tokens=False).input_ids)
        if total >= budget:
            break
    return selected