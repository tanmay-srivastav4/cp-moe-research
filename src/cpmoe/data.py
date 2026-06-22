from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import json

from .config import DataConfig


@dataclass
class TaskDataset:
    name: str
    train: list[dict[str, str]]
    eval: list[dict[str, str]]


def load_tasks(config: DataConfig) -> list[TaskDataset]:
    if config.mode == "toy":
        return _load_toy_tasks(config)
    if config.mode == "superni":
        return _load_superni_tasks(config)
    raise ValueError(f"Unsupported data mode: {config.mode}")


def _load_toy_tasks(config: DataConfig) -> list[TaskDataset]:
    tasks = []
    for task in config.tasks or []:
        rows = [{"input": row["input"], "output": row["output"]} for row in task["train"]]
        tasks.append(TaskDataset(name=task["name"], train=rows, eval=rows))
    return tasks


def _load_superni_tasks(config: DataConfig) -> list[TaskDataset]:
    if not config.data_dir:
        raise ValueError("data.data_dir is required for SuperNI mode")

    root = Path(config.data_dir)
    tasks = []
    for task_name in config.order or []:
        task_path = root / f"{task_name}.json"
        if not task_path.exists():
            raise FileNotFoundError(
                f"Missing {task_path}. Prepare SuperNI task JSON files before the real run."
            )
        task = _read_superni_task(task_path)
        tasks.append(TaskDataset(name=task_name, train=task, eval=task))
    return tasks


def _read_superni_task(path: Path) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    definition = " ".join(raw.get("Definition", []))
    rows = []
    for instance in raw.get("Instances", []):
        input_text = instance.get("input", "")
        outputs = instance.get("output", [])
        output = outputs[0] if isinstance(outputs, list) and outputs else str(outputs)
        rows.append({"input": f"Definition: {definition}\nInput: {input_text}\nOutput:", "output": output})
    return rows


def iter_token_budget(rows: Iterable[dict[str, str]], tokenizer, budget: int) -> list[dict[str, str]]:
    selected = []
    total = 0
    for row in rows:
        selected.append(row)
        total += len(tokenizer(row["input"] + row["output"], add_special_tokens=False).input_ids)
        if total >= budget:
            break
    return selected

