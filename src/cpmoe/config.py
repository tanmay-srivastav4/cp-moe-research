from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    name: str
    output_dir: str
    seed: int = 7


@dataclass
class ModelConfig:
    name_or_path: str
    torch_dtype: str = "float32"
    load_in_4bit: bool = False
    trust_remote_code: bool = False


@dataclass
class CPMoEConfig:
    num_experts: int
    top_k: int
    rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    cp_bias_alpha: float
    reg_lambda: float
    aux_gamma: float
    damping_xi: float


@dataclass
class TransientConfig:
    warmup_tokens: int
    max_steps: int
    lr: float


@dataclass
class TrainingConfig:
    lr: float
    batch_size: int
    gradient_accumulation_steps: int
    epochs_per_task: int
    max_seq_length: int
    save_after_each_task: bool = True
    eval_batch_size: int = 4
    max_new_tokens: int = 64
    train_max_samples: int | None = None
    eval_max_samples: int | None = None
    evaluate_seen_after_each_task: bool = True


@dataclass
class DataConfig:
    mode: str
    tasks: list[dict[str, Any]] | None = None
    order: list[str] | None = None
    zero_shot: list[str] | None = None
    data_dir: str | None = None
    eval_fraction: float = 0.1
    eval_max_samples: int | None = None
    train_max_samples: int | None = None


@dataclass
class Config:
    experiment: ExperimentConfig
    model: ModelConfig
    cp_moe: CPMoEConfig
    transient: TransientConfig
    training: TrainingConfig
    data: DataConfig


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Config(
        experiment=ExperimentConfig(**raw["experiment"]),
        model=ModelConfig(**raw["model"]),
        cp_moe=CPMoEConfig(**raw["cp_moe"]),
        transient=TransientConfig(**raw["transient"]),
        training=TrainingConfig(**raw["training"]),
        data=DataConfig(**raw["data"]),
    )


def asdict_shallow(config: Config) -> dict[str, Any]:
    return {
        "experiment": vars(config.experiment),
        "model": vars(config.model),
        "cp_moe": vars(config.cp_moe),
        "transient": vars(config.transient),
        "training": vars(config.training),
        "data": vars(config.data),
    }