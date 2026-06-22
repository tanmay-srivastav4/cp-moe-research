from __future__ import annotations

import torch

from .modeling import MoELoraLinear


@torch.no_grad()
def reset_cp_biases(layers: list[MoELoraLinear]) -> None:
    for layer in layers:
        layer.zero_cp_bias()


@torch.no_grad()
def apply_uniform_cp_bias(layers: list[MoELoraLinear], alpha: float) -> None:
    for layer in layers:
        scores = torch.ones(layer.num_experts, device=layer.cp_bias.device) / layer.num_experts
        layer.set_cp_bias(scores, alpha)


def expert_regularization(
    layers: list[MoELoraLinear],
    snapshots: list[dict[str, torch.Tensor]],
    importance: list[dict[str, torch.Tensor]],
) -> torch.Tensor:
    if not layers:
        raise ValueError("No MoE layers passed to expert_regularization")
    device = layers[0].lora_a.device
    total = torch.zeros((), device=device)
    for layer, old, omega in zip(layers, snapshots, importance, strict=True):
        total = total + (omega["a"] * (layer.lora_a - old["a"]).pow(2)).sum()
        total = total + (omega["b"] * (layer.lora_b - old["b"]).pow(2)).sum()
    return total


@torch.no_grad()
def snapshot_experts(layers: list[MoELoraLinear]) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "a": layer.lora_a.detach().clone(),
            "b": layer.lora_b.detach().clone(),
        }
        for layer in layers
    ]


@torch.no_grad()
def zero_importance(layers: list[MoELoraLinear]) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "a": torch.zeros_like(layer.lora_a),
            "b": torch.zeros_like(layer.lora_b),
        }
        for layer in layers
    ]


@torch.no_grad()
def accumulate_dummy_importance(
    layers: list[MoELoraLinear],
    current: list[dict[str, torch.Tensor]],
    scale: float = 1.0,
) -> None:
    """Temporary stand-in until full transient probe trajectory code is wired.

    This lets us validate the regularized continual training loop before adding
    the expensive transient expert path-integral implementation.
    """
    for layer, omega in zip(layers, current, strict=True):
        omega["a"].add_(torch.ones_like(layer.lora_a) * scale)
        omega["b"].add_(torch.ones_like(layer.lora_b) * scale)

