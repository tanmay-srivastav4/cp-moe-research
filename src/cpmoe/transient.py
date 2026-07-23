from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .modeling import MoELoraLinear


@dataclass
class ProbeStats:
    warmup_steps: int
    mean_cka: float
    mean_importance: float


@torch.no_grad()
def reset_cp_biases(layers: list[MoELoraLinear]) -> None:
    for layer in layers:
        layer.zero_cp_bias()


def expert_regularization(
    layers: list[MoELoraLinear],
    snapshots: list[dict[str, torch.Tensor]],
    importance: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> torch.Tensor:
    if not layers:
        raise ValueError("No MoE layers passed to expert_regularization")
    # Use the first non-zero loss device; fall back to CPU.
    # The actual tensors are moved inside the loop so device_map splits are safe.
    total = torch.zeros((), device=device)
    for layer, old, omega in zip(layers, snapshots, importance, strict=True):
        # lora_a/lora_b live on CPU (Parameters), compute on CPU to avoid
        # cross-device ops; the backward graph stays correct.
        layer_device = layer.lora_a.device
        old_a = old["a"].to(layer_device)
        old_b = old["b"].to(layer_device)
        omega_a = omega["a"].to(layer_device)
        omega_b = omega["b"].to(layer_device)
        diff_a = (layer.lora_a - old_a).pow(2)
        diff_b = (layer.lora_b - old_b).pow(2)
        term = (omega_a * diff_a).sum() + (omega_b * diff_b).sum()
        total = total + term.to(device)
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


def run_transient_probe(
    model: nn.Module,
    layers: list[MoELoraLinear],
    warmup_loader: Iterable[dict[str, torch.Tensor]],
    importance: list[dict[str, torch.Tensor]],
    *,
    max_steps: int,
    lr: float,
    damping_xi: float,
    cp_bias_alpha: float,
    device: torch.device,
    cka_tokens_per_layer: int = 256,
) -> ProbeStats:
    if not layers:
        raise ValueError("No MoE layers passed to run_transient_probe")

    original_train_mode = model.training
    requires_grad_state = {
        name: param.requires_grad for name, param in model.named_parameters()
    }
    transient_initial: list[dict[str, torch.Tensor]] = []
    path_work: list[dict[str, torch.Tensor]] = []

    try:
        _set_all_requires_grad(model, False)
        for layer in layers:
            layer.start_transient_probe()
            for param in layer.transient_parameters():
                param.requires_grad = True
            transient_initial.append(
                {
                    "a": layer.transient_a.detach().clone().cpu(),
                    "b": layer.transient_b.detach().clone().cpu(),
                }
            )
            path_work.append(
                {
                    "a": torch.zeros_like(layer.transient_a, device="cpu"),
                    "b": torch.zeros_like(layer.transient_b, device="cpu"),
                }
            )

        optimizer = torch.optim.AdamW(_transient_parameters(layers), lr=lr)
        warmup_steps = _run_warmup_steps(
            model=model,
            layers=layers,
            warmup_loader=warmup_loader,
            optimizer=optimizer,
            path_work=path_work,
            max_steps=max_steps,
            device=device,
        )

        transient_importance = _normalise_importance(
            layers=layers,
            initial=transient_initial,
            path_work=path_work,
            damping_xi=damping_xi,
        )
        cka_scores = _estimate_cka_scores(
            model=model,
            layers=layers,
            warmup_loader=warmup_loader,
            device=device,
            cka_tokens_per_layer=cka_tokens_per_layer,
        )
        _apply_probe_results(
            layers=layers,
            importance=importance,
            transient_importance=transient_importance,
            cka_scores=cka_scores,
            cp_bias_alpha=cp_bias_alpha,
        )
        return ProbeStats(
            warmup_steps=warmup_steps,
            mean_cka=_mean_score(cka_scores),
            mean_importance=_mean_importance(transient_importance),
        )
    finally:
        for layer in layers:
            layer.stop_transient_probe()
        _restore_requires_grad(model, requires_grad_state)
        model.train(original_train_mode)


def _run_warmup_steps(
    *,
    model: nn.Module,
    layers: list[MoELoraLinear],
    warmup_loader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    path_work: list[dict[str, torch.Tensor]],
    max_steps: int,
    device: torch.device,
) -> int:
    model.train()
    steps = 0
    for batch in warmup_loader:
        if steps >= max_steps:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        before = _snapshot_transient(layers)
        out = model(**batch)
        out.loss.backward()
        grads = _snapshot_transient_grads(layers)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _accumulate_path_work(layers, before, grads, path_work)
        steps += 1
    if steps == 0:
        raise ValueError("Transient warm-up received no batches")
    return steps


@torch.no_grad()
def _estimate_cka_scores(
    *,
    model: nn.Module,
    layers: list[MoELoraLinear],
    warmup_loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    cka_tokens_per_layer: int,
) -> list[torch.Tensor]:
    model.eval()
    for layer in layers:
        layer.start_input_capture(cka_tokens_per_layer)

    batch = next(iter(warmup_loader), None)
    if batch is None:
        raise ValueError("Cannot estimate CKA without warm-up batches")
    batch = {k: v.to(device) for k, v in batch.items()}
    model(**batch)

    all_scores = []
    for layer in layers:
        flat_x = layer.stop_input_capture()  # always CPU from _maybe_capture_input
        if flat_x is None or flat_x.numel() == 0:
            scores = torch.ones(layer.num_experts) / layer.num_experts
        else:
            # CKA runs on CPU to avoid device-split issues
            scores = _layer_cka_scores(layer, flat_x.cpu())
        all_scores.append(scores)
    return all_scores


@torch.no_grad()
def _layer_cka_scores(
    layer: MoELoraLinear, flat_x: torch.Tensor
) -> torch.Tensor:
    # Both transient_output and expert_outputs move weights to flat_x.device
    transient_z = layer._transient_output(flat_x)
    expert_z = layer._expert_outputs(flat_x)
    scores = []
    for i in range(layer.num_experts):
        scores.append(_linear_cka(transient_z, expert_z[:, i, :]))
    stacked = torch.stack(scores)
    if stacked.sum().abs() < 1e-8:
        return torch.ones_like(stacked) / stacked.numel()
    return stacked.clamp_min(0.0)


@torch.no_grad()
def _linear_cka(
    x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    x = x.float() - x.float().mean(dim=0, keepdim=True)
    y = y.float() - y.float().mean(dim=0, keepdim=True)
    xy = torch.matmul(x.t(), y)
    xx = torch.matmul(x.t(), x)
    yy = torch.matmul(y.t(), y)
    numerator = xy.pow(2).sum()
    denominator = torch.linalg.matrix_norm(xx) * torch.linalg.matrix_norm(yy) + eps
    return (numerator / denominator).clamp(0.0, 1.0)


@torch.no_grad()
def _normalise_importance(
    *,
    layers: list[MoELoraLinear],
    initial: list[dict[str, torch.Tensor]],
    path_work: list[dict[str, torch.Tensor]],
    damping_xi: float,
) -> list[dict[str, torch.Tensor]]:
    normalised = []
    for layer, start, work in zip(layers, initial, path_work, strict=True):
        delta_a = layer.transient_a.cpu() - start["a"]
        delta_b = layer.transient_b.cpu() - start["b"]
        omega_a = (work["a"] / (delta_a.pow(2) + damping_xi)).clamp_min(0.0)
        omega_b = (work["b"] / (delta_b.pow(2) + damping_xi)).clamp_min(0.0)
        normalised.append({"a": omega_a, "b": omega_b})
    return normalised


@torch.no_grad()
def _apply_probe_results(
    *,
    layers: list[MoELoraLinear],
    importance: list[dict[str, torch.Tensor]],
    transient_importance: list[dict[str, torch.Tensor]],
    cka_scores: list[torch.Tensor],
    cp_bias_alpha: float,
) -> None:
    for layer, total, omega, scores in zip(
        layers, importance, transient_importance, cka_scores, strict=True
    ):
        layer.set_cp_bias(scores, cp_bias_alpha)
        target = total["a"].device
        scores_t = scores.to(target).float()
        omega_a = omega["a"].to(target)
        omega_b = omega["b"].to(target)
        total["a"].add_(scores_t.view(-1, 1, 1) * omega_a.unsqueeze(0))
        total["b"].add_(scores_t.view(-1, 1, 1) * omega_b.unsqueeze(0))


@torch.no_grad()
def _snapshot_transient(
    layers: list[MoELoraLinear],
) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "a": layer.transient_a.detach().clone().cpu(),
            "b": layer.transient_b.detach().clone().cpu(),
        }
        for layer in layers
    ]


@torch.no_grad()
def _snapshot_transient_grads(
    layers: list[MoELoraLinear],
) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "a": layer.transient_a.grad.detach().clone().cpu(),
            "b": layer.transient_b.grad.detach().clone().cpu(),
        }
        for layer in layers
    ]


@torch.no_grad()
def _accumulate_path_work(
    layers: list[MoELoraLinear],
    before: list[dict[str, torch.Tensor]],
    grads: list[dict[str, torch.Tensor]],
    path_work: list[dict[str, torch.Tensor]],
) -> None:
    for layer, old, grad, work in zip(layers, before, grads, path_work, strict=True):
        # Eq. 6: omega += -g * delta_phi  (all CPU)
        work["a"].add_(-grad["a"] * (layer.transient_a.cpu() - old["a"]))
        work["b"].add_(-grad["b"] * (layer.transient_b.cpu() - old["b"]))


def _transient_parameters(layers: list[MoELoraLinear]) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for layer in layers:
        params.extend(layer.transient_parameters())
    return params


def _set_all_requires_grad(model: nn.Module, value: bool) -> None:
    for param in model.parameters():
        param.requires_grad = value


def _restore_requires_grad(model: nn.Module, state: dict[str, bool]) -> None:
    for name, param in model.named_parameters():
        if name in state:
            param.requires_grad = state[name]


def _mean_score(scores: list[torch.Tensor]) -> float:
    if not scores:
        return 0.0
    return float(
        torch.stack([s.detach().float().cpu().mean() for s in scores]).mean()
    )


def _mean_importance(importance: list[dict[str, torch.Tensor]]) -> float:
    if not importance:
        return 0.0
    values = []
    for item in importance:
        values.append(item["a"].float().cpu().mean())
        values.append(item["b"].float().cpu().mean())
    return float(torch.stack(values).mean())