from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:  # pragma: no cover - transformers is optional for syntax-only checks.
    Conv1D = ()  # type: ignore[assignment]


@dataclass
class MoEMetrics:
    aux_loss: torch.Tensor
    router_entropy: torch.Tensor


class MoELoraLinear(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        num_experts: int,
        rank: int,
        top_k: int,
        lora_alpha: int,
        lora_dropout: float,
    ) -> None:
        super().__init__()
        self.base = base
        self.num_experts = num_experts
        self.rank = rank
        self.top_k = top_k
        self.scaling = lora_alpha / rank
        self.dropout = nn.Dropout(lora_dropout)

        for param in self.base.parameters():
            param.requires_grad = False

        in_features, out_features = self._feature_dims(base)
        self.lora_a = nn.Parameter(torch.zeros(num_experts, rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(num_experts, out_features, rank))
        self.router = nn.Linear(in_features, num_experts, bias=False)
        self.cp_bias = nn.Parameter(torch.zeros(num_experts), requires_grad=False)

        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        nn.init.zeros_(self.lora_b)
        nn.init.zeros_(self.router.weight)

        self.last_aux_loss: torch.Tensor | None = None
        self.last_router_entropy: torch.Tensor | None = None

    @staticmethod
    def _feature_dims(base: nn.Module) -> tuple[int, int]:
        if isinstance(base, nn.Linear):
            return base.in_features, base.out_features
        if Conv1D and isinstance(base, Conv1D):
            # HF GPT-style Conv1D stores weight as [in_features, out_features].
            return int(base.weight.shape[0]), int(base.weight.shape[1])
        raise TypeError(f"Unsupported base module for MoE-LoRA: {type(base).__name__}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        original_shape = x.shape
        flat_x = x.reshape(-1, original_shape[-1])
        dropped = self.dropout(flat_x)

        logits = self.router(flat_x)
        biased_logits = logits + self.cp_bias.to(logits.dtype)
        top_values, top_indices = torch.topk(biased_logits, k=self.top_k, dim=-1)
        top_weights = F.softmax(top_values, dim=-1)

        expert_hidden = torch.einsum("bi,eri->ber", dropped, self.lora_a)
        expert_out = torch.einsum("ber,eor->beo", expert_hidden, self.lora_b) * self.scaling
        chosen = torch.gather(
            expert_out,
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, expert_out.shape[-1]),
        )
        mixed = (chosen * top_weights.unsqueeze(-1)).sum(dim=1)
        mixed = mixed.reshape(*original_shape[:-1], -1)

        self._record_router_metrics(logits, top_indices)
        return base_out + mixed

    def _record_router_metrics(self, logits: torch.Tensor, top_indices: torch.Tensor) -> None:
        probs = F.softmax(logits, dim=-1)
        load = F.one_hot(top_indices, num_classes=self.num_experts).float().mean(dim=(0, 1))
        prob_mean = probs.mean(dim=0)
        self.last_aux_loss = self.num_experts * torch.sum(load * prob_mean)
        self.last_router_entropy = -(prob_mean * (prob_mean + 1e-8).log()).sum()

    def set_cp_bias(self, scores: torch.Tensor, alpha: float) -> None:
        if scores.numel() != self.num_experts:
            raise ValueError("CP bias score count must match num_experts")
        self.cp_bias.data.copy_(alpha * scores.to(self.cp_bias.device, self.cp_bias.dtype))

    def zero_cp_bias(self) -> None:
        self.cp_bias.data.zero_()


def replace_target_linears(
    model: nn.Module,
    target_modules: list[str],
    num_experts: int,
    rank: int,
    top_k: int,
    lora_alpha: int,
    lora_dropout: float,
) -> list[MoELoraLinear]:
    wrapped: list[MoELoraLinear] = []
    for parent, child_name, child in _iter_named_children_with_parent(model):
        if not _is_supported_linear(child):
            continue
        if child_name not in target_modules:
            continue
        moe = MoELoraLinear(
            child,
            num_experts=num_experts,
            rank=rank,
            top_k=top_k,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        setattr(parent, child_name, moe)
        wrapped.append(moe)
    if not wrapped:
        raise ValueError(f"No target Linear modules matched: {target_modules}")
    return wrapped


def _is_supported_linear(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) or bool(Conv1D and isinstance(module, Conv1D))


def _iter_named_children_with_parent(module: nn.Module):
    for child_name, child in module.named_children():
        yield module, child_name, child
        yield from _iter_named_children_with_parent(child)


def collect_moe_layers(model: nn.Module) -> list[MoELoraLinear]:
    return [module for module in model.modules() if isinstance(module, MoELoraLinear)]


def moe_auxiliary_loss(model: nn.Module, device: torch.device) -> torch.Tensor:
    losses = [
        layer.last_aux_loss
        for layer in collect_moe_layers(model)
        if layer.last_aux_loss is not None
    ]
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


