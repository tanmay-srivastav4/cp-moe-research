from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:
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
        self.in_features = in_features
        self.out_features = out_features

        # Keep all MoE params as plain Parameters (CPU at init).
        # forward() moves them to flat_x.device on the fly, which correctly
        # handles device_map="auto" multi-GPU splits.
        self.lora_a = nn.Parameter(torch.zeros(num_experts, rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(num_experts, out_features, rank))
        self.router_weight = nn.Parameter(torch.zeros(num_experts, in_features))
        # cp_bias is not a gradient parameter; it's set externally.
        self.cp_bias = nn.Parameter(torch.zeros(num_experts), requires_grad=False)

        self.register_parameter("transient_a", None)
        self.register_parameter("transient_b", None)

        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        nn.init.zeros_(self.lora_b)
        nn.init.zeros_(self.router_weight)

        self.last_aux_loss: torch.Tensor | None = None
        self.last_router_entropy: torch.Tensor | None = None
        self._capture_inputs = False
        self._capture_max_tokens = 0
        self._captured_inputs: list[torch.Tensor] = []

    @staticmethod
    def _feature_dims(base: nn.Module) -> tuple[int, int]:
        if isinstance(base, nn.Linear):
            return base.in_features, base.out_features
        if Conv1D and isinstance(base, Conv1D):
            return int(base.weight.shape[0]), int(base.weight.shape[1])
        # bitsandbytes / other quantized linears
        if hasattr(base, "in_features") and hasattr(base, "out_features"):
            return base.in_features, base.out_features
        raise TypeError(f"Unsupported base module for MoE-LoRA: {type(base).__name__}")

    @property
    def transient_active(self) -> bool:
        return self.transient_a is not None and self.transient_b is not None

    def start_transient_probe(self) -> None:
        ta = nn.Parameter(torch.zeros(self.rank, self.in_features))
        tb = nn.Parameter(torch.zeros(self.out_features, self.rank))
        nn.init.kaiming_uniform_(ta, a=5**0.5)
        nn.init.zeros_(tb)
        self.transient_a = ta
        self.transient_b = tb

    def stop_transient_probe(self) -> None:
        self.register_parameter("transient_a", None)
        self.register_parameter("transient_b", None)

    def transient_parameters(self) -> list[nn.Parameter]:
        if not self.transient_active:
            return []
        return [self.transient_a, self.transient_b]

    def _to_x(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Move tensor t to same device+dtype as x (handles multi-GPU splits)."""
        return t.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        original_shape = x.shape
        flat_x = x.reshape(-1, original_shape[-1])
        self._maybe_capture_input(flat_x)

        if self.transient_active:
            mixed = self._transient_output(flat_x).reshape(*original_shape[:-1], -1)
            self.last_aux_loss = None
            self.last_router_entropy = None
            return base_out + mixed

        dropped = self.dropout(flat_x)

        # Route: move router_weight to x's device inline (Eq. 3)
        logits = F.linear(flat_x, self._to_x(self.router_weight, flat_x))
        biased_logits = logits + self._to_x(self.cp_bias, flat_x)
        top_values, top_indices = torch.topk(biased_logits, k=self.top_k, dim=-1)
        top_weights = F.softmax(top_values, dim=-1)

        # Expert outputs (Eq. 4)
        expert_out = self._expert_outputs(dropped)
        chosen = torch.gather(
            expert_out,
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, expert_out.shape[-1]),
        )
        mixed = (chosen * top_weights.unsqueeze(-1)).sum(dim=1)
        mixed = mixed.reshape(*original_shape[:-1], -1)

        self._record_router_metrics(logits, top_indices)
        return base_out + mixed

    def _expert_outputs(self, flat_x: torch.Tensor) -> torch.Tensor:
        lora_a = self._to_x(self.lora_a, flat_x)
        lora_b = self._to_x(self.lora_b, flat_x)
        h = torch.einsum("bi,eri->ber", flat_x, lora_a)
        return torch.einsum("ber,eor->beo", h, lora_b) * self.scaling

    def _transient_output(self, flat_x: torch.Tensor) -> torch.Tensor:
        if not self.transient_active:
            raise RuntimeError("Transient probe is not active")
        ta = self._to_x(self.transient_a, flat_x)
        tb = self._to_x(self.transient_b, flat_x)
        return torch.matmul(torch.matmul(flat_x, ta.t()), tb.t()) * self.scaling

    # ------------------------------------------------------------------ #
    # Input capture for CKA                                                #
    # ------------------------------------------------------------------ #
    def start_input_capture(self, max_tokens: int) -> None:
        self._capture_inputs = True
        self._capture_max_tokens = max_tokens
        self._captured_inputs = []

    def stop_input_capture(self) -> torch.Tensor | None:
        self._capture_inputs = False
        if not self._captured_inputs:
            return None
        captured = torch.cat(self._captured_inputs, dim=0)
        self._captured_inputs = []
        return captured

    def _maybe_capture_input(self, flat_x: torch.Tensor) -> None:
        if not self._capture_inputs:
            return
        already = sum(t.shape[0] for t in self._captured_inputs)
        remaining = self._capture_max_tokens - already
        if remaining <= 0:
            return
        self._captured_inputs.append(flat_x[:remaining].detach().cpu())

    def _record_router_metrics(
        self, logits: torch.Tensor, top_indices: torch.Tensor
    ) -> None:
        # Eq. 12: f_i and P_i both from native (un-biased) logits.
        probs = F.softmax(logits, dim=-1)
        native_top = torch.topk(logits, k=self.top_k, dim=-1).indices
        load = (
            F.one_hot(native_top, num_classes=self.num_experts)
            .float()
            .mean(dim=(0, 1))
        )
        prob_mean = probs.mean(dim=0)
        self.last_aux_loss = self.num_experts * torch.sum(load * prob_mean)
        self.last_router_entropy = -(prob_mean * (prob_mean + 1e-8).log()).sum()

    def set_cp_bias(self, scores: torch.Tensor, alpha: float) -> None:
        if scores.numel() != self.num_experts:
            raise ValueError("CP bias score count must match num_experts")
        self.cp_bias.data.copy_(alpha * scores.float())

    def zero_cp_bias(self) -> None:
        self.cp_bias.data.zero_()


# ------------------------------------------------------------------ #
# Model-level helpers                                                   #
# ------------------------------------------------------------------ #

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
    if isinstance(module, nn.Linear):
        return True
    if Conv1D and isinstance(module, Conv1D):
        return True
    return hasattr(module, "in_features") and hasattr(module, "out_features")


def _iter_named_children_with_parent(module: nn.Module):
    for child_name, child in module.named_children():
        yield module, child_name, child
        yield from _iter_named_children_with_parent(child)


def collect_moe_layers(model: nn.Module) -> list[MoELoraLinear]:
    return [m for m in model.modules() if isinstance(m, MoELoraLinear)]


def moe_auxiliary_loss(model: nn.Module, device: torch.device) -> torch.Tensor:
    losses = [
        layer.last_aux_loss
        for layer in collect_moe_layers(model)
        if layer.last_aux_loss is not None
    ]
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack([l.to(device) for l in losses]).mean()