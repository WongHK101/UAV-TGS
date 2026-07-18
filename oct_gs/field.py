"""Thermometric sidecar parameters for OCT-GS."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


OCT_FIELD_SCHEMA = "uav-tgs-oct-field-v1"
OCT_VARIANTS = ("oct_scalar", "oct_residual")
OCT_NUMERIC_DTYPE = torch.float32


@dataclass(frozen=True)
class OCTConfig:
    num_gaussians: int
    tmin_c: float
    tmax_c: float
    variant: str = "oct_scalar"
    initial_temperature_c: float | None = None
    residual_bound_fraction: float = 0.05
    learn_uncertainty: bool = False

    def validate(self) -> None:
        if int(self.num_gaussians) <= 0:
            raise ValueError("num_gaussians must be positive")
        if self.variant not in OCT_VARIANTS:
            raise ValueError(f"variant must be one of {OCT_VARIANTS}, got {self.variant!r}")
        if not np.isfinite(self.tmin_c) or not np.isfinite(self.tmax_c):
            raise ValueError("temperature range must be finite")
        if float(self.tmax_c) <= float(self.tmin_c):
            raise ValueError("tmax_c must be greater than tmin_c")
        if not (0.0 < float(self.residual_bound_fraction) <= 0.25):
            raise ValueError("residual_bound_fraction must be in (0, 0.25]")
        if bool(self.learn_uncertainty):
            raise ValueError("OCT-GS v1 explicitly disables uncertainty/NLL")
        initial = self.resolved_initial_temperature_c
        if not (float(self.tmin_c) < initial < float(self.tmax_c)):
            raise ValueError("initial temperature must lie strictly inside the scene range")

    @property
    def resolved_initial_temperature_c(self) -> float:
        if self.initial_temperature_c is None:
            return 0.5 * (float(self.tmin_c) + float(self.tmax_c))
        return float(self.initial_temperature_c)

    @property
    def residual_bound_c(self) -> float:
        return float(self.residual_bound_fraction) * (
            float(self.tmax_c) - float(self.tmin_c)
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["residual_bound_c"] = self.residual_bound_c
        payload["resolved_initial_temperature_c"] = self.resolved_initial_temperature_c
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OCTConfig":
        keys = {
            "num_gaussians",
            "tmin_c",
            "tmax_c",
            "variant",
            "initial_temperature_c",
            "residual_bound_fraction",
            "learn_uncertainty",
        }
        config = cls(**{key: payload[key] for key in keys if key in payload})
        config.validate()
        return config


def _inverse_tanh(value: torch.Tensor) -> torch.Tensor:
    eps = 4.0 * torch.finfo(value.dtype).eps
    return torch.atanh(value.clamp(-1.0 + eps, 1.0 - eps))


class OCTGaussianField(nn.Module):
    """Per-Gaussian apparent temperature without thermal geometry/opacity.

    ``oct_residual`` adds one scalar coefficient per Gaussian.  Its basis is
    ``dot(n_weak, v)``: a first-order, one-dimensional directional function
    with zero spherical mean and range [-1, 1].  Consequently the latent
    Celsius residual is strictly bounded and cannot become unconstrained RGB
    spherical harmonics.
    """

    def __init__(self, config: OCTConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        midpoint = 0.5 * (float(config.tmin_c) + float(config.tmax_c))
        half_range = 0.5 * (float(config.tmax_c) - float(config.tmin_c))
        normalized = (config.resolved_initial_temperature_c - midpoint) / half_range
        raw_initial = _inverse_tanh(torch.tensor(normalized, dtype=torch.float32))
        self.raw_base_temperature = nn.Parameter(
            raw_initial.expand(int(config.num_gaussians), 1).clone()
        )
        if config.variant == "oct_residual":
            self.raw_residual_amplitude = nn.Parameter(
                torch.zeros((int(config.num_gaussians), 1), dtype=torch.float32)
            )
        else:
            self.register_parameter("raw_residual_amplitude", None)

    @property
    def num_gaussians(self) -> int:
        return int(self.raw_base_temperature.shape[0])

    def _base_temperature(self) -> torch.Tensor:
        midpoint = self.raw_base_temperature.new_tensor(
            0.5 * (self.config.tmin_c + self.config.tmax_c)
        )
        half_range = self.raw_base_temperature.new_tensor(
            0.5 * (self.config.tmax_c - self.config.tmin_c)
        )
        return midpoint + half_range * torch.tanh(self.raw_base_temperature)

    def forward(
        self,
        view_directions: torch.Tensor | None = None,
        weak_directions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if self.raw_base_temperature.dtype != OCT_NUMERIC_DTYPE:
            raise TypeError("OCT-GS v1 requires float32 field parameters")
        base = self._base_temperature()
        residual = torch.zeros_like(base)
        basis = torch.zeros_like(base)
        if self.config.variant == "oct_residual":
            if view_directions is None or weak_directions is None:
                raise ValueError("oct_residual requires view_directions and weak_directions")
            expected = (self.num_gaussians, 3)
            if tuple(view_directions.shape) != expected or tuple(weak_directions.shape) != expected:
                raise ValueError(
                    f"direction tensors must have shape {expected}, got "
                    f"{tuple(view_directions.shape)} and {tuple(weak_directions.shape)}"
                )
            views = F.normalize(view_directions, dim=-1, eps=1e-12)
            weak = F.normalize(weak_directions, dim=-1, eps=1e-12)
            basis = (views * weak).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
            lower_room = (base - float(self.config.tmin_c)).clamp_min(0.0)
            upper_room = (float(self.config.tmax_c) - base).clamp_min(0.0)
            configured_bound = torch.full_like(base, self.config.residual_bound_c)
            # Shrink the amplitude near the scene endpoints.  This preserves the
            # full base-temperature range while making the *actual* view
            # residual exactly odd/zero-mean on the sphere and guaranteeing the
            # final temperature remains inside [Tmin, Tmax].
            safe_bound = torch.minimum(
                configured_bound, torch.minimum(lower_room, upper_room)
            )
            amplitude = safe_bound * torch.tanh(self.raw_residual_amplitude)
            residual = amplitude * basis
            temperature = base + residual
        else:
            temperature = base
        return {
            "temperature_c": temperature,
            "base_temperature_c": base,
            "latent_residual_c": residual,
            "view_residual_c": residual,
            "residual_basis": basis,
            "log_uncertainty_c": None,
        }

    def protocol_metadata(self) -> dict[str, Any]:
        return {
            "schema": OCT_FIELD_SCHEMA,
            "config": self.config.to_dict(),
            "learned_parameter_names": [name for name, _ in self.named_parameters()],
            "owned_geometry_fields": [],
            "owned_opacity_fields": [],
            "topology_mutation_supported": False,
            "numeric_dtype": "float32",
            "uncertainty_enabled": False,
            "residual_basis": (
                None
                if self.config.variant == "oct_scalar"
                else "dot(frozen_anchor_weak_axis, point_to_camera_unit_direction)"
            ),
            "residual_zero_mean_domain": (
                None if self.config.variant == "oct_scalar" else "unit_sphere"
            ),
        }


def build_oct_optimizer(
    field: OCTGaussianField,
    *,
    temperature_lr: float,
    residual_lr: float | None = None,
    uncertainty_lr: float | None = None,
) -> torch.optim.Adam:
    """Build a field-only Adam; no anchor tensor can enter its parameter groups."""

    if not np.isfinite(temperature_lr) or float(temperature_lr) <= 0.0:
        raise ValueError("temperature_lr must be finite and positive")
    groups: list[dict[str, Any]] = [
        {
            "params": [field.raw_base_temperature],
            "lr": float(temperature_lr),
            "name": "apparent_temperature",
        }
    ]
    if field.raw_residual_amplitude is not None:
        resolved = float(temperature_lr if residual_lr is None else residual_lr)
        if not np.isfinite(resolved) or resolved <= 0.0:
            raise ValueError("residual_lr must be finite and positive")
        groups.append(
            {
                "params": [field.raw_residual_amplitude],
                "lr": resolved,
                "name": "bounded_view_residual",
            }
        )
    elif residual_lr is not None:
        raise ValueError("residual_lr is invalid for oct_scalar")
    if uncertainty_lr is not None:
        raise ValueError("OCT-GS v1 explicitly disables uncertainty/NLL")
    return torch.optim.Adam(groups, lr=0.0, eps=1e-15)


def optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group.get("params", [])
    }


def verify_field_only_optimizer(
    field: OCTGaussianField,
    optimizer: torch.optim.Optimizer,
    forbidden_parameters: Iterable[torch.Tensor] = (),
) -> None:
    expected = {id(parameter) for parameter in field.parameters()}
    actual = optimizer_parameter_ids(optimizer)
    if actual != expected:
        raise RuntimeError("OCT optimizer must contain every and only OCT field parameter")
    forbidden = {id(parameter) for parameter in forbidden_parameters}
    overlap = actual & forbidden
    if overlap:
        raise RuntimeError("OCT optimizer contains frozen anchor parameters")
