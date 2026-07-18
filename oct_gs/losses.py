"""Direct thermometric and forward-color losses for OCT-GS."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
import torch
import torch.nn.functional as F

from .radiance import BandRadianceProxy, temperature_to_hot_iron


OCT_LOSS_FORMULA_VERSION = "uav-tgs-oct-direct-thermometric-loss-v1"


@dataclass(frozen=True)
class OCTLossWeights:
    thermometric: float = 1.0
    color_l1: float = 1.0
    color_dssim: float = 1.0
    uncertainty_nll: float = 0.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or float(value) < 0.0:
                raise ValueError(f"loss weight {name} must be finite and non-negative")
        if float(self.thermometric) <= 0.0:
            raise ValueError("thermometric loss weight must be positive")
        if float(self.uncertainty_nll) != 0.0:
            raise ValueError("OCT-GS v1 explicitly disables uncertainty/NLL")

    def to_dict(self) -> dict[str, float]:
        self.validate()
        return {key: float(value) for key, value in asdict(self).items()}


def _as_bchw(value: torch.Tensor, channels: int) -> tuple[torch.Tensor, bool]:
    if value.ndim == 2 and channels == 1:
        return value[None, None], True
    if value.ndim == 3:
        if value.shape[0] != channels:
            raise ValueError(f"expected {channels} channels, got shape {tuple(value.shape)}")
        return value.unsqueeze(0), True
    if value.ndim == 4 and value.shape[1] == channels:
        return value, False
    raise ValueError(f"unsupported tensor shape {tuple(value.shape)} for {channels} channels")


def _mask_b1hw(mask: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones(
            (reference.shape[0], 1, reference.shape[2], reference.shape[3]),
            device=reference.device,
            dtype=reference.dtype,
        )
    if mask.dtype == torch.bool:
        value = mask.to(device=reference.device, dtype=torch.float32)
    elif mask.dtype == torch.float32:
        value = mask.to(device=reference.device)
    else:
        raise TypeError("OCT support mask must be bool or float32")
    if not value.is_cuda and not bool(torch.isfinite(value).all()):
        raise ValueError("OCT support mask contains NaN or Inf")
    if not value.is_cuda and (
        bool((value < 0.0).any()) or bool((value > 1.0).any())
    ):
        raise ValueError("OCT support mask must lie in [0,1]")
    if value.ndim == 2:
        value = value[None, None]
    elif value.ndim == 3:
        value = value.unsqueeze(1) if value.shape[0] == reference.shape[0] else value.unsqueeze(0)
    elif value.ndim != 4:
        raise ValueError("mask must be HW, 1HW, BHW, or B1HW")
    if value.shape[1] != 1:
        value = value[:, :1]
    if value.shape[-2:] != reference.shape[-2:]:
        value = F.interpolate(value, size=reference.shape[-2:], mode="nearest")
    if value.shape[0] == 1 and reference.shape[0] > 1:
        value = value.expand(reference.shape[0], -1, -1, -1)
    if value.shape[0] != reference.shape[0]:
        raise ValueError("mask batch size does not match prediction")
    return value.clamp(0.0, 1.0)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand(value.shape[0], value.shape[1], *value.shape[-2:])
    return (value * expanded).sum() / expanded.sum().clamp_min(1e-12)


def _masked_ssim(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    channels = int(prediction.shape[1])
    coords = torch.arange(11, device=prediction.device, dtype=prediction.dtype) - 5.0
    kernel_1d = torch.exp(-(coords.square()) / (2.0 * 1.5**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).view(1, 1, 11, 11)
    kernel = kernel.expand(channels, 1, 11, 11).contiguous()
    expanded_mask = mask.expand(-1, channels, -1, -1)
    local_weight = F.conv2d(expanded_mask, kernel, padding=5, groups=channels)
    denominator = local_weight.clamp_min(1e-12)
    # Renormalized masked moments prevent invalid pixels (including support
    # boundaries) from entering the SSIM neighborhood statistics.
    mu_pred = F.conv2d(prediction * expanded_mask, kernel, padding=5, groups=channels) / denominator
    mu_target = F.conv2d(target * expanded_mask, kernel, padding=5, groups=channels) / denominator
    pred_sq = mu_pred.square()
    target_sq = mu_target.square()
    pred_target = mu_pred * mu_target
    var_pred = F.conv2d(prediction.square() * expanded_mask, kernel, padding=5, groups=channels) / denominator - pred_sq
    var_target = F.conv2d(target.square() * expanded_mask, kernel, padding=5, groups=channels) / denominator - target_sq
    covariance = F.conv2d(prediction * target * expanded_mask, kernel, padding=5, groups=channels) / denominator - pred_target
    c1, c2 = 0.01**2, 0.03**2
    score = ((2.0 * pred_target + c1) * (2.0 * covariance + c2)) / (
        (pred_sq + target_sq + c1) * (var_pred + var_target + c2)
    ).clamp_min(1e-12)
    local_valid = mask * (local_weight[:, :1] > 1e-12).to(mask.dtype)
    return _masked_mean(score, local_valid)


def _strict_float32_temperature(value: torch.Tensor, label: str) -> torch.Tensor:
    if value.dtype != torch.float32:
        raise TypeError(f"{label} must be float32 Celsius, got {value.dtype}")
    if not value.is_cuda and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} contains NaN or Inf")
    return value


def _strict_color(value: torch.Tensor, label: str, *, prediction: bool) -> torch.Tensor:
    if value.dtype == torch.uint8:
        if prediction:
            raise TypeError("OCT prediction color must be differentiable float32")
        value = value.to(torch.float32) / 255.0
    elif value.dtype != torch.float32:
        raise TypeError(f"{label} must be float32 [0,1] or explicit uint8 target")
    if not value.is_cuda and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} contains NaN or Inf")
    if not value.is_cuda and (
        bool((value < 0.0).any()) or bool((value > 1.0).any())
    ):
        raise ValueError(f"{label} must lie in [0,1]")
    return value


def oct_rendering_loss(
    prediction_temperature_c: torch.Tensor,
    target_temperature_c: torch.Tensor,
    prediction_hot_iron: torch.Tensor,
    *,
    radiance_proxy: BandRadianceProxy,
    target_hot_iron: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    weights: OCTLossWeights | None = None,
    thermometric_domain: str = "celsius",
    charbonnier_eps_c: float = 0.05,
    charbonnier_eps_radiance: float = 1e-3,
    prediction_log_uncertainty_c: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | str | dict[str, float]]:
    """Compute direct target-domain and forward-display losses.

    This function never inverts Hot-Iron.  The thermometric target is the
    float32 TSDK-referenced Celsius map, while Hot-Iron is only generated in the
    forward direction for the appearance term.
    """

    resolved_weights = OCTLossWeights() if weights is None else weights
    resolved_weights.validate()
    if thermometric_domain not in ("celsius", "radiance"):
        raise ValueError("thermometric_domain must be 'celsius' or 'radiance'")
    pred_t, _ = _as_bchw(
        _strict_float32_temperature(prediction_temperature_c, "prediction temperature"), 1
    )
    radiance_proxy.to(device=pred_t.device, dtype=pred_t.dtype)
    target_t, _ = _as_bchw(
        _strict_float32_temperature(target_temperature_c, "target temperature").to(
            device=pred_t.device
        ), 1
    )
    if pred_t.shape != target_t.shape:
        raise ValueError(f"temperature shapes differ: {pred_t.shape} vs {target_t.shape}")
    pred_rgb, _ = _as_bchw(
        _strict_color(prediction_hot_iron, "prediction Hot-Iron", prediction=True), 3
    )
    if pred_rgb.shape[-2:] != pred_t.shape[-2:] or pred_rgb.shape[0] != pred_t.shape[0]:
        raise ValueError("Hot-Iron prediction and temperature prediction differ in shape")
    if target_hot_iron is None:
        target_rgb_hwc = temperature_to_hot_iron(
            target_t[:, 0],
            radiance_proxy.tmin_c,
            radiance_proxy.tmax_c,
            exact_display=False,
        )
        target_rgb = target_rgb_hwc.movedim(-1, 1)
    else:
        normalized_target = _strict_color(
            target_hot_iron, "target Hot-Iron", prediction=False
        ).to(device=pred_rgb.device)
        target_rgb, _ = _as_bchw(normalized_target, 3)
    if target_rgb.shape != pred_rgb.shape:
        raise ValueError(f"color shapes differ: {pred_rgb.shape} vs {target_rgb.shape}")
    valid = _mask_b1hw(mask, pred_t)

    if thermometric_domain == "celsius":
        thermo_error = pred_t - target_t
        thermo_eps = float(charbonnier_eps_c)
    else:
        pred_r = radiance_proxy.normalize(radiance_proxy(pred_t))
        target_r = radiance_proxy.normalize(radiance_proxy(target_t))
        thermo_error = pred_r - target_r
        thermo_eps = float(charbonnier_eps_radiance)
    if not np.isfinite(thermo_eps) or thermo_eps <= 0.0:
        raise ValueError("Charbonnier epsilon must be finite and positive")
    thermometric = _masked_mean(
        torch.sqrt(thermo_error.square() + thermo_eps**2), valid
    )
    color_l1 = _masked_mean(torch.abs(pred_rgb - target_rgb), valid)
    color_dssim = 1.0 - _masked_ssim(pred_rgb, target_rgb, valid)

    if prediction_log_uncertainty_c is not None:
        raise ValueError("OCT-GS v1 explicitly disables uncertainty/NLL")

    total = (
        float(resolved_weights.thermometric) * thermometric
        + float(resolved_weights.color_l1) * color_l1
        + float(resolved_weights.color_dssim) * color_dssim
    )
    return {
        "total": total,
        "thermometric": thermometric,
        "color_l1": color_l1,
        "color_dssim": color_dssim,
        "uncertainty_nll": pred_t.new_zeros(()),
        "thermometric_domain": thermometric_domain,
        "weights": resolved_weights.to_dict(),
        "formula_version": OCT_LOSS_FORMULA_VERSION,
        "valid_fraction": valid.mean(),
    }
