"""Monotonic apparent-radiance proxy and forward display mapping for OCT-GS.

The proxy integrates Planck radiance over a fixed 7.5--13.5 um band.  It is a
monotonic representation used to composite apparent-temperature observations
with the RGB anchor's occupancy weights.  It is *not* a claim that the exact
H30T spectral response is known, and it does not repeat the emissivity,
reflection, atmosphere, or distance correction already used by the TSDK target
decoder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any

import numpy as np
import torch
from torch import nn

from tools.thermal_radiometry.palette_lut import hot_iron_lut, lut_sha256


RADIANCE_FORMULA_VERSION = "uav-tgs-oct-band-radiance-float32-lut-v2"
TARGET_SEMANTICS = "tsdk-referenced-apparent-temperature"
METHOD_SEMANTICS = "measurement-conditioned apparent-radiance rendering"


@dataclass(frozen=True)
class BandRadianceConfig:
    wavelength_min_um: float = 7.5
    wavelength_max_um: float = 13.5
    wavelength_samples: int = 64
    inverse_samples: int = 4096

    def validate(self) -> None:
        if not (0.0 < self.wavelength_min_um < self.wavelength_max_um):
            raise ValueError("invalid wavelength interval")
        if int(self.wavelength_samples) < 8:
            raise ValueError("wavelength_samples must be at least 8")
        if int(self.inverse_samples) < 256:
            raise ValueError("inverse_samples must be at least 256")


def _band_radiance(
    temperature_c: torch.Tensor,
    config: BandRadianceConfig,
) -> torch.Tensor:
    """Integrate spectral radiance over the configured band.

    Constants are SI and the result is proportional to W m^-2 sr^-1.  Only its
    monotonicity and relative scale are used by OCT-GS.
    """

    config.validate()
    kelvin = temperature_c + temperature_c.new_tensor(273.15)
    # Avoid a device synchronization in the per-view CUDA hot path.  OCT field
    # temperatures are bounded by construction; CPU utilities retain explicit
    # validation and training-level finite checks catch catastrophic GPU faults.
    if not kelvin.is_cuda:
        if not bool(torch.isfinite(kelvin).all()):
            raise ValueError("temperature contains NaN or Inf")
        if bool((kelvin <= 0.0).any()):
            raise ValueError("temperature must be above absolute zero")
    wavelengths = torch.linspace(
        config.wavelength_min_um * 1e-6,
        config.wavelength_max_um * 1e-6,
        int(config.wavelength_samples),
        device=temperature_c.device,
        dtype=temperature_c.dtype,
    )
    h = temperature_c.new_tensor(6.62607015e-34)
    c = temperature_c.new_tensor(299792458.0)
    k = temperature_c.new_tensor(1.380649e-23)
    wavelength = wavelengths.view(*([1] * temperature_c.ndim), -1)
    temp = kelvin.unsqueeze(-1)
    exponent = h * c / (wavelength * k * temp)
    spectral = (2.0 * h * c.square()) / (
        wavelength.pow(5) * torch.expm1(exponent)
    )
    return torch.trapezoid(spectral, wavelengths, dim=-1)


class BandRadianceProxy(nn.Module):
    """Differentiable forward proxy plus piecewise-linear monotonic inverse."""

    def __init__(
        self,
        tmin_c: float,
        tmax_c: float,
        config: BandRadianceConfig | None = None,
    ) -> None:
        super().__init__()
        self.tmin_c = float(tmin_c)
        self.tmax_c = float(tmax_c)
        if not np.isfinite(self.tmin_c) or not np.isfinite(self.tmax_c):
            raise ValueError("temperature range must be finite")
        if self.tmax_c <= self.tmin_c:
            raise ValueError("tmax_c must be greater than tmin_c")
        self.config = BandRadianceConfig() if config is None else config
        self.config.validate()
        # Planck integration is performed exactly once while constructing the
        # proxy.  The training hot path uses only float32 piecewise-linear LUT
        # interpolation; it never materializes an Nx64 spectral tensor.
        integration_t = torch.linspace(
            self.tmin_c,
            self.tmax_c,
            int(self.config.inverse_samples),
            dtype=torch.float64,
        )
        integration_l = _band_radiance(integration_t, self.config)
        lookup_t = integration_t.to(torch.float32)
        lookup_l = integration_l.to(torch.float32)
        if not bool((lookup_l[1:] > lookup_l[:-1]).all()):
            raise RuntimeError("float32 band-radiance lookup must be strictly monotonic")
        self.register_buffer("lookup_temperature_c", lookup_t, persistent=False)
        self.register_buffer("lookup_radiance", lookup_l, persistent=False)

    def forward(self, temperature_c: torch.Tensor) -> torch.Tensor:
        self._require_float32(temperature_c, "temperature")
        table_t = self.lookup_temperature_c.to(device=temperature_c.device)
        table_l = self.lookup_radiance.to(device=temperature_c.device)
        clipped = temperature_c.clamp(table_t[0], table_t[-1])
        upper = torch.searchsorted(table_t, clipped.contiguous(), right=False)
        upper = upper.clamp(1, table_t.numel() - 1)
        lower = upper - 1
        fraction = (clipped - table_t[lower]) / (
            table_t[upper] - table_t[lower]
        ).clamp_min(torch.finfo(torch.float32).eps)
        return table_l[lower] + fraction * (table_l[upper] - table_l[lower])

    @staticmethod
    def _require_float32(value: torch.Tensor, label: str) -> None:
        if value.dtype != torch.float32:
            raise TypeError(f"OCT-GS v1 {label} must be float32, got {value.dtype}")
        if not value.is_cuda and not bool(torch.isfinite(value).all()):
            raise ValueError(f"OCT-GS v1 {label} contains NaN or Inf")

    def normalize(self, radiance: torch.Tensor) -> torch.Tensor:
        self._require_float32(radiance, "radiance")
        low = self.lookup_radiance[0].to(device=radiance.device)
        high = self.lookup_radiance[-1].to(device=radiance.device)
        return ((radiance - low) / (high - low)).clamp(0.0, 1.0)

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        self._require_float32(normalized, "normalized radiance")
        low = self.lookup_radiance[0].to(device=normalized.device)
        high = self.lookup_radiance[-1].to(device=normalized.device)
        return low + normalized.clamp(0.0, 1.0) * (high - low)

    def inverse(self, radiance: torch.Tensor) -> torch.Tensor:
        """Invert the monotonic lookup with local differentiable interpolation."""

        self._require_float32(radiance, "radiance")
        table_l = self.lookup_radiance.to(device=radiance.device)
        table_t = self.lookup_temperature_c.to(device=radiance.device)
        clipped = radiance.clamp(table_l[0], table_l[-1])
        upper = torch.searchsorted(table_l, clipped.contiguous(), right=False)
        upper = upper.clamp(1, table_l.numel() - 1)
        lower = upper - 1
        l0 = table_l[lower]
        l1 = table_l[upper]
        t0 = table_t[lower]
        t1 = table_t[upper]
        fraction = (clipped - l0) / (l1 - l0).clamp_min(
            torch.finfo(torch.float32).eps
        )
        return t0 + fraction * (t1 - t0)

    def metadata(self) -> dict[str, Any]:
        lut_bytes = (
            self.lookup_temperature_c.detach().cpu().contiguous().numpy().tobytes()
            + self.lookup_radiance.detach().cpu().contiguous().numpy().tobytes()
        )
        return {
            "formula_version": RADIANCE_FORMULA_VERSION,
            "method_semantics": METHOD_SEMANTICS,
            "target_semantics": TARGET_SEMANTICS,
            "temperature_range_c": [self.tmin_c, self.tmax_c],
            "band_config": asdict(self.config),
            "runtime_forward": "piecewise-linear float32 lookup interpolation",
            "runtime_planck_integration": False,
            "numeric_dtype": "float32",
            "radiance_lut_sha256": hashlib.sha256(lut_bytes).hexdigest(),
            "tsdk_environmental_correction_in_renderer": False,
            "reason_correction_is_not_reapplied": (
                "the float Celsius target is already decoded under the fixed "
                "TSDK benchmark assumptions"
            ),
            "absolute_thermometry_claimed": False,
        }


def temperature_to_hot_iron(
    temperature_c: torch.Tensor,
    tmin_c: float,
    tmax_c: float,
    *,
    exact_display: bool = False,
    straight_through: bool = False,
) -> torch.Tensor:
    """Forward-map Celsius to the fixed repository Hot-Iron display palette.

    Linear interpolation is used for differentiable training.  ``exact_display``
    selects an exact LUT entry for reporting.  If ``straight_through`` is also
    true, exact colors are used in the forward pass while gradients follow the
    linear interpolation.
    """

    low, high = float(tmin_c), float(tmax_c)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("invalid temperature range")
    if exact_display and temperature_c.requires_grad and not straight_through:
        # This is allowed for evaluation, but an explicit detach avoids implying
        # that nearest-bin colors provide a useful training gradient.
        temperature_c = temperature_c.detach()
    if temperature_c.dtype != torch.float32:
        raise TypeError(
            f"OCT-GS v1 Hot-Iron input must be float32, got {temperature_c.dtype}"
        )
    lut = torch.as_tensor(
        hot_iron_lut().astype(np.float32) / 255.0,
        device=temperature_c.device,
        dtype=temperature_c.dtype,
    )
    position = ((temperature_c - low) / (high - low)).clamp(0.0, 1.0) * 255.0
    lower = torch.floor(position).to(torch.long)
    upper = (lower + 1).clamp_max(255)
    fraction = (position - lower.to(position.dtype)).unsqueeze(-1)
    linear = lut[lower] + fraction * (lut[upper] - lut[lower])
    if not exact_display:
        return linear
    nearest = lut[torch.round(position).to(torch.long)]
    return linear + (nearest - linear).detach() if straight_through else nearest


def display_metadata() -> dict[str, Any]:
    return {
        "palette": "uav-tgs-hot-iron-v1",
        "lut_sha256_uint8_rgb": lut_sha256(),
        "role": "forward display mapping only",
        "training_mapping": "piecewise-linear LUT interpolation",
        "evaluation_mapping": "nearest exact LUT entry",
    }
