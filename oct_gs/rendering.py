"""Anchor-bound OCT rendering with shared, immutable RGB occupancy."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F

from .field import OCTGaussianField
from .radiance import BandRadianceProxy, temperature_to_hot_iron


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Device-independent [w,x,y,z] quaternion conversion."""

    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError("quaternion must have shape Nx4")
    if quaternion.dtype != torch.float32:
        raise TypeError("OCT-GS v1 anchor rotation must be float32")
    q = F.normalize(quaternion, dim=1, eps=1e-12)
    w, x, y, z = q.unbind(dim=1)
    matrix = torch.empty((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)
    matrix[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[:, 0, 1] = 2.0 * (x * y - w * z)
    matrix[:, 0, 2] = 2.0 * (x * z + w * y)
    matrix[:, 1, 0] = 2.0 * (x * y + w * z)
    matrix[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[:, 1, 2] = 2.0 * (y * z - w * x)
    matrix[:, 2, 0] = 2.0 * (x * z - w * y)
    matrix[:, 2, 1] = 2.0 * (y * z + w * x)
    matrix[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


class FrozenGaussianView:
    """Cloned activated occupancy tensors bound to one RGB anchor state."""

    def __init__(self, anchor: Any) -> None:
        self._xyz = anchor.get_xyz.detach().clone()
        self._scaling = anchor.get_scaling.detach().clone()
        self._rotation = anchor.get_rotation.detach().clone()
        self._opacity = anchor.get_opacity.detach().clone()
        self.active_sh_degree = int(getattr(anchor, "active_sh_degree", 0))
        self.max_sh_degree = int(getattr(anchor, "max_sh_degree", 0))
        counts = {
            int(value.shape[0])
            for value in (self._xyz, self._scaling, self._rotation, self._opacity)
        }
        if len(counts) != 1:
            raise ValueError("anchor occupancy tensors have inconsistent topology")
        for label, value in (
            ("xyz", self._xyz),
            ("scaling", self._scaling),
            ("rotation", self._rotation),
            ("opacity", self._opacity),
        ):
            if value.dtype != torch.float32:
                raise TypeError(f"OCT-GS v1 anchor {label} must be float32")
            if not value.is_cuda and not bool(torch.isfinite(value).all()):
                raise ValueError(f"anchor {label} contains NaN or Inf")

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_scaling(self) -> torch.Tensor:
        return self._scaling

    @property
    def get_rotation(self) -> torch.Tensor:
        return self._rotation

    @property
    def get_opacity(self) -> torch.Tensor:
        return self._opacity

    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        rotation = quaternion_to_matrix(self._rotation)
        diagonal = torch.diag_embed(self._scaling * float(scaling_modifier))
        transform = rotation @ diagonal
        covariance = transform @ transform.transpose(1, 2)
        return torch.stack(
            (
                covariance[:, 0, 0],
                covariance[:, 0, 1],
                covariance[:, 0, 2],
                covariance[:, 1, 1],
                covariance[:, 1, 2],
                covariance[:, 2, 2],
            ),
            dim=1,
        )

    def get_exposure_from_name(self, image_name: str) -> torch.Tensor:
        raise RuntimeError("OCT rendering forbids RGB exposure adaptation")


def weak_axis_from_anchor(anchor_view: FrozenGaussianView) -> torch.Tensor:
    rotation = quaternion_to_matrix(anchor_view.get_rotation)
    smallest = torch.argmin(anchor_view.get_scaling, dim=1)
    weak = rotation.gather(
        2, smallest[:, None, None].expand(-1, 3, 1)
    ).squeeze(-1)
    return F.normalize(weak, dim=1, eps=1e-12)


def _default_renderer() -> Callable[..., dict[str, torch.Tensor]]:
    # Lazy import keeps CPU protocol/tests independent of the CUDA extension.
    from gaussian_renderer import render

    return render


class OCTRendererContext:
    """Cache one anchor clone, weak axis, and the formal radiance proxy."""

    def __init__(
        self,
        anchor: Any,
        radiance_proxy: BandRadianceProxy,
        *,
        renderer: Callable[..., dict[str, torch.Tensor]] | None = None,
    ) -> None:
        from .protocol import capture_occupancy_snapshot

        self.anchor = anchor
        self.anchor_snapshot = capture_occupancy_snapshot(anchor)
        self.frozen = FrozenGaussianView(anchor)
        self.weak_axis = weak_axis_from_anchor(self.frozen).detach()
        self.proxy = radiance_proxy.to(device=self.frozen.get_xyz.device)
        if self.proxy.lookup_radiance.dtype != torch.float32:
            raise TypeError("OCT-GS v1 proxy LUT must remain float32")
        self.renderer = _default_renderer() if renderer is None else renderer
        self.alpha_backend = "native_background_composition"

    def verify_anchor_unchanged(self) -> Mapping[str, Any]:
        from .protocol import verify_occupancy_snapshot

        return verify_occupancy_snapshot(self.anchor, self.anchor_snapshot)

    def render(
        self,
        viewpoint_camera: Any,
        field: OCTGaussianField,
        pipe: Any,
        *,
        background_temperature_c: float | None = None,
        exact_display: bool = False,
    ) -> dict[str, Any]:
        if field.num_gaussians != int(self.frozen.get_xyz.shape[0]):
            raise ValueError("field/anchor topology mismatch")
        if field.raw_base_temperature.device != self.frozen.get_xyz.device:
            raise ValueError("field and anchor must be on the same device")
        if field.raw_base_temperature.dtype != torch.float32:
            raise TypeError("OCT-GS v1 field must be float32")
        center = viewpoint_camera.camera_center.to(
            device=self.frozen.get_xyz.device, dtype=torch.float32
        ).reshape(1, 3)
        field_values = field(center - self.frozen.get_xyz, self.weak_axis)
        gaussian_temperature = field_values["temperature_c"]
        assert isinstance(gaussian_temperature, torch.Tensor)
        gaussian_radiance_u = self.proxy.normalize(self.proxy(gaussian_temperature))
        override = gaussian_radiance_u.expand(-1, 3).contiguous()
        background = self.proxy.tmin_c if background_temperature_c is None else float(
            background_temperature_c
        )
        if not self.proxy.tmin_c <= background <= self.proxy.tmax_c:
            raise ValueError("background temperature must lie in the formal scene range")
        background_tensor = gaussian_temperature.new_tensor(background)
        background_u = self.proxy.normalize(self.proxy(background_tensor))
        # Gaussian splatting is affine in per-Gaussian color and background:
        # render = sum_i(w_i * u_i) + T_end * u_bg. Supplying u_bg directly
        # produces the exact occupancy-conserving radiance composite in one
        # differentiable legacy raster pass.
        radiance_background = background_u.expand(3).contiguous()
        radiance_package = self.renderer(
            viewpoint_camera,
            self.frozen,
            pipe,
            radiance_background,
            override_color=override,
            use_trained_exp=False,
        )
        composed_u = radiance_package["render"].mean(dim=0).clamp(0.0, 1.0)
        composed_radiance = self.proxy.denormalize(composed_u)
        rendered_temperature = self.proxy.inverse(composed_radiance)
        hot_iron = temperature_to_hot_iron(
            rendered_temperature,
            self.proxy.tmin_c,
            self.proxy.tmax_c,
            exact_display=exact_display,
        ).movedim(-1, 0)
        return {
            "temperature_c": rendered_temperature,
            "radiance": composed_radiance,
            "hot_iron": hot_iron,
            "alpha": None,
            "log_uncertainty_c": None,
            "gaussian_temperature_c": gaussian_temperature,
            "gaussian_view_residual_c": field_values["view_residual_c"],
            "gaussian_residual_basis": field_values["residual_basis"],
            "depth": radiance_package.get("depth"),
            "radii": radiance_package.get("radii"),
            "visibility_filter": radiance_package.get("visibility_filter"),
            "viewspace_points": radiance_package.get("viewspace_points"),
            "raster_passes": 1,
            "alpha_backend": self.alpha_backend,
            "occupancy_source": "anchor-bound cached RGB xyz/scaling/rotation/opacity",
        }


def render_oct(
    viewpoint_camera: Any,
    anchor: Any,
    field: OCTGaussianField,
    pipe: Any,
    radiance_proxy: BandRadianceProxy,
    *,
    background_temperature_c: float | None = None,
    renderer: Callable[..., dict[str, torch.Tensor]] | None = None,
    exact_display: bool = False,
) -> dict[str, Any]:
    """Compatibility wrapper; formal loops should reuse ``OCTRendererContext``."""

    context = OCTRendererContext(anchor, radiance_proxy, renderer=renderer)
    return context.render(
        viewpoint_camera,
        field,
        pipe,
        background_temperature_c=background_temperature_c,
        exact_display=exact_display,
    )
