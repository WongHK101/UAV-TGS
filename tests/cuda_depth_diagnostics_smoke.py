"""Minimal analytic CUDA smoke test for opt-in rasterizer diagnostics.

Run after rebuilding ``diff-gaussian-rasterization`` in the active environment:

    python tests/cuda_depth_diagnostics_smoke.py
"""

from __future__ import annotations

import torch

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization import _C


def _rasterizer() -> GaussianRasterizer:
    device = torch.device("cuda")
    return GaussianRasterizer(
        GaussianRasterizationSettings(
            image_height=1,
            image_width=1,
            tanfovx=1.0,
            tanfovy=1.0,
            bg=torch.zeros(3, device=device),
            scale_modifier=1.0,
            viewmatrix=torch.eye(4, device=device),
            projmatrix=torch.eye(4, device=device),
            sh_degree=0,
            campos=torch.zeros(3, device=device),
            prefiltered=False,
            debug=True,
            antialiasing=False,
        )
    )


def _render(opacities: list[float], *, diagnostics: bool):
    count = len(opacities)
    means3d = torch.tensor([[0.0, 0.0, 2.0 + 2.0 * index] for index in range(count)], device="cuda")
    means2d = torch.zeros_like(means3d, requires_grad=True)
    opacity = torch.tensor(opacities, device="cuda").reshape(-1, 1)
    colors = torch.eye(3, device="cuda")[:count]
    scales = torch.full((count, 3), 0.1, device="cuda")
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count, device="cuda")
    outputs = _rasterizer()(
        means3d,
        means2d,
        opacity,
        colors_precomp=colors,
        scales=scales,
        rotations=rotations,
        return_diagnostics=diagnostics,
    )
    return outputs, means2d


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    native_args = (
        torch.zeros(3, device="cuda"),
        torch.tensor([[0.0, 0.0, 2.0]], device="cuda"),
        torch.tensor([[1.0, 0.0, 0.0]], device="cuda"),
        torch.tensor([[0.6]], device="cuda"),
        torch.full((1, 3), 0.1, device="cuda"),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
        1.0,
        torch.empty(0),
        torch.eye(4, device="cuda"),
        torch.eye(4, device="cuda"),
        1.0,
        1.0,
        1,
        1,
        torch.empty(0),
        0,
        torch.zeros(3, device="cuda"),
        False,
        False,
        True,
    )
    assert len(native_args) == 20
    assert len(_C.rasterize_gaussians(*native_args)) == 7
    assert len(_C.rasterize_gaussians_diagnostics(*native_args)) == 13
    legacy, _ = _render([0.6, 0.5], diagnostics=False)
    diagnostics, means2d = _render([0.6, 0.5], diagnostics=True)
    assert len(legacy) == 3
    assert len(diagnostics) == 9
    for legacy_tensor, diagnostic_tensor in zip(legacy, diagnostics[:3]):
        assert torch.equal(legacy_tensor, diagnostic_tensor)
    legacy_invdepth, expected, median, maximum, top_index, top_weight, accumulated_opacity = [
        tensor.detach().cpu() for tensor in diagnostics[2:]
    ]
    assert abs(float(legacy_invdepth.item()) - 0.35) < 1e-5
    assert abs(float(expected.item()) - 2.5) < 1e-5
    assert abs(float(median.item()) - 2.0) < 1e-5
    assert abs(float(maximum.item()) - 2.0) < 1e-5
    assert int(top_index.item()) == 0
    assert abs(float(top_weight.item()) - 0.6) < 1e-5
    assert abs(float(accumulated_opacity.item()) - 0.8) < 1e-5
    diagnostics[0].sum().backward()
    assert means2d.grad is not None
    assert not expected.requires_grad
    assert not top_weight.requires_grad

    low_opacity, _ = _render([0.4], diagnostics=True)
    assert abs(float(low_opacity[3].item()) - 2.0) < 1e-5
    assert float(low_opacity[4].item()) == 0.0  # transmittance never reaches 0.5
    assert abs(float(low_opacity[5].item()) - 2.0) < 1e-5
    print("CUDA_DEPTH_DIAGNOSTICS_SMOKE_OK")


if __name__ == "__main__":
    main()
