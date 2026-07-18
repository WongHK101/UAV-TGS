"""Occupancy-Conserving Thermometric Gaussian Splatting (OCT-GS).

OCT-GS is deliberately implemented as a sidecar to the RGB Gaussian anchor.
It owns thermometric parameters only; geometry, opacity, and topology stay in
the anchor and are fingerprinted as immutable protocol inputs.
"""

from .field import OCTConfig, OCTGaussianField, build_oct_optimizer
from .formal import FormalOCTBinding, FormalOCTTargetStore, build_formal_binding
from .losses import OCTLossWeights, oct_rendering_loss
from .protocol import (
    BuildingGradientCalibrator,
    OCTStageCostTracker,
    capture_occupancy_snapshot,
    inspect_oct_checkpoint,
    load_frozen_calibration,
    load_oct_checkpoint,
    load_oct_protocol_manifest,
    restore_oct_optimizer_state,
    save_oct_checkpoint,
    validate_training_source_provenance,
    verify_oct_field_finite,
    verify_oct_post_step_finite,
    verify_occupancy_snapshot,
    write_oct_protocol_manifest,
)
from .radiance import BandRadianceProxy, temperature_to_hot_iron
from .rendering import OCTRendererContext, render_oct

__all__ = [
    "BandRadianceProxy",
    "BuildingGradientCalibrator",
    "FormalOCTBinding",
    "FormalOCTTargetStore",
    "OCTConfig",
    "OCTGaussianField",
    "OCTLossWeights",
    "OCTStageCostTracker",
    "build_oct_optimizer",
    "build_formal_binding",
    "capture_occupancy_snapshot",
    "inspect_oct_checkpoint",
    "load_frozen_calibration",
    "load_oct_checkpoint",
    "load_oct_protocol_manifest",
    "oct_rendering_loss",
    "save_oct_checkpoint",
    "temperature_to_hot_iron",
    "OCTRendererContext",
    "render_oct",
    "restore_oct_optimizer_state",
    "verify_occupancy_snapshot",
    "validate_training_source_provenance",
    "verify_oct_field_finite",
    "verify_oct_post_step_finite",
    "write_oct_protocol_manifest",
]
