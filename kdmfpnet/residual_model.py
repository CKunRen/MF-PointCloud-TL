"""Residual multi-fidelity field model and layer-freezing utilities.

The method represents the fine-grid prediction as a frozen (or partly frozen)
low-fidelity prior plus a learned correction,

    f(P, c) = f_prior(P, c) + delta(P, c),

where ``f_prior`` is the G2-pretrained dual-output network and ``delta`` is a
correction network whose final layer is zero-initialised, so fine-tuning starts
exactly from the coarse-grid prediction (see the paper's Eq. (residual)).

A single network emits a multi-channel field: channel 0 is Cp (primary), and
channels 1..3 are the wall-velocity vector (auxiliary).
"""
from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from .kd_model import PressureKDRegressor

FIELD_CHANNELS = 4  # Cp + 3 velocity components


def build_field_model(
    knn_k: int = 24,
    dropout: float = 0.14,
    point_channels: tuple[int, ...] = (64, 128, 256),
    condition_hidden: tuple[int, ...] = (32, 64),
    geometry_hidden: tuple[int, ...] = (256,),
    decoder_hidden: tuple[int, ...] = (256, 128, 64),
    residual_hidden: tuple[int, ...] = (128, 64),
    use_edge_conv: bool = True,
    out_channels: int = FIELD_CHANNELS,
    fourier_bands: int = 4,
    extra_channels: int = 0,
) -> PressureKDRegressor:
    return PressureKDRegressor(
        point_channels=list(point_channels),
        condition_hidden=list(condition_hidden),
        geometry_hidden=list(geometry_hidden),
        decoder_hidden=list(decoder_hidden),
        dropout=dropout,
        fourier_bands=fourier_bands,
        use_residual_decoder=True,
        residual_hidden=list(residual_hidden),
        distill_dim=256,
        knn_k=knn_k,
        use_edge_conv=use_edge_conv,
        out_channels=out_channels,
        extra_channels=extra_channels,
    )


def _zero_final_conv(decoder: nn.Module) -> None:
    """Zero the last Conv1d (weight and bias) of a PointwiseDecoder."""
    last_conv = None
    for module in decoder.modules():
        if isinstance(module, nn.Conv1d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv1d found to zero-initialise")
    nn.init.zeros_(last_conv.weight)
    if last_conv.bias is not None:
        nn.init.zeros_(last_conv.bias)


def zero_init_correction(model: PressureKDRegressor) -> None:
    """Make the correction branch output exactly zero at initialisation."""
    _zero_final_conv(model.decoder)
    if getattr(model, "residual_decoder", None) is not None:
        _zero_final_conv(model.residual_decoder)


class ResidualFieldModel(nn.Module):
    """f_prior + delta, with the prior frozen or partially frozen.

    ``forward_features`` returns the combined field plus the prior's own field
    for diagnostics. Both sub-models must emit the same number of channels.
    """

    def __init__(self, prior: PressureKDRegressor, correction: PressureKDRegressor) -> None:
        super().__init__()
        if prior.out_channels != correction.out_channels:
            raise ValueError("prior and correction must share out_channels")
        self.prior = prior
        self.correction = correction
        self.out_channels = prior.out_channels

    def forward_features(self, points, condition, mask=None, knn_idx=None, extra=None) -> dict:
        # The prior keeps its original input width; the extra per-point side
        # features (local normal + shape descriptors) go to the correction branch
        # only, so no pretrained prior has to be rebuilt.
        prior_features = self.prior.forward_features(points, condition, mask=mask, knn_idx=knn_idx)
        correction_features = self.correction.forward_features(
            points, condition, mask=mask, knn_idx=knn_idx, extra=extra
        )
        field = prior_features["field"] + correction_features["field"]
        result = dict(prior_features)
        result["field"] = field
        result["cp"] = field[:, :, 0:1]
        result["prior_field"] = prior_features["field"]
        result["prior_cp"] = prior_features["cp"]
        result["correction_field"] = correction_features["field"]
        if self.out_channels >= 4:
            result["velocity"] = field[:, :, 1:4]
        return result

    def forward(self, points, condition, mask=None, knn_idx=None, extra=None) -> torch.Tensor:
        # Primary output is the pressure field, matching PressureKDRegressor.
        return self.forward_features(points, condition, mask=mask, knn_idx=knn_idx, extra=extra)["cp"]

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)


def apply_freezing_strategy(prior: PressureKDRegressor, strategy: str) -> None:
    """Set requires_grad on the prior's sub-modules for the residual method and
    its layer-freezing ablations.

    * ``frozen`` -- residual default: the whole prior is fixed, so only the
      correction branch trains (f(P) = f_prior(P) + delta(P)).
    * ``none``   -- full fine-tuning: every prior parameter stays trainable.
    * ``early``  -- freeze the Fourier encoder, both point-encoder MLPs, the
      EdgeConv block and the flight-condition gate (the geometry-driven front
      end), leaving the pooling heads and decoders trainable.
    * ``late``   -- freeze the decoders, leaving the front end trainable.

    The correction branch is never touched here; it always trains.
    """
    strategy = strategy.lower()
    if strategy not in {"frozen", "none", "early", "late"}:
        raise ValueError(f"Unknown freezing strategy: {strategy!r}")

    if strategy == "frozen":
        for parameter in prior.parameters():
            parameter.requires_grad_(False)
        return

    for parameter in prior.parameters():
        parameter.requires_grad_(True)

    early_modules = [
        prior.fourier_encoder,
        prior.point_encoder_early,
        prior.point_encoder_late,
        prior.condition_gate,
    ]
    if getattr(prior, "edge_conv", None) is not None:
        early_modules.append(prior.edge_conv)
    late_modules = [prior.decoder]
    if getattr(prior, "residual_decoder", None) is not None:
        late_modules.append(prior.residual_decoder)

    if strategy == "early":
        for module in early_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    elif strategy == "late":
        for module in late_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
