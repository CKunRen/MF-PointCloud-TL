from __future__ import annotations

import torch
from torch import nn

from .model import MLP, SharedPointMLP, _group_norm_1d
from .pressure_model import FourierFeatureEncoder, PointwiseDecoder


def _resolve_knn_k(k: int, mask: torch.Tensor | None, n_points: int) -> int:
    """Clamp k to (smallest number of valid points in the batch) - 1.

    The -1 reserves a slot for each point's self-match (distance 0, always
    the nearest "neighbour" of itself), which the caller drops before use --
    see _chunked_knn_indices. With batch_size=1 (the KD training default)
    the clamp itself is a no-op since every real sample has >>k points; it
    only matters for the padded-batch test fixtures, where some rows have
    fewer valid points than k.
    """
    if mask is None:
        return max(1, min(k, max(n_points - 1, 1)))
    valid_counts = mask.sum(dim=1)
    min_valid = int(valid_counts.min().item()) if valid_counts.numel() > 0 else n_points
    return max(1, min(k, max(min_valid - 1, 1)))


def _chunked_knn_indices(
    points: torch.Tensor,
    k: int,
    mask: torch.Tensor | None = None,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Return [B, N, k_eff] indices of the k nearest *other* points.

    Computed in query chunks rather than one [B, N, N] distance matrix -- for
    a ~20k-point cloud a full matrix is ~1.6 GB in float32 per sample, which
    is unnecessary given only the top-k per row is ever used (measured: peak
    memory drops to ~0.3 GB with chunk_size=2048, at no speed cost).

    Queries k_eff+1 and drops the closest match (distance 0, i.e. the point
    itself) rather than the naive top-k_eff, to match kd_data.py's
    _compute_knn_indices (used at data-loading time to precompute this same
    neighbourhood so the model doesn't pay for this search every step -- see
    EdgeConvBlock.forward). Both must agree on "self excluded", otherwise
    switching between the precomputed and online path would silently change
    which points enter the edge-feature aggregation. Including self would
    also waste one of the k aggregation slots on a trivial zero-difference
    edge (center - center = 0).
    """
    b, n, _ = points.shape
    k_eff = _resolve_knn_k(k, mask, n)
    idx_chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = points[:, start:end, :]
        dist = torch.cdist(chunk, points)
        if mask is not None:
            dist = dist.masked_fill(~mask.unsqueeze(1), float("inf"))
        idx_chunks.append(dist.topk(k=k_eff + 1, largest=False, dim=-1).indices[:, :, 1:])
    return torch.cat(idx_chunks, dim=1)


def _gather_neighbor_features(feat_channel_last: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather per-point features at neighbour indices.

    feat_channel_last: [B, N, C], idx: [B, N, k] -> returns [B, N, k, C].
    """
    b, n, c = feat_channel_last.shape
    k = idx.shape[-1]
    idx_flat = idx.reshape(b, n * k)
    gathered = torch.gather(feat_channel_last, 1, idx_flat.unsqueeze(-1).expand(-1, -1, c))
    return gathered.reshape(b, n, k, c)


class EdgeConvBlock(nn.Module):
    """DGCNN-style local neighbourhood aggregation (Wang et al. 2019 EdgeConv).

    The base PressureKDRegressor is a pure PointNet: every point's feature is
    computed independently by 1x1 convs, and the only cross-point information
    is a single global max/mean/top-k pooled summary shared by all points.
    That has no notion of "what do my geometric neighbours look like", which
    is exactly the information needed to resolve local pressure peaks,
    shock-affected regions, and fin-root junctions -- the regions the KD
    literature review (see 开发日志.md, PointNet++/DGCNN/PointConv references)
    consistently flags as the failure mode of plain PointNet baselines. This
    block computes, for each point, edge features to its k nearest geometric
    neighbours and max-pools them into a per-point local-context feature.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden: int, k: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.k = int(k)
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels * 2, hidden, kernel_size=1, bias=False),
            _group_norm_1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            _group_norm_1d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        points: torch.Tensor,
        feat: torch.Tensor,
        mask: torch.Tensor | None = None,
        knn_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """points: [B, N, 3] coordinates used ONLY for the fallback online
        search below when knn_idx is None. Must be physical (unstandardized)
        coordinates -- G2/G4 each fit a per-axis Standardizer independently,
        and axis stds differ by ~9x (x vs y/z, see 开发日志.md), so nearest
        neighbours computed in standardized space are not the same points as
        in physical space. The production training path (kd_train.py's
        _run_epoch) always supplies knn_idx (precomputed once in physical
        space by kd_data.py at data-loading time), so `points` here ends up
        being the standardized `points_t` -- correct for the Fourier
        coordinate features elsewhere in the model, but NOT a valid distance
        metric for the fallback path. Only rely on the fallback (knn_idx=None)
        with genuinely physical coordinates.

        feat: [B, C, N] channel-first per-point features to aggregate.
        knn_idx: optional precomputed [B, N, k'] neighbour indices computed in
        physical coordinate space (see kd_data.py's _compute_knn_indices) --
        when given, the (currently ~74ms-at-N=20000, O(N^2)) brute-force GPU
        search below is skipped entirely. k' may differ from self.k (e.g. it
        is fixed at data-loading time); only the first self.k columns are
        used, clamped if k' < self.k.
        Returns [B, out_channels, N].
        """
        if knn_idx is not None:
            idx = knn_idx[:, :, : self.k]
            if idx.shape[-1] < self.k:
                pad = idx[:, :, -1:].expand(-1, -1, self.k - idx.shape[-1])
                idx = torch.cat([idx, pad], dim=-1)
        else:
            idx = _chunked_knn_indices(points, k=self.k, mask=mask)
        feat_last = feat.transpose(1, 2)
        neighbor_feat = _gather_neighbor_features(feat_last, idx)
        center_feat = feat_last.unsqueeze(2).expand(-1, -1, neighbor_feat.shape[2], -1)
        edge_feat = torch.cat([center_feat, neighbor_feat - center_feat], dim=-1)
        edge_feat = edge_feat.permute(0, 3, 1, 2)
        out = self.net(edge_feat)
        return out.max(dim=-1).values


def _masked_max_pool(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = mask.unsqueeze(1)
    masked = features.masked_fill(~mask_expanded, torch.finfo(features.dtype).min)
    pooled = masked.max(dim=2).values
    return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))


def _masked_mean_pool(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(1).to(features.dtype)
    total = (features * mask_f).sum(dim=2)
    count = mask_f.sum(dim=2).clamp_min(1.0)
    return total / count


def _masked_topk_mean_pool(features: torch.Tensor, mask: torch.Tensor, topk_fraction: float = 0.1) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError(f"Expected features with shape [B, C, N], got {tuple(features.shape)}")
    if mask.ndim != 2:
        raise ValueError(f"Expected mask with shape [B, N], got {tuple(mask.shape)}")

    scores = torch.linalg.norm(features, dim=1)
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    count = max(1, int(round(features.shape[2] * float(topk_fraction))))
    count = min(count, features.shape[2])
    indices = torch.topk(scores, k=count, dim=1, largest=True, sorted=False).indices
    gathered = torch.gather(
        features,
        2,
        indices.unsqueeze(1).expand(-1, features.shape[1], -1),
    )
    selected_mask = torch.gather(mask, 1, indices).unsqueeze(1)
    gathered = gathered * selected_mask.to(features.dtype)
    denom = selected_mask.sum(dim=2).clamp_min(1.0)
    return gathered.sum(dim=2) / denom


class PressureKDRegressor(nn.Module):
    def __init__(
        self,
        point_channels: list[int],
        condition_hidden: list[int],
        geometry_hidden: list[int],
        decoder_hidden: list[int],
        dropout: float = 0.08,
        fourier_bands: int = 4,
        use_residual_decoder: bool = True,
        residual_hidden: list[int] | None = None,
        distill_dim: int = 256,
        knn_k: int = 16,
        point_feat_distill_channels: int | None = None,
        use_edge_conv: bool = True,
        out_channels: int = 1,
        extra_channels: int = 0,
    ) -> None:
        super().__init__()
        residual_hidden = residual_hidden or [128, 64]
        self.use_edge_conv = bool(use_edge_conv)
        self.out_channels = int(out_channels)
        # Optional per-point side features (e.g. local surface normal + shape
        # descriptors) concatenated to the Fourier-encoded coordinates. They let
        # the network see *where the surface is sharp* instead of having to infer
        # it from coordinates alone -- the region where the field error is
        # concentrated (the sharpest 10% of nodes carry ~69% of the total SSE).
        self.extra_channels = int(extra_channels)
        self.fourier_encoder = FourierFeatureEncoder(fourier_bands)
        encoded_point_channels = self.fourier_encoder.out_channels + self.extra_channels
        # point_encoder is split so EdgeConv sits after the *first* (smallest-
        # channel) layer rather than the last (largest-channel) one. The
        # Conv2d over the materialised [B, 2C, N, k] edge tensor scales with
        # C: at the teacher's final width (C=384) it alone measured ~250ms
        # for a single forward+backward on a 20k-point cloud -- more than the
        # rest of the network combined, and enough to turn a ~55min training
        # run into several hours. At the first layer's width (C=96 for the
        # teacher, 64 for the student) the same block measured ~54/27ms,
        # cheap enough to keep. This does mean the later per-point MLP layers
        # only see coordinates + first-layer-plus-local-context features, not
        # a second round of local aggregation at higher channel counts --
        # that's the deliberate cost/accuracy tradeoff being made here.
        self.point_encoder_early = SharedPointMLP(encoded_point_channels, point_channels[:1])
        self.point_encoder_late = SharedPointMLP(point_channels[0], point_channels[1:])
        # use_edge_conv=False collapses the model to a plain PointNet (no
        # local-neighbourhood aggregation) -- built only for the ablation
        # comparing against the EdgeConv-equipped production architecture
        # (see 开发日志.md 33 for the original k=16/k=24 EdgeConv sweep).
        self.edge_conv = (
            EdgeConvBlock(
                in_channels=point_channels[0],
                out_channels=point_channels[0],
                hidden=point_channels[0],
                k=knn_k,
                dropout=dropout,
            )
            if self.use_edge_conv
            else None
        )
        self.geometry_fuse = MLP(point_channels[-1] * 2, geometry_hidden, distill_dim, dropout=dropout)
        self.hard_geometry_fuse = MLP(point_channels[-1] * 3, geometry_hidden, distill_dim, dropout=dropout)
        self.condition_gate = MLP(2, condition_hidden, point_channels[-1], dropout=dropout)
        self.condition_encoder = MLP(2, condition_hidden, condition_hidden[-1], dropout=dropout)
        decoder_in = point_channels[-1] + distill_dim + condition_hidden[-1] + encoded_point_channels
        self.decoder = PointwiseDecoder(decoder_in, decoder_hidden, dropout=dropout, out_channels=self.out_channels)
        self.use_residual_decoder = bool(use_residual_decoder)
        if self.use_residual_decoder:
            residual_in = encoded_point_channels + point_channels[-1] + condition_hidden[-1]
            self.residual_decoder = PointwiseDecoder(residual_in, residual_hidden, dropout=dropout, out_channels=self.out_channels)
        else:
            self.residual_decoder = None
        # Optional 1x1 projection of the per-point feature map to a shared
        # dimension for point-level KD (see forward_features' "point_distill"
        # output). Teacher (point_channels[-1]=384) and student (=256) have
        # different widths, so point-level MSE needs both projected into the
        # same space -- distill_dim is reused since geo/hard_geo already share
        # it. None by default so existing checkpoints/configs are unaffected.
        if point_feat_distill_channels is not None:
            self.point_feat_proj = nn.Conv1d(point_channels[-1], point_feat_distill_channels, kernel_size=1)
        else:
            self.point_feat_proj = None

    def _encode(
        self,
        points: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor | None = None,
        knn_idx: torch.Tensor | None = None,
        extra: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        coords = self.fourier_encoder(points)
        if self.extra_channels > 0:
            if extra is None:
                raise ValueError(f"model expects {self.extra_channels} extra channels but none were given")
            if extra.shape[-1] != self.extra_channels:
                raise ValueError(f"extra features must have {self.extra_channels} channels, got {extra.shape[-1]}")
            coords = torch.cat([coords, extra.transpose(1, 2)], dim=1)
        point_feat = self.point_encoder_early(coords)
        # Residual local-context injection at the first (smallest-channel)
        # layer -- see __init__'s comment on why here and not after the full
        # point_encoder. point_feat alone is purely per-point (see
        # EdgeConvBlock docstring for why that is the gap this closes).
        # Adding rather than replacing keeps the pure-PointNet path intact as
        # a fallback signal.
        if self.edge_conv is not None:
            point_feat = point_feat + self.edge_conv(points, point_feat, mask=mask, knn_idx=knn_idx)
        point_feat = self.point_encoder_late(point_feat)
        gate = torch.sigmoid(self.condition_gate(condition)).unsqueeze(-1)
        point_feat = point_feat * (1.0 + gate)
        if mask is None:
            max_pool = torch.max(point_feat, dim=2).values
            mean_pool = torch.mean(point_feat, dim=2)
            hard_pool = torch.topk(point_feat.norm(dim=1), k=max(1, int(round(point_feat.shape[2] * 0.1))), dim=1).indices
            hard_pool = torch.gather(
                point_feat,
                2,
                hard_pool.unsqueeze(1).expand(-1, point_feat.shape[1], -1),
            ).mean(dim=2)
        else:
            max_pool = _masked_max_pool(point_feat, mask)
            mean_pool = _masked_mean_pool(point_feat, mask)
            hard_pool = _masked_topk_mean_pool(point_feat, mask)
        geo = self.geometry_fuse(torch.cat([max_pool, mean_pool], dim=1))
        hard_geo = self.hard_geometry_fuse(torch.cat([max_pool, mean_pool, hard_pool], dim=1))
        cond = self.condition_encoder(condition)
        return {"coords": coords, "point_feat": point_feat, "geo": geo, "hard_geo": hard_geo, "cond": cond}

    def forward_features(
        self,
        points: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor | None = None,
        knn_idx: torch.Tensor | None = None,
        extra: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if points.ndim != 3:
            raise ValueError(f"Expected points with shape [B, N, 3], got {tuple(points.shape)}")
        if condition.ndim != 2:
            raise ValueError(f"Expected condition with shape [B, 2], got {tuple(condition.shape)}")
        if mask is not None and mask.ndim != 2:
            raise ValueError(f"Expected mask with shape [B, N], got {tuple(mask.shape)}")
        if mask is not None and mask.shape[:2] != points.shape[:2]:
            raise ValueError(
                f"Expected mask shape {tuple(points.shape[:2])}, got {tuple(mask.shape)}"
            )

        encoded = self._encode(points, condition, mask=mask, knn_idx=knn_idx, extra=extra)
        n_points = points.shape[1]
        distill = 0.5 * (encoded["geo"] + encoded["hard_geo"])
        geo_expand = distill.unsqueeze(-1).expand(-1, -1, n_points)
        cond_expand = encoded["cond"].unsqueeze(-1).expand(-1, -1, n_points)
        fused = torch.cat([encoded["coords"], encoded["point_feat"], geo_expand, cond_expand], dim=1)
        output = self.decoder(fused)
        if self.residual_decoder is not None:
            residual_fused = torch.cat([encoded["coords"], encoded["point_feat"], cond_expand], dim=1)
            output = output + self.residual_decoder(residual_fused)
        field = output.transpose(1, 2)  # [B, N, out_channels]
        cp = field[:, :, 0:1]
        # Teacher (no proj layer, point_feat_proj=None) exposes its raw
        # point_feat as the point-level KD target; student projects its own
        # (narrower) point_feat up to the teacher's width so the two live in
        # the same space. This keeps the projection a *trainable* part of
        # whichever model has it -- if teacher also needed a fresh random
        # projection post-hoc, it would inject untrained noise into a frozen,
        # eval-only model with no gradient path to fix it.
        point_distill = self.point_feat_proj(encoded["point_feat"]) if self.point_feat_proj is not None else encoded["point_feat"]
        result = {
            "field": field,
            "cp": cp,
            "point_feat": encoded["point_feat"],
            "point_distill": point_distill,
            "geo": encoded["geo"],
            "hard_geo": encoded["hard_geo"],
            "peak_geo": encoded["hard_geo"],
            "cond": encoded["cond"],
            "mask": mask,
            "distill": distill,
        }
        if self.out_channels >= 4:
            # Channels 1..3 carry the wall-velocity vector (see field_train.py).
            result["velocity"] = field[:, :, 1:4]
        return result

    def forward(
        self,
        points: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor | None = None,
        knn_idx: torch.Tensor | None = None,
        extra: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_features(points, condition, mask=mask, knn_idx=knn_idx, extra=extra)["cp"]


def build_teacher_model(
    knn_k: int = 16,
    dropout: float = 0.08,
    point_feat_distill_channels: int | None = None,
    use_edge_conv: bool = True,
) -> PressureKDRegressor:
    return PressureKDRegressor(
        point_channels=[96, 192, 384],
        condition_hidden=[64, 128],
        geometry_hidden=[384, 256],
        decoder_hidden=[384, 192, 96],
        dropout=dropout,
        fourier_bands=4,
        use_residual_decoder=True,
        residual_hidden=[128, 64],
        distill_dim=256,
        knn_k=knn_k,
        point_feat_distill_channels=point_feat_distill_channels,
        use_edge_conv=use_edge_conv,
    )


def build_student_model(
    knn_k: int = 16,
    dropout: float = 0.08,
    point_feat_distill_channels: int | None = None,
    use_edge_conv: bool = True,
) -> PressureKDRegressor:
    return PressureKDRegressor(
        point_channels=[64, 128, 256],
        condition_hidden=[32, 64],
        geometry_hidden=[256],
        decoder_hidden=[256, 128, 64],
        dropout=dropout,
        fourier_bands=4,
        use_residual_decoder=True,
        residual_hidden=[128, 64],
        distill_dim=256,
        knn_k=knn_k,
        point_feat_distill_channels=point_feat_distill_channels,
        use_edge_conv=use_edge_conv,
    )
