from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _group_norm_1d(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    # KD training runs at batch_size=1 (whole point clouds, variable N per
    # sample), which makes BatchNorm1d's running stats chase a single sample's
    # statistics each step and disagree badly with the momentum-averaged stats
    # used at eval time -- verified empirically: switching the same checkpoint
    # from train() to eval() on an unchanged input shifted predicted Cp by
    # ~0.02, comparable to the model's actual error. GroupNorm normalises
    # within each sample (over points, per channel group) so it needs no
    # cross-batch running statistics and is train/eval-consistent by
    # construction. Pick the largest group count in {1,2,4,8} that divides
    # num_channels evenly.
    for groups in (max_groups, 4, 2, 1):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class SharedPointMLP(nn.Module):
    def __init__(self, in_channels: int, channels: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_channels
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv1d(current, out_channels, kernel_size=1, bias=False),
                    _group_norm_1d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            current = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: list[int], out_features: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_features
        for size in hidden:
            layers.extend(
                [
                    nn.Linear(current, size),
                    nn.LayerNorm(size),
                    nn.ReLU(inplace=True),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current = size
        layers.append(nn.Linear(current, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Stage1PointRegressor(nn.Module):
    def __init__(
        self,
        point_in_channels: int = 3,
        geo_channels: list[int] | None = None,
        condition_hidden: list[int] | None = None,
        fusion_hidden: list[int] | None = None,
        output_dim: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        geo_channels = geo_channels or [64, 128, 256]
        condition_hidden = condition_hidden or [32, 64]
        fusion_hidden = fusion_hidden or [256, 128]

        self.point_encoder = SharedPointMLP(point_in_channels, geo_channels)
        geo_out = geo_channels[-1]
        self.geo_fuse = MLP(geo_out * 2, [geo_out], geo_out, dropout=dropout)
        self.condition_encoder = MLP(2, condition_hidden, condition_hidden[-1], dropout=dropout)
        fusion_in = geo_out + condition_hidden[-1]
        self.head = MLP(fusion_in, fusion_hidden, output_dim, dropout=dropout)

    def forward(self, points: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3:
            raise ValueError(f"Expected points with shape [B, N, 3], got {tuple(points.shape)}")
        if condition.ndim != 2:
            raise ValueError(f"Expected condition with shape [B, 2], got {tuple(condition.shape)}")
        x = points.transpose(1, 2)
        features = self.point_encoder(x)
        max_pool = torch.max(features, dim=2).values
        mean_pool = torch.mean(features, dim=2)
        geo = self.geo_fuse(torch.cat([max_pool, mean_pool], dim=1))
        cond = self.condition_encoder(condition)
        fused = torch.cat([geo, cond], dim=1)
        return self.head(fused)

