from __future__ import annotations

import torch
from torch import nn

from .model import MLP, SharedPointMLP, _group_norm_1d


class PointwiseDecoder(nn.Module):
    def __init__(self, in_channels: int, hidden: list[int], dropout: float = 0.1, out_channels: int = 1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_channels
        for size in hidden:
            layers.extend(
                [
                    nn.Conv1d(current, size, kernel_size=1, bias=False),
                    _group_norm_1d(size),
                    nn.ReLU(inplace=True),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current = size
        layers.append(nn.Conv1d(current, int(out_channels), kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourierFeatureEncoder(nn.Module):
    def __init__(self, num_bands: int = 0) -> None:
        super().__init__()
        self.num_bands = int(num_bands)
        if self.num_bands > 0:
            bands = torch.pow(2.0, torch.arange(self.num_bands, dtype=torch.float32))
        else:
            bands = torch.empty(0, dtype=torch.float32)
        self.register_buffer("bands", bands)

    @property
    def out_channels(self) -> int:
        return 3 + 3 * 2 * self.num_bands

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        coords = points.transpose(1, 2)
        if self.num_bands <= 0:
            return coords
        encoded = [coords]
        scaled = coords.unsqueeze(2) * self.bands.view(1, 1, -1, 1) * torch.pi
        encoded.append(torch.sin(scaled).flatten(1, 2))
        encoded.append(torch.cos(scaled).flatten(1, 2))
        return torch.cat(encoded, dim=1)


class Stage1PressureRegressor(nn.Module):
    def __init__(
        self,
        point_in_channels: int = 3,
        point_channels: list[int] | None = None,
        condition_hidden: list[int] | None = None,
        geometry_hidden: list[int] | None = None,
        decoder_hidden: list[int] | None = None,
        dropout: float = 0.1,
        fourier_bands: int = 0,
        use_residual_decoder: bool = False,
        residual_hidden: list[int] | None = None,
    ) -> None:
        super().__init__()
        point_channels = point_channels or [64, 128, 256]
        condition_hidden = condition_hidden or [32, 64]
        geometry_hidden = geometry_hidden or [256]
        decoder_hidden = decoder_hidden or [256, 128, 64]
        residual_hidden = residual_hidden or [128, 64]

        self.fourier_encoder = FourierFeatureEncoder(fourier_bands)
        encoded_point_channels = self.fourier_encoder.out_channels
        self.use_residual_decoder = bool(use_residual_decoder)
        self.point_encoder = SharedPointMLP(encoded_point_channels, point_channels)
        self.geometry_fuse = MLP(point_channels[-1] * 2, geometry_hidden, point_channels[-1], dropout=dropout)
        self.condition_encoder = MLP(2, condition_hidden, condition_hidden[-1], dropout=dropout)
        decoder_in = point_channels[-1] + point_channels[-1] + condition_hidden[-1] + encoded_point_channels
        self.decoder = PointwiseDecoder(decoder_in, decoder_hidden, dropout=dropout)
        if self.use_residual_decoder:
            residual_in = encoded_point_channels + point_channels[-1] + condition_hidden[-1]
            self.residual_decoder = PointwiseDecoder(residual_in, residual_hidden, dropout=dropout)
        else:
            self.residual_decoder = None

    def forward(self, points: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3:
            raise ValueError(f"Expected points with shape [B, N, 3], got {tuple(points.shape)}")
        if condition.ndim != 2:
            raise ValueError(f"Expected condition with shape [B, 2], got {tuple(condition.shape)}")

        coords = self.fourier_encoder(points)
        point_feat = self.point_encoder(coords)
        max_pool = torch.max(point_feat, dim=2).values
        mean_pool = torch.mean(point_feat, dim=2)
        geo = self.geometry_fuse(torch.cat([max_pool, mean_pool], dim=1))
        cond = self.condition_encoder(condition)

        n_points = points.shape[1]
        geo_expand = geo.unsqueeze(-1).expand(-1, -1, n_points)
        cond_expand = cond.unsqueeze(-1).expand(-1, -1, n_points)
        fused = torch.cat([coords, point_feat, geo_expand, cond_expand], dim=1)
        output = self.decoder(fused)
        if self.residual_decoder is not None:
            residual_fused = torch.cat([coords, point_feat, cond_expand], dim=1)
            output = output + self.residual_decoder(residual_fused)
        return output.transpose(1, 2)
