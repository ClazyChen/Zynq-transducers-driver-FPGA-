"""
Proposed network: single-decoder U-Net with dual output heads (phase + amplitude).

- Main class `ProposedUNet`: corresponds to the **Proposed** method in the paper.
- I/O: forward(x, negative_mask=None) -> (phase, amplitude)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ProposedUNet(nn.Module):
    """
    Proposed U-Net (single decoder + dual output heads):
    - Shared encoder (down path)
    - Single shared decoder (up path with skip connections)
    - Two output heads: phase (tanh x pi), amplitude (sigmoid); or ``phase_only`` -> phase head + all-ones amplitude
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        use_negative_mask: bool = False,
        output_size: tuple = (8, 8),
        phase_only: bool = False,
    ):
        super().__init__()

        if use_negative_mask:
            in_channels = 2

        self.phase_only = phase_only

        # Encoder (shared)
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        # Decoder (single shared up path)
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        # Pool to array resolution (e.g., 8x8)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size)

        # Output heads (phase always; amplitude optional)
        self.phase_out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.amplitude_out_conv = None if phase_only else nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.use_negative_mask = use_negative_mask

    def forward(self, x: torch.Tensor, negative_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_negative_mask:
            if negative_mask is None:
                raise ValueError("use_negative_mask=True requires negative_mask argument")
            if negative_mask.dim() == 3:
                negative_mask = negative_mask.unsqueeze(1)
            x = torch.cat([x, negative_mask], dim=1)

        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2, stride=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2, stride=2))
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2, stride=2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2, stride=2))

        # Decoder
        dec4 = self.up4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.up3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        feats = self.adaptive_pool(dec1)

        phase = self.tanh(self.phase_out_conv(feats)) * torch.pi
        if self.phase_only:
            amplitude = torch.ones_like(phase)
            return phase, amplitude
        amplitude = self.sigmoid(self.amplitude_out_conv(feats))
        return phase, amplitude

    def get_phase_matrix(
        self, x: torch.Tensor, negative_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Convenience interface: returns (batch, H, W) phase."""
        phase, _ = self.forward(x, negative_mask)
        if phase.dim() == 4:
            phase = phase.squeeze(1)
        return phase

    def get_amplitude_matrix(
        self, x: torch.Tensor, negative_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Convenience interface: returns (batch, H, W) amplitude."""
        _, amplitude = self.forward(x, negative_mask)
        if amplitude.dim() == 4:
            amplitude = amplitude.squeeze(1)
        return amplitude


# Historical alias used by training scripts
UNetBaseline = ProposedUNet
