"""
Spatial Broadcast Decoder for reconstructing images from slots.

Each slot independently decodes to an image + alpha mask.
Final image = weighted sum of per-slot reconstructions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SpatialBroadcastDecoder(nn.Module):
    """Decodes object slots back to images.

    Each slot is broadcast to a spatial grid, concatenated with
    positional coordinates, and decoded via CNN to produce an
    RGBA image (RGB + alpha mask).
    """

    def __init__(self, slot_dim: int, resolution: int, channels: int = 64):
        super().__init__()
        self.resolution = resolution

        # Project slot to decoder input
        self.slot_proj = nn.Linear(slot_dim, slot_dim)

        # Positional grid (x, y coordinates normalized to [-1, 1])
        grid = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, resolution),
            torch.linspace(-1, 1, resolution),
            indexing="xy",
        ), dim=0)  # [2, H, W]
        self.register_buffer("grid", grid)

        # CNN decoder: (slot_dim + 2) → RGBA
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(slot_dim + 2, channels, 5, padding=2), nn.ReLU(),
            nn.ConvTranspose2d(channels, channels, 5, padding=2), nn.ReLU(),
            nn.ConvTranspose2d(channels, channels, 5, padding=2), nn.ReLU(),
            nn.ConvTranspose2d(channels, 4, 3, padding=1),  # 4 = RGB + alpha
        )

    def forward(self, slots: torch.Tensor) -> tuple:
        """
        Args:
            slots: [B, K, D]
        Returns:
            recon: [B, 3, H, W] reconstructed image
            masks: [B, K, 1, H, W] per-slot alpha masks
            slot_recons: [B, K, 3, H, W] per-slot RGB reconstructions
        """
        B, K, D = slots.shape
        H = W = self.resolution

        # Project and broadcast each slot to spatial grid
        slots = self.slot_proj(slots)
        slots = slots.reshape(B * K, D, 1, 1).expand(-1, -1, H, W)

        # Add positional encoding
        grid = self.grid.unsqueeze(0).expand(B * K, -1, -1, -1)
        decoder_input = torch.cat([slots, grid], dim=1)  # [B*K, D+2, H, W]

        # Decode
        out = self.decoder(decoder_input)  # [B*K, 4, H, W]
        out = rearrange(out, "(b k) c h w -> b k c h w", b=B, k=K)

        # Split RGB and alpha
        rgb = out[:, :, :3]    # [B, K, 3, H, W]
        alpha = out[:, :, 3:]  # [B, K, 1, H, W]

        # Softmax over slots for alpha (competition)
        masks = F.softmax(alpha, dim=1)  # [B, K, 1, H, W]

        # Weighted combination
        recon = (rgb * masks).sum(dim=1)  # [B, 3, H, W]

        return recon, masks, rgb
