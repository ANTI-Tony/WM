"""
Slot Attention module for object discovery.

Based on Locatello et al. (2020) "Object-Centric Learning with Slot Attention"
with minor improvements from recent literature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SlotAttention(nn.Module):
    """Iterative slot attention mechanism.

    Slots compete via softmax-over-slots to bind to input features.
    Each iteration refines the slot representations.
    """

    def __init__(self, num_slots: int, slot_dim: int, input_dim: int,
                 num_iterations: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations

        # Learnable slot initialization (Gaussian params)
        self.slot_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

        # Attention projections
        self.to_q = nn.Linear(slot_dim, slot_dim)
        self.to_k = nn.Linear(input_dim, slot_dim)
        self.to_v = nn.Linear(input_dim, slot_dim)

        # Slot update via GRU
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # Layer norms
        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_pre_ff = nn.LayerNorm(slot_dim)

        # Residual MLP
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )

        self.scale = slot_dim ** -0.5

    def forward(self, inputs: torch.Tensor, num_slots: int = None) -> torch.Tensor:
        """
        Args:
            inputs: [B, N, D_in] spatial features from encoder
            num_slots: override number of slots (optional)
        Returns:
            slots: [B, K, D_slot]
            attn_weights: [B, K, N] attention maps (for visualization)
        """
        B, N, _ = inputs.shape
        K = num_slots or self.num_slots

        # Initialize slots from learned Gaussian
        mu = self.slot_mu.expand(B, K, -1)
        sigma = self.slot_log_sigma.exp().expand(B, K, -1)
        slots = mu + sigma * torch.randn_like(mu)

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)  # [B, N, D]
        v = self.to_v(inputs)  # [B, N, D]

        attn_weights = None
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)  # [B, K, D]

            # Attention: dots[b, k, n] = q[b,k] . k[b,n]
            dots = torch.einsum("bkd,bnd->bkn", q, k) * self.scale

            # Softmax over SLOTS (competition)
            attn = F.softmax(dots, dim=1)          # [B, K, N]
            attn_weights = attn

            # Weighted mean (normalize per slot)
            attn_norm = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            updates = torch.einsum("bkn,bnd->bkd", attn_norm, v)

            # GRU update + residual MLP
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            ).reshape(B, K, self.slot_dim)

            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots, attn_weights


class SlotEncoder(nn.Module):
    """CNN encoder that maps video frames to spatial feature maps,
    then applies Slot Attention to extract object slots.
    """

    def __init__(self, resolution: int, num_slots: int, slot_dim: int,
                 num_iterations: int = 3, encoder_channels: int = 64):
        super().__init__()
        self.resolution = resolution

        # Simple CNN encoder (4 conv layers, stride 1, same padding)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, encoder_channels, 5, padding=2), nn.ReLU(),
            nn.Conv2d(encoder_channels, encoder_channels, 5, padding=2), nn.ReLU(),
            nn.Conv2d(encoder_channels, encoder_channels, 5, padding=2), nn.ReLU(),
            nn.Conv2d(encoder_channels, encoder_channels, 5, padding=2), nn.ReLU(),
        )

        # Positional embedding for spatial features
        self.pos_embed = nn.Parameter(
            torch.randn(1, encoder_channels, resolution, resolution) * 0.02
        )

        # Project to slot attention input dim
        self.layer_norm = nn.LayerNorm(encoder_channels)

        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            input_dim=encoder_channels,
            num_iterations=num_iterations,
        )

    def forward(self, images: torch.Tensor):
        """
        Args:
            images: [B, 3, H, W] single frame or [B, T, 3, H, W] video
        Returns:
            slots: [B, K, D] or [B, T, K, D]
            attn: [B, K, N] or [B, T, K, N]
        """
        if images.dim() == 5:
            # Video: process each frame
            B, T, C, H, W = images.shape
            images_flat = rearrange(images, "b t c h w -> (b t) c h w")
            slots_flat, attn_flat = self._encode_single(images_flat)
            K, D = slots_flat.shape[1], slots_flat.shape[2]
            N = attn_flat.shape[2]
            slots = rearrange(slots_flat, "(b t) k d -> b t k d", b=B, t=T)
            attn = rearrange(attn_flat, "(b t) k n -> b t k n", b=B, t=T)
            return slots, attn
        else:
            return self._encode_single(images)

    def _encode_single(self, images: torch.Tensor):
        """Encode single batch of images."""
        feat = self.encoder(images)  # [B, C, H, W]
        feat = feat + self.pos_embed[:, :, :feat.shape[2], :feat.shape[3]]
        feat = rearrange(feat, "b c h w -> b (h w) c")  # [B, N, C]
        feat = self.layer_norm(feat)
        return self.slot_attention(feat)
