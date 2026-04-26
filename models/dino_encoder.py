"""
DINOv2-based encoder for stronger object-centric slot representations.

Uses frozen DINOv2 ViT-S/14 patch features as input to Slot Attention,
replacing the simple 4-layer CNN encoder. DINOv2 features are much more
semantically meaningful, leading to better object decomposition.

Reference: Causal-JEPA (2026) uses a similar DINOv2+SlotAttention pipeline.
"""

import torch
import torch.nn as nn
from einops import rearrange

from .slot_attention import SlotAttention


class DINOSlotEncoder(nn.Module):
    """DINOv2 frozen backbone + Slot Attention.

    DINOv2 ViT-S/14 outputs 384-dim patch tokens at stride 14.
    For a 224x224 image: 16x16 = 256 patch tokens.
    For a 64x64 image (after resize to 224): still 16x16 = 256 tokens.
    """

    def __init__(self, num_slots: int = 8, slot_dim: int = 128,
                 num_iterations: int = 3, dino_model: str = "dinov2_vits14"):
        super().__init__()
        self.num_slots = num_slots

        # Load frozen DINOv2
        self.dino = torch.hub.load("facebookresearch/dinov2", dino_model)
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad = False

        dino_dim = self.dino.embed_dim  # 384 for ViT-S

        # Project DINOv2 features to slot attention input
        self.proj = nn.Sequential(
            nn.LayerNorm(dino_dim),
            nn.Linear(dino_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
        )

        # Slot Attention
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            input_dim=slot_dim,
            num_iterations=num_iterations,
        )

        # DINOv2 expects 224x224, we'll resize
        self.resize = nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False)

        # Normalization for DINOv2 (ImageNet stats)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def _extract_dino_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract patch features from frozen DINOv2.

        Args:
            images: [B, 3, H, W] in [0, 1]
        Returns:
            features: [B, N_patches, dino_dim]
        """
        # Resize and normalize
        x = self.resize(images)
        x = (x - self.mean) / self.std

        # Extract patch tokens (exclude CLS token)
        features = self.dino.forward_features(x)
        patch_tokens = features["x_norm_patchtokens"]  # [B, N, D]
        return patch_tokens

    def forward(self, images: torch.Tensor):
        """
        Args:
            images: [B, 3, H, W] or [B, T, 3, H, W]
        Returns:
            slots: [B, K, D] or [B, T, K, D]
            attn: [B, K, N] or [B, T, K, N]
        """
        if images.dim() == 5:
            B, T, C, H, W = images.shape
            images_flat = rearrange(images, "b t c h w -> (b t) c h w")
            slots_flat, attn_flat = self._encode_single(images_flat)
            K, D = slots_flat.shape[1], slots_flat.shape[2]
            slots = rearrange(slots_flat, "(b t) k d -> b t k d", b=B, t=T)
            attn = rearrange(attn_flat, "(b t) k n -> b t k n", b=B, t=T)
            return slots, attn
        else:
            return self._encode_single(images)

    def _encode_single(self, images: torch.Tensor):
        """Encode batch of images."""
        # DINOv2 features
        dino_feat = self._extract_dino_features(images)  # [B, N, dino_dim]

        # Project
        feat = self.proj(dino_feat)  # [B, N, slot_dim]

        # Slot Attention
        return self.slot_attention(feat)
