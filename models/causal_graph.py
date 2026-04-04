"""
Causal Interaction Graph Discovery module.

Core innovation #1: Discovers which objects causally influence each other,
AND classifies each interaction into a discrete type (collision, contact, etc.).
Uses both statistical signals and interventional verification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gumbel_softmax(logits: torch.Tensor, temperature: float = 0.5,
                   hard: bool = False) -> torch.Tensor:
    """Gumbel-Softmax with optional straight-through estimator."""
    return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)


class CausalGraphDiscovery(nn.Module):
    """Discovers causal interaction graph between object slots.

    For each pair (i, j):
    1. Predicts edge existence probability: e_ij ∈ [0, 1]
    2. Classifies interaction type: τ_ij ∈ {0, ..., M-1}
    3. (During training) Verifies via interventional masking
    """

    def __init__(self, slot_dim: int, num_interaction_types: int = 4,
                 hidden_dim: int = 128, gumbel_temperature: float = 0.5):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_types = num_interaction_types
        self.temperature = gumbel_temperature

        # Edge existence predictor: takes [s_i; s_j; |s_i - s_j|]
        self.edge_mlp = nn.Sequential(
            nn.Linear(slot_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # logit for edge existence
        )

        # Interaction type classifier: takes [s_i; s_j; s_i - s_j; s_i * s_j]
        self.type_mlp = nn.Sequential(
            nn.Linear(slot_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_interaction_types),
        )

        # Learnable null token for interventional masking
        self.null_token = nn.Parameter(torch.randn(slot_dim) * 0.02)

    def forward(self, slots: torch.Tensor, hard: bool = False):
        """
        Args:
            slots: [B, K, D] object slot representations
            hard: if True, use hard Gumbel-Softmax (for inference)
        Returns:
            edge_probs: [B, K, K] edge existence probabilities (0 on diagonal)
            edge_types: [B, K, K, M] type assignment (soft or hard)
            graph_info: dict with auxiliary info for loss computation
        """
        B, K, D = slots.shape

        # Compute all pairwise features
        # slots_i: [B, K, 1, D], slots_j: [B, 1, K, D]
        si = slots.unsqueeze(2).expand(B, K, K, D)  # sender
        sj = slots.unsqueeze(1).expand(B, K, K, D)  # receiver

        # --- Edge existence ---
        edge_input = torch.cat([si, sj, (si - sj).abs()], dim=-1)  # [B,K,K,3D]
        edge_logits = self.edge_mlp(edge_input).squeeze(-1)         # [B,K,K]
        edge_probs = torch.sigmoid(edge_logits)

        # Zero out diagonal (no self-loops)
        mask = ~torch.eye(K, dtype=torch.bool, device=slots.device)
        edge_probs = edge_probs * mask.unsqueeze(0)

        # --- Interaction type classification ---
        type_input = torch.cat([si, sj, si - sj, si * sj], dim=-1)  # [B,K,K,4D]
        type_logits = self.type_mlp(type_input)                       # [B,K,K,M]

        # Gumbel-Softmax for differentiable discrete type selection
        if self.training:
            edge_types = gumbel_softmax(type_logits, self.temperature, hard=hard)
        else:
            edge_types = F.one_hot(
                type_logits.argmax(dim=-1), self.num_types
            ).float()

        # Weight types by edge probability (no edge → no type matters)
        edge_types = edge_types * edge_probs.unsqueeze(-1)

        graph_info = {
            "edge_logits": edge_logits,
            "edge_probs": edge_probs,
            "type_logits": type_logits,
        }

        return edge_probs, edge_types, graph_info

    def interventional_verify(self, slots: torch.Tensor, predictor: nn.Module,
                              edge_probs: torch.Tensor,
                              threshold: float = 0.1) -> torch.Tensor:
        """Verify edges via interventional masking (used during training).

        For each candidate edge (j→i), mask slot j and check if
        the prediction of slot i changes.

        Args:
            slots: [B, K, D]
            predictor: dynamics model that predicts next slots
            edge_probs: [B, K, K] current edge probabilities
            threshold: minimum causal effect to confirm edge
        Returns:
            causal_mask: [B, K, K] binary mask of verified edges
            causal_effects: [B, K, K] magnitude of causal effects
        """
        B, K, D = slots.shape

        # Full prediction (no intervention)
        with torch.no_grad():
            pred_full = predictor.predict_no_graph(slots)  # [B, K, D]

        causal_effects = torch.zeros(B, K, K, device=slots.device)

        # For each potential cause j
        for j in range(K):
            # Intervene: replace slot j with null token
            slots_masked = slots.clone()
            slots_masked[:, j] = self.null_token

            with torch.no_grad():
                pred_masked = predictor.predict_no_graph(slots_masked)

            # Causal effect on each target i
            for i in range(K):
                if i == j:
                    continue
                effect = (pred_full[:, i] - pred_masked[:, i]).pow(2).sum(-1).sqrt()
                causal_effects[:, j, i] = effect

        causal_mask = (causal_effects > threshold).float()
        return causal_mask, causal_effects

    def compute_loss(self, graph_info: dict, causal_mask: torch.Tensor = None):
        """Compute graph discovery losses.

        Args:
            graph_info: dict from forward()
            causal_mask: [B, K, K] from interventional_verify (optional)
        Returns:
            loss_dict: dict of individual loss terms
        """
        edge_probs = graph_info["edge_probs"]
        type_logits = graph_info["type_logits"]

        losses = {}

        # Sparsity loss: encourage fewer edges
        losses["sparsity"] = edge_probs.sum(dim=(-1, -2)).mean()

        # Type entropy loss: encourage crisp type assignment
        type_dist = F.softmax(type_logits, dim=-1)
        entropy = -(type_dist * (type_dist + 1e-8).log()).sum(dim=-1)
        # Only count entropy for active edges
        losses["type_entropy"] = (entropy * edge_probs).sum(dim=(-1, -2)).mean()

        # Interventional consistency loss (if causal_mask available)
        if causal_mask is not None:
            # Penalize: high causal effect but low edge probability
            consistency = F.binary_cross_entropy(
                edge_probs, causal_mask, reduction="mean"
            )
            losses["causal_consistency"] = consistency

        return losses
