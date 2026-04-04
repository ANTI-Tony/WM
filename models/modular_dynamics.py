"""
Modular Causal Dynamics module.

Core innovation #2: Each interaction type has its own dynamics MLP.
Dynamics modules can be reused across novel object combinations,
enabling compositional generalization.

Architecture inspired by:
- FIOC-WM (NeurIPS 2025): f_self + f_inter separation
- Interaction Networks (Battaglia 2016): relation model + object model
- OOCDM (ICML 2024): class-level parameter sharing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InteractionModule(nn.Module):
    """Single interaction type dynamics module (e.g., collision, contact)."""

    def __init__(self, slot_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Takes [sender_slot; receiver_slot] → effect on receiver
        self.net = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )

    def forward(self, sender: torch.Tensor, receiver: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sender: [*, D] sender slot features
            receiver: [*, D] receiver slot features
        Returns:
            effect: [*, D] predicted effect on receiver
        """
        return self.net(torch.cat([sender, receiver], dim=-1))


class ModularCausalDynamics(nn.Module):
    """Predicts next-step object slots using modular causal dynamics.

    Two components:
    1. f_self: Independent self-evolution per object (gravity, inertia)
    2. f_inter[τ]: Type-specific interaction effects (one MLP per type τ)

    Update rule:
        s_{t+1}^i = s_t^i + f_self(s_t^i) + Σ_j e_ij * Σ_τ w_τ^ij * f_inter[τ](s_t^j, s_t^i)
    """

    def __init__(self, slot_dim: int, num_interaction_types: int = 4,
                 hidden_dim: int = 128, num_message_passing: int = 2,
                 residual: bool = True):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_types = num_interaction_types
        self.num_mp = num_message_passing
        self.residual = residual

        # Self-dynamics (shared across all objects)
        self.f_self = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )

        # Type-specific interaction modules
        self.f_inter = nn.ModuleList([
            InteractionModule(slot_dim, hidden_dim)
            for _ in range(num_interaction_types)
        ])

        # Post-aggregation update (applied after message passing)
        self.update_mlp = nn.Sequential(
            nn.LayerNorm(slot_dim * 2),  # [slot; aggregated_effects]
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )

        # For interventional verification (predict without graph)
        self.f_no_graph = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )

    def forward(self, slots: torch.Tensor, edge_probs: torch.Tensor,
                edge_types: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slots: [B, K, D] current object slots
            edge_probs: [B, K, K] edge existence probabilities
            edge_types: [B, K, K, M] type-weighted edge assignments
        Returns:
            next_slots: [B, K, D] predicted next-step slots
        """
        B, K, D = slots.shape
        h = slots

        for _ in range(self.num_mp):
            # Self-dynamics
            delta_self = self.f_self(h)  # [B, K, D]

            # Interaction dynamics (vectorized over types)
            # For each type τ, compute effects from all senders to all receivers
            delta_inter = torch.zeros_like(h)  # [B, K, D]

            # Expand for pairwise computation
            h_sender = h.unsqueeze(2).expand(B, K, K, D)    # [B, K_s, K_r, D]
            h_receiver = h.unsqueeze(1).expand(B, K, K, D)  # [B, K_s, K_r, D]

            for tau in range(self.num_types):
                # Compute effect of each sender on each receiver for type τ
                effect_tau = self.f_inter[tau](
                    h_sender.reshape(-1, D),
                    h_receiver.reshape(-1, D),
                ).reshape(B, K, K, D)  # [B, K_sender, K_receiver, D]

                # Weight by edge probability AND type weight
                weight = edge_probs.unsqueeze(-1) * \
                         edge_types[..., tau:tau+1]  # [B, K, K, 1]

                # Aggregate: sum over senders for each receiver
                # effect_tau[b, j, i, d] = effect of j on i
                # sum over j (dim=1) to get total effect on i
                delta_inter += (effect_tau * weight).sum(dim=1)  # [B, K, D]

            # Update slots
            combined = torch.cat([h + delta_self, delta_inter], dim=-1)
            update = self.update_mlp(combined)

            if self.residual:
                h = h + update
            else:
                h = update

        return h

    def predict_no_graph(self, slots: torch.Tensor) -> torch.Tensor:
        """Predict next slots without graph (for interventional verification).

        This is a simple independent predictor used only to measure
        causal effects during interventional masking.
        """
        return slots + self.f_no_graph(slots)

    def rollout(self, slots: torch.Tensor, edge_probs: torch.Tensor,
                edge_types: torch.Tensor, steps: int = 5):
        """Multi-step rollout for evaluation.

        Args:
            slots: [B, K, D] initial slots
            edge_probs: [B, K, K] (assumed static for now)
            edge_types: [B, K, K, M] (assumed static for now)
            steps: number of future steps
        Returns:
            trajectory: [B, T, K, D] predicted slot trajectory
        """
        trajectory = [slots]
        current = slots
        for _ in range(steps):
            current = self.forward(current, edge_probs, edge_types)
            trajectory.append(current)
        return torch.stack(trajectory, dim=1)
