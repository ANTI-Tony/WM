"""
Mixture-of-Experts (MoE) baseline for dynamics prediction.

Key difference from CausalComp:
- CausalComp: Gumbel-Softmax routing based on PAIRWISE slot features → typed modules
- MoE: Soft router based on SINGLE object features → expert modules

MoE has the same modular structure (M experts) but without explicit
causal graph discovery. This tests whether typed modules alone suffice,
or whether the causal graph routing is necessary.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEDynamics(nn.Module):
    """MoE dynamics: M expert MLPs with learned soft routing.

    Each object gets a soft assignment over M experts based on its own features.
    No pairwise edge prediction — routing is per-object, not per-interaction.
    """

    def __init__(self, state_dim=8, slot_dim=128, num_experts=8):
        super().__init__()
        self.num_experts = num_experts

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, slot_dim), nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
        )

        # Router: per-object soft assignment to experts
        self.router = nn.Sequential(
            nn.Linear(slot_dim, slot_dim), nn.ReLU(),
            nn.Linear(slot_dim, num_experts),
        )

        # Self-dynamics
        self.f_self = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim), nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
        )

        # Expert interaction modules (same architecture as CausalComp's typed modules)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(slot_dim * 2, slot_dim), nn.ReLU(),
                nn.Linear(slot_dim, slot_dim),
            )
            for _ in range(num_experts)
        ])

        # Update MLP
        self.update = nn.Sequential(
            nn.LayerNorm(slot_dim * 2),
            nn.Linear(slot_dim * 2, slot_dim), nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
        )

        self.state_decoder = nn.Sequential(
            nn.Linear(slot_dim, slot_dim), nn.ReLU(),
            nn.Linear(slot_dim, state_dim),
        )

    def forward(self, gt_states, rollout_steps=8):
        B, T, K, D = gt_states.shape
        h = self.state_encoder(gt_states[:, 0])  # [B, K, slot_dim]

        preds = []
        for _ in range(min(T - 1, rollout_steps)):
            # Router: per-object expert weights
            weights = F.softmax(self.router(h), dim=-1)  # [B, K, M]

            # Self-dynamics
            delta_self = self.f_self(h)

            # Expert interactions (all-to-all, weighted by router)
            si = h.unsqueeze(2).expand(B, K, K, -1)
            sj = h.unsqueeze(1).expand(B, K, K, -1)

            delta_inter = torch.zeros_like(h)
            for m in range(self.num_experts):
                effect = self.experts[m](torch.cat([si, sj], dim=-1))  # [B,K,K,D]
                # Weight by receiver's expert assignment
                w = weights[:, :, m].unsqueeze(1).unsqueeze(-1)  # [B, 1, K, 1]
                delta_inter += (effect * w).mean(dim=1)  # aggregate over senders

            h = self.update(torch.cat([delta_self, delta_inter], dim=-1))
            preds.append(self.state_decoder(h))
            h = self.state_encoder(preds[-1])

        return {
            "pred_states": torch.stack(preds, 1),
            "target_states": gt_states[:, 1:rollout_steps + 1],
            "graph_infos": [],
        }


class MoEPairwise(nn.Module):
    """MoE with PAIRWISE routing — most direct comparison to CausalComp.

    Same as CausalComp but without explicit edge existence prediction.
    All pairs get routed to experts; no sparsity, no causal graph.
    Tests whether the GRAPH STRUCTURE matters, or just having typed modules.
    """

    def __init__(self, state_dim=8, slot_dim=128, num_experts=8):
        super().__init__()
        self.num_experts = num_experts

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, slot_dim), nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
        )

        # Pairwise router (like CausalComp's type classifier but no edge prediction)
        self.router = nn.Sequential(
            nn.Linear(slot_dim * 4, slot_dim), nn.ReLU(),
            nn.Linear(slot_dim, num_experts),
        )

        self.f_self = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim), nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
        )

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(slot_dim * 2, slot_dim), nn.ReLU(),
                nn.Linear(slot_dim, slot_dim),
            )
            for _ in range(num_experts)
        ])

        self.update = nn.Sequential(
            nn.LayerNorm(slot_dim * 2),
            nn.Linear(slot_dim * 2, slot_dim), nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
        )

        self.state_decoder = nn.Sequential(
            nn.Linear(slot_dim, slot_dim), nn.ReLU(),
            nn.Linear(slot_dim, state_dim),
        )

    def forward(self, gt_states, rollout_steps=8):
        B, T, K, D = gt_states.shape
        h = self.state_encoder(gt_states[:, 0])

        preds = []
        for _ in range(min(T - 1, rollout_steps)):
            delta_self = self.f_self(h)

            si = h.unsqueeze(2).expand(B, K, K, -1)
            sj = h.unsqueeze(1).expand(B, K, K, -1)

            # Pairwise routing
            pair_feat = torch.cat([si, sj, si - sj, si * sj], dim=-1)
            weights = F.softmax(self.router(pair_feat), dim=-1)  # [B,K,K,M]

            delta_inter = torch.zeros_like(h)
            for m in range(self.num_experts):
                effect = self.experts[m](torch.cat([si, sj], dim=-1))
                w = weights[..., m:m+1]  # [B,K,K,1]
                delta_inter += (effect * w).sum(dim=1)  # sum over senders

            h = self.update(torch.cat([delta_self, delta_inter], dim=-1))
            preds.append(self.state_decoder(h))
            h = self.state_encoder(preds[-1])

        return {
            "pred_states": torch.stack(preds, 1),
            "target_states": gt_states[:, 1:rollout_steps + 1],
            "graph_infos": [],
        }
