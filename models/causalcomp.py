"""
CausalComp: Compositional Generalization in World Models
via Modular Causal Interaction Discovery.

Main model that integrates all components:
1. SlotEncoder: video frames → object slots
2. CausalGraphDiscovery: slots → causal interaction graph
3. ModularCausalDynamics: slots + graph → predicted next slots
4. SpatialBroadcastDecoder: slots → reconstructed frames
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .slot_attention import SlotEncoder
from .causal_graph import CausalGraphDiscovery
from .modular_dynamics import ModularCausalDynamics
from .decoder import SpatialBroadcastDecoder


class CausalComp(nn.Module):
    """Full CausalComp model."""

    def __init__(self, resolution: int = 128, num_slots: int = 8,
                 slot_dim: int = 128, num_interaction_types: int = 4,
                 encoder_channels: int = 64, dynamics_hidden: int = 128,
                 num_message_passing: int = 2, gumbel_temperature: float = 0.5):
        super().__init__()

        self.resolution = resolution
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # Module 1: Object discovery
        self.encoder = SlotEncoder(
            resolution=resolution,
            num_slots=num_slots,
            slot_dim=slot_dim,
            encoder_channels=encoder_channels,
        )

        # Module 2: Causal graph discovery
        self.graph_discovery = CausalGraphDiscovery(
            slot_dim=slot_dim,
            num_interaction_types=num_interaction_types,
            hidden_dim=dynamics_hidden,
            gumbel_temperature=gumbel_temperature,
        )

        # Module 3: Modular causal dynamics
        self.dynamics = ModularCausalDynamics(
            slot_dim=slot_dim,
            num_interaction_types=num_interaction_types,
            hidden_dim=dynamics_hidden,
            num_message_passing=num_message_passing,
        )

        # Module 4: Decoder (for reconstruction loss)
        self.decoder = SpatialBroadcastDecoder(
            slot_dim=slot_dim,
            resolution=resolution,
        )

    def forward(self, video: torch.Tensor, rollout_steps: int = 1):
        """
        Args:
            video: [B, T, 3, H, W] video clip
            rollout_steps: how many future steps to predict
        Returns:
            output: dict with all predictions and losses
        """
        B, T, C, H, W = video.shape

        # --- Encode all frames to slots ---
        all_slots, all_attn = self.encoder(video)  # [B, T, K, D], [B, T, K, N]

        # --- For each timestep, discover graph and predict next ---
        pred_slots_list = []
        graph_info_list = []
        recon_list = []

        # Reconstruct first frame (sanity check for slot quality)
        recon_t0, masks_t0, _ = self.decoder(all_slots[:, 0])
        recon_list.append(recon_t0)

        for t in range(min(T - 1, rollout_steps)):
            slots_t = all_slots[:, t]  # [B, K, D]

            # Discover causal graph at time t
            edge_probs, edge_types, graph_info = self.graph_discovery(slots_t)
            graph_info_list.append(graph_info)

            # Predict next slots
            pred_slots_next = self.dynamics(slots_t, edge_probs, edge_types)
            pred_slots_list.append(pred_slots_next)

            # Decode predicted slots
            recon_next, _, _ = self.decoder(pred_slots_next)
            recon_list.append(recon_next)

        # Stack predictions
        pred_slots = torch.stack(pred_slots_list, dim=1)  # [B, T', K, D]
        recon_frames = torch.stack(recon_list, dim=1)      # [B, T'+1, 3, H, W]

        return {
            "pred_slots": pred_slots,         # [B, T', K, D]
            "target_slots": all_slots[:, 1:rollout_steps+1],  # [B, T', K, D]
            "recon_frames": recon_frames,      # [B, T'+1, 3, H, W]
            "target_frames": video[:, :rollout_steps+1],      # [B, T'+1, 3, H, W]
            "all_slots": all_slots,            # [B, T, K, D]
            "attn_maps": all_attn,             # [B, T, K, N]
            "graph_infos": graph_info_list,    # list of dicts
            "masks_t0": masks_t0,              # [B, K, 1, H, W]
        }

    def compute_loss(self, output: dict, config) -> dict:
        """Compute all loss terms.

        Args:
            output: dict from forward()
            config: TrainConfig with loss weights
        Returns:
            losses: dict of named loss terms + total loss
        """
        losses = {}

        # L_recon: reconstruction quality
        recon = output["recon_frames"]
        target = output["target_frames"]
        T_recon = min(recon.shape[1], target.shape[1])
        losses["recon"] = F.mse_loss(recon[:, :T_recon], target[:, :T_recon])

        # L_dynamics: slot prediction accuracy
        pred = output["pred_slots"]
        target_slots = output["target_slots"]
        T_pred = min(pred.shape[1], target_slots.shape[1])
        if T_pred > 0:
            losses["dynamics"] = F.mse_loss(pred[:, :T_pred], target_slots[:, :T_pred])
        else:
            losses["dynamics"] = torch.tensor(0.0, device=pred.device)

        # Graph losses (aggregate over timesteps)
        sparsity_losses = []
        entropy_losses = []
        min_connect_losses = []
        for graph_info in output["graph_infos"]:
            gl = self.graph_discovery.compute_loss(graph_info)
            sparsity_losses.append(gl["sparsity"])
            entropy_losses.append(gl["type_entropy"])
            min_connect_losses.append(gl["min_connect"])

        if sparsity_losses:
            losses["sparsity"] = torch.stack(sparsity_losses).mean()
            losses["type_entropy"] = torch.stack(entropy_losses).mean()
            losses["min_connect"] = torch.stack(min_connect_losses).mean()
        else:
            losses["sparsity"] = torch.tensor(0.0, device=pred.device)
            losses["type_entropy"] = torch.tensor(0.0, device=pred.device)
            losses["min_connect"] = torch.tensor(0.0, device=pred.device)

        # Total weighted loss
        # Note: sparsity weight is kept low (0.001) to prevent graph collapse.
        # min_connect weight is high (1.0) to enforce minimum connectivity.
        losses["total"] = (
            config.recon_weight * losses["recon"] +
            config.dynamics_weight * losses["dynamics"] +
            config.sparsity_weight * 0.1 * losses["sparsity"] +  # reduce sparsity push
            config.entropy_weight * losses["type_entropy"] +
            1.0 * losses["min_connect"]  # prevent graph collapse
        )

        return losses

    @torch.no_grad()
    def predict_trajectory(self, video_context: torch.Tensor, future_steps: int = 10):
        """Predict future trajectory given context frames (for evaluation).

        Args:
            video_context: [B, T_ctx, 3, H, W] observed frames
            future_steps: number of future steps to predict
        Returns:
            trajectory: [B, future_steps, K, D] predicted future slots
            recon_trajectory: [B, future_steps, 3, H, W] decoded frames
        """
        self.eval()

        # Encode context
        all_slots, _ = self.encoder(video_context)  # [B, T_ctx, K, D]
        current_slots = all_slots[:, -1]             # [B, K, D] last frame

        # Discover graph from last frame
        edge_probs, edge_types, _ = self.graph_discovery(current_slots, hard=True)

        # Auto-regressive rollout
        pred_slots = []
        pred_frames = []
        for _ in range(future_steps):
            current_slots = self.dynamics(current_slots, edge_probs, edge_types)
            pred_slots.append(current_slots)
            frame, _, _ = self.decoder(current_slots)
            pred_frames.append(frame)

            # Re-discover graph (dynamics may change interactions)
            edge_probs, edge_types, _ = self.graph_discovery(current_slots, hard=True)

        return torch.stack(pred_slots, dim=1), torch.stack(pred_frames, dim=1)
