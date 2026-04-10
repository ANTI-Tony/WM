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

        # --- Autoregressive rollout ---
        # Step 1: use GT slots[0] as starting point
        # Step 2+: use PREDICTED slots as input (errors compound → forces good dynamics)
        pred_slots_list = []
        pred_frames_list = []
        graph_info_list = []

        # Reconstruct first frame (sanity check for slot quality)
        recon_t0, masks_t0, _ = self.decoder(all_slots[:, 0])

        # Start from GT-encoded slots of first frame
        current_slots = all_slots[:, 0]  # [B, K, D]

        for t in range(min(T - 1, rollout_steps)):
            # Discover causal graph from current slots
            edge_probs, edge_types, graph_info = self.graph_discovery(current_slots)
            graph_info_list.append(graph_info)

            # Predict next slots
            pred_slots_next = self.dynamics(current_slots, edge_probs, edge_types)
            pred_slots_list.append(pred_slots_next)

            # Decode predicted slots to frame (for frame-level loss)
            pred_frame, _, _ = self.decoder(pred_slots_next)
            pred_frames_list.append(pred_frame)

            # AUTOREGRESSIVE: use prediction as next input (not GT!)
            current_slots = pred_slots_next

        pred_slots = torch.stack(pred_slots_list, dim=1)    # [B, T', K, D]
        pred_frames = torch.stack(pred_frames_list, dim=1)  # [B, T', 3, H, W]

        return {
            "pred_slots": pred_slots,                        # [B, T', K, D]
            "target_slots": all_slots[:, 1:rollout_steps+1], # [B, T', K, D]
            "recon_t0": recon_t0,                            # [B, 3, H, W]
            "frame_t0": video[:, 0],                         # [B, 3, H, W]
            "pred_frames": pred_frames,                      # [B, T', 3, H, W]
            "target_frames": video[:, 1:rollout_steps+1],    # [B, T', 3, H, W]
            "all_slots": all_slots,                          # [B, T, K, D]
            "attn_maps": all_attn,                           # [B, T, K, N]
            "graph_infos": graph_info_list,
            "masks_t0": masks_t0,                            # [B, K, 1, H, W]
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
        dev = output["pred_slots"].device

        # L_recon: first frame reconstruction (trains encoder + decoder)
        losses["recon"] = F.mse_loss(output["recon_t0"], output["frame_t0"])

        # L_dynamics_frame: predicted future frames vs GT future frames
        # This is the KEY loss — forces the model to predict visually correct futures
        pred_frames = output["pred_frames"]
        target_frames = output["target_frames"]
        T_pred = min(pred_frames.shape[1], target_frames.shape[1])
        if T_pred > 0:
            # Weight later steps more (errors should compound)
            frame_losses = []
            for t in range(T_pred):
                weight = 1.0 + 0.5 * t  # step 0: 1.0, step 4: 3.0
                frame_losses.append(weight * F.mse_loss(pred_frames[:, t], target_frames[:, t]))
            losses["dynamics_frame"] = torch.stack(frame_losses).mean()
        else:
            losses["dynamics_frame"] = torch.tensor(0.0, device=dev)

        # L_dynamics_slot: slot-level prediction (auxiliary)
        pred_slots = output["pred_slots"]
        target_slots = output["target_slots"]
        T_slot = min(pred_slots.shape[1], target_slots.shape[1])
        if T_slot > 0:
            losses["dynamics_slot"] = F.mse_loss(pred_slots[:, :T_slot], target_slots[:, :T_slot])
        else:
            losses["dynamics_slot"] = torch.tensor(0.0, device=dev)

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
            losses["sparsity"] = torch.tensor(0.0, device=dev)
            losses["type_entropy"] = torch.tensor(0.0, device=dev)
            losses["min_connect"] = torch.tensor(0.0, device=dev)

        # Total weighted loss
        # dynamics_frame is the primary learning signal for graph + dynamics
        losses["total"] = (
            config.recon_weight * losses["recon"] +
            2.0 * losses["dynamics_frame"] +           # main signal: future frame prediction
            0.5 * losses["dynamics_slot"] +            # auxiliary: slot-level prediction
            config.sparsity_weight * 0.1 * losses["sparsity"] +
            config.entropy_weight * losses["type_entropy"] +
            1.0 * losses["min_connect"]
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
