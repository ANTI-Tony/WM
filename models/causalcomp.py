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
                 num_message_passing: int = 2, gumbel_temperature: float = 0.5,
                 use_dino: bool = False):
        super().__init__()

        self.resolution = resolution
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # Module 1: Object discovery
        self.use_dino = use_dino
        if use_dino:
            from .dino_encoder import DINOSlotEncoder
            self.encoder = DINOSlotEncoder(
                num_slots=num_slots,
                slot_dim=slot_dim,
            )
        else:
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

    def forward(self, video: torch.Tensor, rollout_steps: int = 1,
                positions: torch.Tensor = None):
        """
        Args:
            video: [B, T, 3, H, W] video clip
            rollout_steps: how many future steps to predict
            positions: [B, T, max_obj, 2] GT object positions (optional)
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
            "positions": positions,                          # [B, T, max_obj, 2] or None
        }

    def compute_loss(self, output: dict, config, epoch: int = 0,
                     collision_adj: torch.Tensor = None) -> dict:
        """Compute all loss terms.

        Two-phase training:
          Phase 1 (epoch < warmup): recon only (learn good slots first)
          Phase 2 (epoch >= warmup): recon + dynamics + graph supervision

        Args:
            output: dict from forward()
            config: TrainConfig with loss weights
            epoch: current epoch (for curriculum)
            collision_adj: [B, T, max_obj, max_obj] GT collision matrix (optional)
        Returns:
            losses: dict of named loss terms + total loss
        """
        losses = {}
        dev = output["pred_slots"].device
        phase2 = (epoch >= config.warmup_epochs)

        # L_recon: first frame reconstruction (always active)
        losses["recon"] = F.mse_loss(output["recon_t0"], output["frame_t0"])

        # L_dynamics_frame: predicted future frames vs GT future frames
        pred_frames = output["pred_frames"]
        target_frames = output["target_frames"]
        T_pred = min(pred_frames.shape[1], target_frames.shape[1])
        if T_pred > 0 and phase2:
            frame_losses = []
            for t in range(T_pred):
                weight = 1.0 + 0.5 * t
                frame_losses.append(weight * F.mse_loss(pred_frames[:, t], target_frames[:, t]))
            losses["dynamics_frame"] = torch.stack(frame_losses).mean()
        else:
            losses["dynamics_frame"] = torch.tensor(0.0, device=dev)

        # L_dynamics_slot: slot-level prediction (auxiliary)
        pred_slots = output["pred_slots"]
        target_slots = output["target_slots"]
        T_slot = min(pred_slots.shape[1], target_slots.shape[1])
        if T_slot > 0 and phase2:
            losses["dynamics_slot"] = F.mse_loss(pred_slots[:, :T_slot], target_slots[:, :T_slot])
        else:
            losses["dynamics_slot"] = torch.tensor(0.0, device=dev)

        # L_edge_supervision: use GT collision adjacency to supervise edges
        # Must match slots to objects first using attention map centroids
        losses["edge_sup"] = torch.tensor(0.0, device=dev)
        positions = output.get("positions", None)
        if (collision_adj is not None and positions is not None
                and phase2 and output["graph_infos"]):
            edge_probs = output["graph_infos"][0]["edge_probs"]  # [B, K, K]
            B_cur, K, _ = edge_probs.shape
            max_obj = collision_adj.shape[-1]
            num_obj = min(max_obj, K)
            attn_maps = output["attn_maps"][:, 0]  # [B, K, N] first frame
            H = W = int(attn_maps.shape[-1] ** 0.5)

            col_any = collision_adj.any(dim=1).float().to(dev)  # [B, max_obj, max_obj]

            # Coordinate grids [0,1]
            yy = torch.linspace(0, 1, H, device=dev)
            xx = torch.linspace(0, 1, W, device=dev)
            grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
            grid_x = grid_x.reshape(1, -1)  # [1, N]
            grid_y = grid_y.reshape(1, -1)

            # Slot centroids from attention maps
            attn_norm = attn_maps / (attn_maps.sum(dim=-1, keepdim=True) + 1e-8)
            slot_cx = (attn_norm * grid_x).sum(dim=-1)  # [B, K]
            slot_cy = (attn_norm * grid_y).sum(dim=-1)  # [B, K]
            slot_pos = torch.stack([slot_cx, slot_cy], dim=-1)  # [B, K, 2]

            obj_pos = positions[:, 0, :num_obj].to(dev)  # [B, num_obj, 2]

            # Match slots to objects per sample using distance matrix
            edge_sup_losses = []
            for b in range(B_cur):
                cost = torch.cdist(slot_pos[b], obj_pos[b])  # [K, num_obj]
                # Greedy match: for each object find nearest unused slot
                matched = {}  # obj_idx → slot_idx
                used = set()
                for _ in range(num_obj):
                    min_val = float('inf')
                    best_s, best_o = 0, 0
                    for o in range(num_obj):
                        if o in matched:
                            continue
                        for s in range(K):
                            if s in used:
                                continue
                            if cost[s, o].item() < min_val:
                                min_val = cost[s, o].item()
                                best_s, best_o = s, o
                    matched[best_o] = best_s
                    used.add(best_s)

                # Reorder collision target to slot space
                col_target_b = torch.zeros(K, K, device=dev)
                for oi, si in matched.items():
                    for oj, sj in matched.items():
                        if si != sj and oi < max_obj and oj < max_obj:
                            col_target_b[si, sj] = col_any[b, oi, oj]

                edge_sup_losses.append(
                    F.binary_cross_entropy(edge_probs[b], col_target_b)
                )

            if edge_sup_losses:
                losses["edge_sup"] = torch.stack(edge_sup_losses).mean()

        # Graph regularization losses
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
        if phase2:
            losses["total"] = (
                config.recon_weight * losses["recon"] +
                2.0 * losses["dynamics_frame"] +
                0.5 * losses["dynamics_slot"] +
                5.0 * losses["edge_sup"] +  # strong supervision to break uniform edges
                config.sparsity_weight * 0.1 * losses["sparsity"] +
                config.entropy_weight * losses["type_entropy"] +
                1.0 * losses["min_connect"]
            )
        else:
            # Phase 1: only reconstruction
            losses["total"] = config.recon_weight * losses["recon"]

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
