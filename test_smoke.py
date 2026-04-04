"""
Smoke test: verify all modules work end-to-end with dummy data.
Run locally (no GPU needed, no real data needed).

Usage:
    cd ~/Desktop/CausalComp
    python test_smoke.py
"""

import torch
import sys

def test_slot_attention():
    print("Testing SlotAttention...", end=" ")
    from models.slot_attention import SlotAttention, SlotEncoder

    # Test SlotAttention alone
    sa = SlotAttention(num_slots=6, slot_dim=64, input_dim=32)
    x = torch.randn(2, 100, 32)  # [B, N, D_in]
    slots, attn = sa(x)
    assert slots.shape == (2, 6, 64), f"Expected (2,6,64), got {slots.shape}"
    assert attn.shape == (2, 6, 100), f"Expected (2,6,100), got {attn.shape}"

    # Test SlotEncoder with single image
    encoder = SlotEncoder(resolution=64, num_slots=6, slot_dim=64, encoder_channels=32)
    img = torch.randn(2, 3, 64, 64)
    slots, attn = encoder(img)
    assert slots.shape == (2, 6, 64)

    # Test SlotEncoder with video
    video = torch.randn(2, 4, 3, 64, 64)  # [B, T, C, H, W]
    slots, attn = encoder(video)
    assert slots.shape == (2, 4, 6, 64), f"Expected (2,4,6,64), got {slots.shape}"

    print("OK")


def test_causal_graph():
    print("Testing CausalGraphDiscovery...", end=" ")
    from models.causal_graph import CausalGraphDiscovery

    cg = CausalGraphDiscovery(slot_dim=64, num_interaction_types=4, hidden_dim=64)
    slots = torch.randn(2, 6, 64)

    edge_probs, edge_types, graph_info = cg(slots)
    assert edge_probs.shape == (2, 6, 6)
    assert edge_types.shape == (2, 6, 6, 4)
    assert edge_probs.diagonal(dim1=-2, dim2=-1).sum() == 0  # no self-loops

    # Test loss computation
    losses = cg.compute_loss(graph_info)
    assert "sparsity" in losses
    assert "type_entropy" in losses

    print("OK")


def test_modular_dynamics():
    print("Testing ModularCausalDynamics...", end=" ")
    from models.modular_dynamics import ModularCausalDynamics

    dyn = ModularCausalDynamics(slot_dim=64, num_interaction_types=4, hidden_dim=64)
    slots = torch.randn(2, 6, 64)
    edge_probs = torch.rand(2, 6, 6) * 0.5
    edge_types = torch.randn(2, 6, 6, 4).softmax(dim=-1)

    next_slots = dyn(slots, edge_probs, edge_types)
    assert next_slots.shape == (2, 6, 64)

    # Test rollout
    traj = dyn.rollout(slots, edge_probs, edge_types, steps=5)
    assert traj.shape == (2, 6, 6, 64), f"Expected (2,6,6,64), got {traj.shape}"

    # Test no-graph prediction (for interventional verification)
    pred = dyn.predict_no_graph(slots)
    assert pred.shape == (2, 6, 64)

    print("OK")


def test_decoder():
    print("Testing SpatialBroadcastDecoder...", end=" ")
    from models.decoder import SpatialBroadcastDecoder

    dec = SpatialBroadcastDecoder(slot_dim=64, resolution=64, channels=32)
    slots = torch.randn(2, 6, 64)

    recon, masks, slot_recons = dec(slots)
    assert recon.shape == (2, 3, 64, 64)
    assert masks.shape == (2, 6, 1, 64, 64)
    assert slot_recons.shape == (2, 6, 3, 64, 64)

    # Masks should sum to ~1 at each pixel
    mask_sum = masks.sum(dim=1)
    assert torch.allclose(mask_sum, torch.ones_like(mask_sum), atol=1e-5)

    print("OK")


def test_full_model():
    print("Testing CausalComp (full model)...", end=" ")
    from models.causalcomp import CausalComp
    from configs import Config

    model = CausalComp(
        resolution=64, num_slots=6, slot_dim=64,
        num_interaction_types=4, encoder_channels=32,
        dynamics_hidden=64, num_message_passing=1,
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"({num_params:,} params)", end=" ")

    # Forward pass
    video = torch.randn(2, 8, 3, 64, 64)  # [B, T, C, H, W]
    output = model(video, rollout_steps=3)

    assert "pred_slots" in output
    assert "recon_frames" in output
    assert "graph_infos" in output

    # Loss computation
    config = Config()
    losses = model.compute_loss(output, config.train)
    assert "total" in losses
    assert losses["total"].requires_grad

    # Backward pass
    losses["total"].backward()
    print("OK")

    # Check gradients flow to all modules
    has_grad = {
        "encoder": any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.encoder.parameters()),
        "graph": any(p.grad is not None and p.grad.abs().sum() > 0
                     for p in model.graph_discovery.parameters()),
        "dynamics": any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in model.dynamics.parameters()),
        "decoder": any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in model.decoder.parameters()),
    }
    print(f"  Gradient flow: {has_grad}")


def test_predict_trajectory():
    print("Testing predict_trajectory...", end=" ")
    from models.causalcomp import CausalComp

    model = CausalComp(
        resolution=64, num_slots=6, slot_dim=64,
        num_interaction_types=4, encoder_channels=32,
        dynamics_hidden=64,
    )
    model.eval()

    context = torch.randn(2, 4, 3, 64, 64)
    pred_slots, pred_frames = model.predict_trajectory(context, future_steps=5)

    assert pred_slots.shape == (2, 5, 6, 64)
    assert pred_frames.shape == (2, 5, 3, 64, 64)
    print("OK")


if __name__ == "__main__":
    print("=" * 60)
    print("CausalComp Smoke Test")
    print("=" * 60)

    try:
        test_slot_attention()
        test_causal_graph()
        test_modular_dynamics()
        test_decoder()
        test_full_model()
        test_predict_trajectory()
        print("=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
