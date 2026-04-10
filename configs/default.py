"""CausalComp default configuration."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SlotConfig:
    num_slots: int = 8          # max objects in CLEVRER
    slot_dim: int = 128
    num_iterations: int = 3     # slot attention iterations
    hidden_dim: int = 128
    encoder_channels: int = 64


@dataclass
class CausalGraphConfig:
    edge_hidden_dim: int = 128
    num_interaction_types: int = 4   # collision, contact, approach, none
    gumbel_temperature: float = 0.5
    sparsity_weight: float = 0.01    # lambda_3: encourage sparse graphs
    entropy_weight: float = 0.01     # lambda_4: encourage crisp type assignment
    intervention_threshold: float = 0.1


@dataclass
class DynamicsConfig:
    hidden_dim: int = 128
    num_message_passing: int = 2
    residual: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 200
    warmup_epochs: int = 10
    # loss weights
    recon_weight: float = 1.0       # lambda for reconstruction
    dynamics_weight: float = 1.0    # lambda_1
    causal_weight: float = 0.1      # lambda_2
    sparsity_weight: float = 0.01   # lambda_3
    entropy_weight: float = 0.01    # lambda_4
    # schedule
    rollout_steps: int = 5          # multi-step prediction during training
    grad_clip: float = 1.0
    # logging
    log_interval: int = 50
    eval_interval: int = 5
    save_interval: int = 10
    wandb_project: str = "causalcomp"


@dataclass
class DataConfig:
    data_dir: str = "./data/clevrer"
    resolution: int = 128           # resize frames to 128x128
    num_frames: int = 16            # frames per clip
    frame_skip: int = 8             # sample every 8th frame (larger gap → more motion)
    num_workers: int = 4


@dataclass
class Config:
    slot: SlotConfig = field(default_factory=SlotConfig)
    causal: CausalGraphConfig = field(default_factory=CausalGraphConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: int = 42
    device: str = "cuda"
    exp_name: str = "causalcomp_v1"
