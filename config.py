"""
Configuration for MOPI-HFRS training.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    embedding_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    feature_threshold: float = 0.3


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Basic training
    epochs: int = 500
    batch_size: int = 2048
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    
    # Learning rate schedule
    lr_decay_gamma: float = 0.95
    lr_decay_step: int = 200
    
    # Evaluation
    eval_every: int = 50
    k: int = 20  # Top-K for evaluation
    
    # Regularization
    lambda_val: float = 1e-6
    
    # Pareto optimization
    use_pareto: bool = True
    normalization_type: str = 'l2'
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Reproducibility
    seed: int = 42
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = 'mopi-hfrs'
    wandb_entity: Optional[str] = None
    
    # Checkpointing
    save_dir: str = 'checkpoints'
    save_every: int = 100


@dataclass
class DataConfig:
    """Data configuration."""
    data_dir: str = 'MOPI-HFRS_gdrive/processed_data'
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    normalize_features: bool = True
    
    # Benchmark type: 'macro' (7 nutrients) or 'all' (16 nutrients)
    # 'macro': calories, carbohydrates, protein, saturated fat, cholesterol, sugar, dietary fiber
    # 'all': macro + sodium, potassium, phosphorus, iron, calcium, folic acid, vitamin C, D, B12
    benchmark_type: str = 'macro'
    
    # Whether to use pre-processed .pt file or load from CSV
    use_pt_file: bool = False
    pt_file: Optional[str] = None


@dataclass
class Config:
    """Full configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    def __post_init__(self):
        """Validate configuration."""
        assert 0 < self.data.train_ratio < 1
        assert 0 < self.data.val_ratio < 1
        assert self.data.train_ratio + self.data.val_ratio < 1
        assert self.model.embedding_dim > 0
        assert self.model.num_layers > 0


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def get_colab_config() -> Config:
    """Get configuration optimized for Colab with A100."""
    config = Config()
    
    # Larger batch size for A100
    config.training.batch_size = 4096
    
    # More frequent evaluation
    config.training.eval_every = 25
    
    # Enable mixed precision
    config.training.device = 'cuda'
    
    return config


def get_debug_config() -> Config:
    """Get configuration for debugging (small and fast)."""
    config = Config()
    
    config.training.epochs = 10
    config.training.batch_size = 256
    config.training.eval_every = 5
    config.model.embedding_dim = 32
    config.model.num_layers = 2
    
    return config

