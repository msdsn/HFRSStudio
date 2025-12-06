"""
Training script for MOPI-HFRS.

This script provides a complete training pipeline that can be run
locally or on Google Colab with A100 GPU.

Usage:
    python train.py --data_dir path/to/data --epochs 500
    
For Colab:
    !python train.py --data_dir /content/drive/MyDrive/data --use_colab
"""

import os
import sys
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_sparse import SparseTensor
from torch_geometric.utils import structured_negative_sampling
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config, get_default_config, get_colab_config, get_debug_config
from data.data_loader import HFRSDataset, HFRSDatasetFromPT, create_hetero_graph
from models.mopi_hfrs import MOPI_HFRS, create_model
from utils.losses import bpr_loss, health_loss, diversity_loss
from utils.pareto import pareto_loss, ParetoMTL
from utils.metrics import get_metrics, evaluate_model, MetricTracker


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sample_mini_batch(
    batch_size: int,
    edge_index: torch.Tensor,
    num_items: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample a mini-batch for training.
    
    Args:
        batch_size: Size of the batch
        edge_index: Edge index tensor
        num_items: Number of items (for negative sampling)
        
    Returns:
        Tuple of (user_indices, pos_item_indices, neg_item_indices)
    """
    # Use structured negative sampling
    edges = structured_negative_sampling(edge_index)
    edges = torch.stack(edges, dim=0)
    
    # Random sample indices
    num_edges = edges.size(1)
    indices = random.choices(range(num_edges), k=batch_size)
    
    batch = edges[:, indices]
    user_indices = batch[0]
    pos_item_indices = batch[1]
    
    # Random negative items
    neg_item_indices = torch.randint(0, num_items, (batch_size,), dtype=torch.long)
    
    return user_indices, pos_item_indices, neg_item_indices


def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    feature_dict: Dict[str, torch.Tensor],
    edge_index: torch.Tensor,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    user_tags: torch.Tensor,
    food_tags: torch.Tensor,
    user_features: torch.Tensor,
    food_features: torch.Tensor,
    config: Config,
    device: torch.device
) -> Dict[str, float]:
    """
    Train for one epoch - follows original MOPI-HFRS training loop exactly.
    
    Original flow from main.py:
    1. Forward pass to get all embeddings
    2. Mini-batch sampling
    3. Get batch embeddings and compute loss
    4. Backward and optimize
    
    Args:
        model: MOPI-HFRS model
        optimizer: Optimizer
        feature_dict: Feature dictionary
        edge_index: Training edge index
        pos_edge_index: Positive edges (for this split)
        neg_edge_index: Negative edges (for this split)
        user_tags: User health tags
        food_tags: Food health tags
        user_features: User features
        food_features: Food features
        config: Training configuration
        device: Device
        
    Returns:
        Dictionary of loss values
    """
    model.train()
    
    # Forward pass - get all embeddings first (same as original)
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model(
        feature_dict, edge_index, pos_edge_index, neg_edge_index
    )
    
    # Mini-batch sampling (same as original sample_mini_batch)
    user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(
        config.training.batch_size,
        edge_index,
        items_emb_final.size(0)
    )
    
    # Move indices to device
    user_indices = user_indices.to(device)
    pos_item_indices = pos_item_indices.to(device)
    neg_item_indices = neg_item_indices.to(device)
    
    # Get batch embeddings (same variable names as original)
    users_emb_final_batch = users_emb_final[user_indices]
    users_emb_0_batch = users_emb_0[user_indices]
    pos_items_emb_final_batch = items_emb_final[pos_item_indices]
    pos_items_emb_0_batch = items_emb_0[pos_item_indices]
    neg_items_emb_final_batch = items_emb_final[neg_item_indices]
    neg_items_emb_0_batch = items_emb_0[neg_item_indices]
    
    # Get batch tags
    user_tags_batch = user_tags[user_indices]
    pos_item_tags_batch = food_tags[pos_item_indices]
    neg_item_tags_batch = food_tags[neg_item_indices]
    
    # Pad user features to match food features (same as original)
    user_features_batch = user_features[user_indices]
    user_features_batch = torch.nn.functional.pad(
        user_features_batch, 
        (0, food_features.size(1) - user_features_batch.size(1))
    )
    
    pos_item_features_batch = food_features[pos_item_indices]
    neg_item_features_batch = food_features[neg_item_indices]
    
    # Compute loss using Pareto optimization (same as original)
    if config.training.use_pareto:
        train_loss, loss_data, _ = pareto_loss(
            model,
            users_emb_final_batch, users_emb_0_batch,
            pos_items_emb_final_batch, pos_items_emb_0_batch,
            neg_items_emb_final_batch, neg_items_emb_0_batch,
            user_features_batch, pos_item_features_batch, neg_item_features_batch,
            user_tags_batch, pos_item_tags_batch, neg_item_tags_batch,
            config.training.lambda_val,
            config.training.normalization_type
        )
    else:
        # Simple weighted sum (fallback)
        l_bpr = bpr_loss(
            users_emb_final_batch, users_emb_0_batch,
            pos_items_emb_final_batch, pos_items_emb_0_batch,
            neg_items_emb_final_batch, neg_items_emb_0_batch,
            config.training.lambda_val
        )
        l_health = health_loss(
            users_emb_final_batch, pos_items_emb_final_batch, neg_items_emb_final_batch,
            user_tags_batch, pos_item_tags_batch, neg_item_tags_batch
        )
        l_diversity = diversity_loss(
            users_emb_final_batch, pos_items_emb_final_batch, neg_items_emb_final_batch,
            user_features_batch, pos_item_features_batch, neg_item_features_batch
        )
        
        train_loss = l_bpr + l_health + l_diversity
        loss_data = {'bpr': l_bpr, 'health': l_health, 'diversity': l_diversity}
    
    # Backward pass
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    return {k: v.item() for k, v in loss_data.items()}


def evaluate(
    model: nn.Module,
    feature_dict: Dict[str, torch.Tensor],
    edge_index: torch.Tensor,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    user_tags: torch.Tensor,
    food_tags: torch.Tensor,
    exclude_edges: torch.Tensor,
    k: int = 20
) -> Dict[str, float]:
    """
    Evaluate model - follows original MOPI-HFRS eval() function.
    
    Args:
        model: MOPI-HFRS model
        feature_dict: Feature dictionary
        edge_index: Evaluation edge index
        pos_edge_index: Positive edges for this split
        neg_edge_index: Negative edges for this split
        user_tags: User health tags
        food_tags: Food health tags
        exclude_edges: Edges to exclude (e.g., training edges)
        k: Top-K for metrics
        
    Returns:
        Dictionary of metric values
    """
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model(
            feature_dict, edge_index, pos_edge_index, neg_edge_index
        )
        
        # Compute metrics using the embeddings
        metrics = get_metrics(
            users_emb_final,
            items_emb_final,
            user_tags,
            food_tags,
            edge_index,
            [exclude_edges],
            k
        )
    
    return metrics


def train(config: Config, use_pt_file: bool = False, pt_file: str = None):
    """
    Main training function.
    
    Args:
        config: Training configuration
        use_pt_file: Whether to use pre-processed .pt file
        pt_file: Path to .pt file (if use_pt_file is True)
    """
    # Set seed
    set_seed(config.training.seed)
    
    # Device
    device = torch.device(config.training.device)
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    
    # Determine which .pt file to use based on benchmark_type
    pt_file_path = None
    if use_pt_file and pt_file:
        pt_file_path = pt_file
    else:
        # Auto-detect .pt file from data_dir
        import os
        data_dir = config.data.data_dir
        benchmark_type = config.data.benchmark_type
        pt_filename = f"benchmark_{benchmark_type}.pt"
        potential_pt = os.path.join(data_dir, pt_filename)
        if os.path.exists(potential_pt):
            pt_file_path = potential_pt
            print(f"Found benchmark file: {pt_file_path}")
    
    if pt_file_path:
        # Use pre-processed .pt file (RECOMMENDED - matches original paper)
        dataset = HFRSDatasetFromPT(
            pt_file_path,
            test_size=config.data.test_size,
            val_size=config.data.val_size,
            seed=config.training.seed
        )
    else:
        # Fallback to CSV files (may produce different results)
        print("WARNING: Using CSV files instead of .pt benchmark. Results may differ from paper.")
        dataset = HFRSDataset(
            config.data.data_dir,
            test_size=config.data.test_size,
            val_size=config.data.val_size,
            seed=config.training.seed,
            normalize=config.data.normalize_features,
            benchmark_type=config.data.benchmark_type
        )
    
    # Move data to device
    dataset = dataset.to(device)
    
    # Get data
    train_edge_index = dataset.splits['train_edge_index']
    val_edge_index = dataset.splits['val_edge_index']
    test_edge_index = dataset.splits['test_edge_index']
    
    # Get pos/neg edge indices for each split (following original split_data_new)
    if 'pos_train_edge_index' in dataset.splits:
        # New format with per-split pos/neg edges
        pos_train_edge_index = dataset.splits['pos_train_edge_index'].to(device)
        neg_train_edge_index = dataset.splits['neg_train_edge_index'].to(device)
        pos_val_edge_index = dataset.splits['pos_val_edge_index'].to(device)
        neg_val_edge_index = dataset.splits['neg_val_edge_index'].to(device)
        pos_test_edge_index = dataset.splits['pos_test_edge_index'].to(device)
        neg_test_edge_index = dataset.splits['neg_test_edge_index'].to(device)
    elif 'pos_edge_index' in dataset.splits:
        # Old format - use same pos/neg for all splits
        pos_train_edge_index = dataset.splits['pos_edge_index'].to(device)
        neg_train_edge_index = dataset.splits['neg_edge_index'].to(device)
        pos_val_edge_index = pos_train_edge_index
        neg_val_edge_index = neg_train_edge_index
        pos_test_edge_index = pos_train_edge_index
        neg_test_edge_index = neg_train_edge_index
    else:
        # From graph
        pos_train_edge_index = dataset.graph['user', 'eats', 'food'].pos_edge_index.to(device)
        neg_train_edge_index = dataset.graph['user', 'eats', 'food'].neg_edge_index.to(device)
        pos_val_edge_index = pos_train_edge_index
        neg_val_edge_index = neg_train_edge_index
        pos_test_edge_index = pos_train_edge_index
        neg_test_edge_index = neg_train_edge_index
    
    feature_dict = dataset.get_feature_dict()
    user_tags = dataset.user_tags
    food_tags = dataset.food_tags
    user_features = dataset.user_features
    food_features = dataset.food_features
    
    print(f"Users: {dataset.num_users}, Foods: {dataset.num_foods}")
    print(f"Train edges: {train_edge_index.shape[1]}")
    print(f"Val edges: {val_edge_index.shape[1]}")
    print(f"Test edges: {test_edge_index.shape[1]}")
    
    # Create model
    print("Creating model...")
    model = create_model(
        num_users=dataset.num_users,
        num_foods=dataset.num_foods,
        user_feature_dim=user_features.shape[1],
        food_feature_dim=food_features.shape[1],
        config={
            'embedding_dim': config.model.embedding_dim,
            'num_layers': config.model.num_layers,
            'num_heads': config.model.num_heads,
            'feature_threshold': config.model.feature_threshold
        }
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer and scheduler (matching original: Adam + ExponentialLR)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate
        # Note: original doesn't use weight_decay in optimizer
    )
    # Original uses ExponentialLR with gamma=0.95
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.95
    )
    
    # Metric tracker
    metric_tracker = MetricTracker([
        'recall', 'precision', 'ndcg', 
        'health_score', 'avg_health_tags', 'pct_foods_recommended'
    ])
    
    # Create save directory
    save_dir = Path(config.training.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("Starting training...")
    best_recall = 0.0
    
    for epoch in tqdm(range(config.training.epochs), desc="Training"):
        # Train using training split's pos/neg edges
        loss_dict = train_epoch(
            model, optimizer,
            feature_dict, train_edge_index, pos_train_edge_index, neg_train_edge_index,
            user_tags, food_tags, user_features, food_features,
            config, device
        )
        
        # Update scheduler every lr_decay_step epochs (original: 200)
        if (epoch + 1) % config.training.lr_decay_step == 0:
            scheduler.step()
        
        # Evaluate on validation set
        if (epoch + 1) % config.training.eval_every == 0:
            val_metrics = evaluate(
                model, feature_dict,
                val_edge_index, pos_val_edge_index, neg_val_edge_index,
                user_tags, food_tags, neg_train_edge_index,  # Exclude negative training edges
                config.training.k
            )
            
            metric_tracker.update(val_metrics)
            
            print(f"\nEpoch {epoch + 1}/{config.training.epochs}")
            print(f"  Loss: BPR={loss_dict.get('bpr', 0):.4f}, "
                  f"Health={loss_dict.get('health', 0):.4f}, "
                  f"Diversity={loss_dict.get('diversity', 0):.4f}")
            print(f"  Val: {metric_tracker.summary()}")
            
            # Save best model
            if val_metrics['recall'] > best_recall:
                best_recall = val_metrics['recall']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': val_metrics,
                    'config': config
                }, save_dir / 'best_model.pt')
                print(f"  Saved best model (recall={best_recall:.4f})")
        
        # Save checkpoint
        if (epoch + 1) % config.training.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, save_dir / f'checkpoint_{epoch + 1}.pt')
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate(
        model, feature_dict,
        test_edge_index, pos_test_edge_index, neg_test_edge_index,
        user_tags, food_tags, neg_train_edge_index,  # Exclude negative training edges
        config.training.k
    )
    
    print("\nTest Results:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_metrics': test_metrics,
        'config': config
    }, save_dir / 'final_model.pt')
    
    print(f"\nTraining complete! Models saved to {save_dir}")
    
    return test_metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train MOPI-HFRS model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='MOPI-HFRS_gdrive/processed_data',
                        help='Path to data directory')
    parser.add_argument('--benchmark_type', type=str, default='macro',
                        choices=['macro', 'all'],
                        help='Benchmark type: macro (7 nutrients) or all (16 nutrients)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GCN layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--feature_threshold', type=float, default=0.3,
                        help='Feature similarity threshold')
    
    # Evaluation arguments
    parser.add_argument('--k', type=int, default=20,
                        help='Top-K for evaluation')
    parser.add_argument('--eval_every', type=int, default=50,
                        help='Evaluate every N epochs')
    
    # Other arguments
    parser.add_argument('--use_pareto', action='store_true', default=True,
                        help='Use Pareto optimization')
    parser.add_argument('--no_pareto', action='store_false', dest='use_pareto',
                        help='Disable Pareto optimization')
    parser.add_argument('--use_colab', action='store_true',
                        help='Use Colab-optimized settings')
    parser.add_argument('--debug', action='store_true',
                        help='Use debug settings (small and fast)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save models')
    parser.add_argument('--pt_file', type=str, default=None,
                        help='Path to pre-processed .pt benchmark file')
    
    args = parser.parse_args()
    
    # Get configuration
    if args.debug:
        config = get_debug_config()
    elif args.use_colab:
        config = get_colab_config()
    else:
        config = get_default_config()
    
    # Override with command line arguments
    config.data.data_dir = args.data_dir
    config.data.benchmark_type = args.benchmark_type
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.seed = args.seed
    config.training.k = args.k
    config.training.eval_every = args.eval_every
    config.training.use_pareto = args.use_pareto
    config.training.save_dir = args.save_dir
    config.model.embedding_dim = args.embedding_dim
    config.model.num_layers = args.num_layers
    config.model.num_heads = args.num_heads
    config.model.feature_threshold = args.feature_threshold
    
    # Train
    use_pt = args.pt_file is not None
    train(config, use_pt_file=use_pt, pt_file=args.pt_file)


if __name__ == '__main__':
    main()

