"""
Training script for baseline models comparison.

This script trains multiple baseline models and MOPI-HFRS for comparison,
generating results similar to the paper's Table 2.

Usage:
    python train_baselines.py --data_dir /path/to/data --epochs 500
    
    # Train specific model:
    python train_baselines.py --model lightgcn --epochs 500
    
    # Train all models:
    python train_baselines.py --model all --epochs 500 --num_runs 3
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
import numpy as np
import random
from sklearn.model_selection import train_test_split
from torch_geometric.utils import structured_negative_sampling
from tqdm import tqdm
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from models.baselines import get_baseline_model, BaseLightGCN, BaseGCN, BaseGraphSAGE, BaseGAT, NGCF


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch_geometric.seed_everything(seed)


def split_data(edge_index, edge_label_index, test_size=0.2, val_size=0.25, seed=42):
    """Split edges into train/val/test sets."""
    edges = edge_index.numpy().T
    train_edges, test_edges = train_test_split(edges, test_size=test_size, random_state=seed)
    train_edges, val_edges = train_test_split(train_edges, test_size=val_size, random_state=seed)

    train_edge_index = torch.LongTensor(train_edges).T
    val_edge_index = torch.LongTensor(val_edges).T
    test_edge_index = torch.LongTensor(test_edges).T

    return train_edge_index, val_edge_index, test_edge_index


def sample_mini_batch(batch_size, edge_index):
    """Sample a mini-batch for training."""
    edges = structured_negative_sampling(edge_index)
    edges = torch.stack(edges, dim=0)
    indices = random.choices(range(edges[0].shape[0]), k=batch_size)
    batch = edges[:, indices]
    user_indices, pos_item_indices, _ = batch[0], batch[1], batch[2]
    neg_item_indices = torch.randint(0, int(edge_index[1].max()), size=(batch_size,), dtype=torch.long)
    return user_indices, pos_item_indices, neg_item_indices


def bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, 
             neg_items_emb_final, neg_items_emb_0, lambda_val):
    """Compute BPR loss."""
    reg_loss = lambda_val * (
        users_emb_0.norm(2).pow(2) +
        pos_items_emb_0.norm(2).pow(2) +
        neg_items_emb_0.norm(2).pow(2)
    )

    pos_scores = torch.sum(users_emb_final * pos_items_emb_final, dim=-1)
    neg_scores = torch.sum(users_emb_final * neg_items_emb_final, dim=-1)

    bpr = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
    
    return bpr + reg_loss


def get_user_positive_items(edge_index):
    """Get positive items for each user."""
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)
    return user_pos_items


def calculate_metrics(users_emb, items_emb, edge_index, exclude_edge_indices, 
                     user_tags, food_tags, k=20):
    """Calculate evaluation metrics."""
    # Compute ratings
    rating = torch.matmul(users_emb, items_emb.T)
    
    # Exclude training items
    for exclude_edge_index in exclude_edge_indices:
        user_pos_items = get_user_positive_items(exclude_edge_index)
        for user, items in user_pos_items.items():
            rating[user, items] = -(1 << 10)
    
    # Get top-K items
    _, top_K_items = torch.topk(rating, k=k)
    
    # Get test users
    users = edge_index[0].unique()
    test_user_pos_items = get_user_positive_items(edge_index)
    
    # Calculate recall, precision, NDCG
    recalls = []
    precisions = []
    ndcgs = []
    
    for user in users:
        user_idx = user.item()
        if user_idx not in test_user_pos_items:
            continue
            
        ground_truth = set(test_user_pos_items[user_idx])
        recommended = set(top_K_items[user_idx].cpu().tolist())
        
        hits = len(ground_truth & recommended)
        
        recall = hits / len(ground_truth) if len(ground_truth) > 0 else 0
        precision = hits / k
        
        # NDCG
        dcg = 0
        for i, item in enumerate(top_K_items[user_idx].cpu().tolist()):
            if item in ground_truth:
                dcg += 1 / np.log2(i + 2)
        
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
        ndcg = dcg / idcg if idcg > 0 else 0
        
        recalls.append(recall)
        precisions.append(precision)
        ndcgs.append(ndcg)
    
    # Health score
    user_tags_batch = user_tags[users].cpu()
    recommended_items = top_K_items[users].cpu()
    food_tags_batch = food_tags[recommended_items].cpu()
    
    user_tags_expanded = user_tags_batch.unsqueeze(1)
    common_tag = torch.logical_and(user_tags_expanded, food_tags_batch).sum(dim=2) > 0
    health_score = common_tag.float().mean().item()
    
    # Average health tags
    tags_per_food = food_tags_batch.sum(dim=2)
    avg_health_tags = tags_per_food.float().mean().item()
    
    # Percentage of foods recommended
    unique_foods = top_K_items[users].cpu().flatten().unique()
    pct_foods = len(unique_foods) / items_emb.size(0)
    
    return {
        'recall': np.mean(recalls),
        'precision': np.mean(precisions),
        'ndcg': np.mean(ndcgs),
        'health_score': health_score,
        'avg_health_tags': avg_health_tags,
        'pct_foods': pct_foods
    }


def train_baseline(model, model_name, run_id, train_edge_index, val_edge_index, test_edge_index,
                   user_tags, food_tags, device, args):
    """Train a baseline model."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    best_recall = 0
    best_metrics = None
    best_model_state = None
    
    train_edge_index = train_edge_index.to(device)
    val_edge_index = val_edge_index.to(device)
    test_edge_index = test_edge_index.to(device)
    user_tags = user_tags.to(device)
    food_tags = food_tags.to(device)
    
    pbar = tqdm(range(args.epochs), desc="Training")
    
    for epoch in pbar:
        model.train()
        
        # Forward pass
        users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model(train_edge_index)
        
        # Sample mini-batch
        user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(
            args.batch_size, train_edge_index
        )
        user_indices = user_indices.to(device)
        pos_item_indices = pos_item_indices.to(device)
        neg_item_indices = neg_item_indices.to(device)
        
        # Get embeddings for batch
        users_emb_batch = users_emb_final[user_indices]
        users_emb_0_batch = users_emb_0[user_indices]
        pos_items_emb_batch = items_emb_final[pos_item_indices]
        pos_items_emb_0_batch = items_emb_0[pos_item_indices]
        neg_items_emb_batch = items_emb_final[neg_item_indices]
        neg_items_emb_0_batch = items_emb_0[neg_item_indices]
        
        # Compute loss
        loss = bpr_loss(
            users_emb_batch, users_emb_0_batch,
            pos_items_emb_batch, pos_items_emb_0_batch,
            neg_items_emb_batch, neg_items_emb_0_batch,
            args.lambda_val
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Learning rate decay
        if (epoch + 1) % args.lr_decay_step == 0:
            scheduler.step()
        
        # Evaluation
        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                users_emb_final, _, items_emb_final, _ = model(train_edge_index)
                
                metrics = calculate_metrics(
                    users_emb_final, items_emb_final,
                    val_edge_index, [train_edge_index],
                    user_tags, food_tags, k=args.k
                )
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'recall': f'{metrics["recall"]:.4f}'
                })
                
                if metrics['recall'] > best_recall:
                    best_recall = metrics['recall']
                    best_metrics = metrics.copy()
                    best_model_state = model.state_dict().copy()
    
    # Final test evaluation
    model.eval()
    with torch.no_grad():
        users_emb_final, _, items_emb_final, _ = model(train_edge_index)
        
        test_metrics = calculate_metrics(
            users_emb_final, items_emb_final,
            test_edge_index, [train_edge_index],
            user_tags, food_tags, k=args.k
        )
    
    # Save best model
    if args.save_dir and best_model_state is not None:
        save_path = os.path.join(args.save_dir, f'{model_name}_run{run_id}_best.pt')
        torch.save({
            'model_state_dict': best_model_state,
            'model_name': model_name,
            'run_id': run_id,
            'best_val_recall': best_recall,
            'test_metrics': test_metrics,
            'args': vars(args)
        }, save_path)
        print(f"  Saved best model to {save_path}")
    
    return test_metrics, best_model_state


def main(args):
    print(f"\n{'='*60}")
    print("Baseline Models Comparison")
    print(f"{'='*60}\n")
    
    # Load data
    data_path = os.path.join(args.data_dir, 'benchmark_macro.pt')
    print(f"Loading data from: {data_path}")
    graph = torch.load(data_path, weights_only=False)
    
    num_users = graph['user'].num_nodes
    num_foods = graph['food'].num_nodes
    edge_index = graph[('user', 'eats', 'food')].edge_index
    edge_label_index = graph[('user', 'eats', 'food')].edge_label_index
    user_tags = graph['user'].tags
    food_tags = graph['food'].tags
    
    print(f"Users: {num_users}, Foods: {num_foods}")
    print(f"Edges: {edge_index.shape[1]}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Models will be saved to: {args.save_dir}")
    
    # Models to train
    if args.model == 'all':
        model_names = ['gcn', 'graphsage', 'gat', 'lightgcn', 'ngcf']
    else:
        model_names = [args.model]
    
    # Results storage
    all_results = {}
    
    for model_name in model_names:
        print(f"\n{'='*40}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*40}")
        
        run_results = []
        
        for run in range(args.num_runs):
            print(f"\nRun {run + 1}/{args.num_runs}")
            
            # Set seed for reproducibility
            seed = args.seed + run
            set_seed(seed)
            
            # Split data
            train_edge_index, val_edge_index, test_edge_index = split_data(
                edge_index, edge_label_index, seed=seed
            )
            
            # Create model
            model = get_baseline_model(
                model_name,
                num_users,
                num_foods,
                embedding_dim=args.embedding_dim,
                num_layers=args.num_layers
            )
            
            # Train
            metrics, best_state = train_baseline(
                model, model_name, run + 1, 
                train_edge_index, val_edge_index, test_edge_index,
                user_tags, food_tags, device, args
            )
            
            run_results.append(metrics)
            
            print(f"  Recall@{args.k}: {metrics['recall']*100:.2f}")
            print(f"  NDCG@{args.k}: {metrics['ndcg']*100:.2f}")
            print(f"  H-Score@{args.k}: {metrics['health_score']*100:.2f}")
            print(f"  AvgTags@{args.k}: {metrics['avg_health_tags']:.2f}")
            print(f"  %Foods@{args.k}: {metrics['pct_foods']*100:.3f}")
        
        # Aggregate results
        all_results[model_name] = {
            'recall': {
                'mean': np.mean([r['recall'] for r in run_results]) * 100,
                'std': np.std([r['recall'] for r in run_results]) * 100
            },
            'ndcg': {
                'mean': np.mean([r['ndcg'] for r in run_results]) * 100,
                'std': np.std([r['ndcg'] for r in run_results]) * 100
            },
            'health_score': {
                'mean': np.mean([r['health_score'] for r in run_results]) * 100,
                'std': np.std([r['health_score'] for r in run_results]) * 100
            },
            'avg_health_tags': {
                'mean': np.mean([r['avg_health_tags'] for r in run_results]),
                'std': np.std([r['avg_health_tags'] for r in run_results])
            },
            'pct_foods': {
                'mean': np.mean([r['pct_foods'] for r in run_results]) * 100,
                'std': np.std([r['pct_foods'] for r in run_results]) * 100
            }
        }
    
    # Print final table
    k = args.k
    print(f"\n{'='*100}")
    print(f"RESULTS TABLE (Nutrition-macro only, K={k})")
    print(f"{'='*100}")
    print(f"{'Model':<15} {'Recall@'+str(k):>15} {'NDCG@'+str(k):>15} {'H-Score@'+str(k):>15} {'AvgTags@'+str(k):>12} {'%Foods@'+str(k):>12}")
    print("-" * 100)
    
    for model_name, results in all_results.items():
        recall = f"{results['recall']['mean']:.2f}±{results['recall']['std']:.2f}"
        ndcg = f"{results['ndcg']['mean']:.2f}±{results['ndcg']['std']:.2f}"
        health = f"{results['health_score']['mean']:.2f}±{results['health_score']['std']:.2f}"
        tags = f"{results['avg_health_tags']['mean']:.2f}±{results['avg_health_tags']['std']:.2f}"
        foods = f"{results['pct_foods']['mean']:.3f}±{results['pct_foods']['std']:.3f}"
        
        print(f"{model_name.upper():<15} {recall:>15} {ndcg:>15} {health:>15} {tags:>12} {foods:>12}")
    
    print(f"{'='*100}")
    
    # Print in paper markdown format
    print(f"\nMarkdown Table Format:")
    print(f"| Model | Recall@{k} | NDCG@{k} | H-Score@{k} | AvgTags@{k} | %Foods@{k} |")
    print(f"|-------|----------|---------|------------|------------|-----------|")
    for model_name, results in all_results.items():
        print(f"| {model_name.upper()} | "
              f"{results['recall']['mean']:.2f}±{results['recall']['std']:.2f} | "
              f"{results['ndcg']['mean']:.2f}±{results['ndcg']['std']:.2f} | "
              f"{results['health_score']['mean']:.2f}±{results['health_score']['std']:.2f} | "
              f"{results['avg_health_tags']['mean']:.2f}±{results['avg_health_tags']['std']:.2f} | "
              f"{results['pct_foods']['mean']:.3f}±{results['pct_foods']['std']:.3f} |")
    
    # Save aggregated results
    if args.save_dir:
        results_file = os.path.join(args.save_dir, f"baseline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAggregated results saved to: {results_file}")
        
        # List all saved models
        print(f"\nSaved model checkpoints:")
        for f in sorted(os.listdir(args.save_dir)):
            if f.endswith('.pt'):
                print(f"  - {f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train baseline models for comparison')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, 
                        default='MOPI-HFRS_gdrive/processed_data',
                        help='Directory containing benchmark_macro.pt')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='all',
                        choices=['gcn', 'graphsage', 'gat', 'lightgcn', 'ngcf', 'all'],
                        help='Model to train')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--lambda_val', type=float, default=1e-6,
                        help='L2 regularization coefficient')
    parser.add_argument('--lr_decay_step', type=int, default=200,
                        help='LR decay step')
    parser.add_argument('--eval_every', type=int, default=50,
                        help='Evaluation frequency')
    parser.add_argument('--k', type=int, default=20,
                        help='Top-K for evaluation')
    
    # Experiment arguments
    parser.add_argument('--num_runs', type=int, default=3,
                        help='Number of runs for averaging')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    
    args = parser.parse_args()
    
    main(args)

