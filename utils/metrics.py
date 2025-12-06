"""
Evaluation metrics for MOPI-HFRS.

Implements metrics for:
1. Recommendation quality: Recall@K, Precision@K, NDCG@K
2. Health awareness: H-Score, Average Health Tags
3. Diversity: Percentage of Foods Recommended
"""

import torch
import numpy as np
from torch import Tensor
from typing import Dict, List, Tuple, Optional


def get_user_positive_items(edge_index: Tensor) -> Dict[int, List[int]]:
    """
    Generate dictionary of positive items for each user.
    
    Args:
        edge_index: Edge index tensor (2, num_edges)
        
    Returns:
        Dictionary mapping user_id to list of positive item_ids
    """
    user_pos_items = {}
    
    for i in range(edge_index.shape[1]):
        user = edge_index[0, i].item()
        item = edge_index[1, i].item()
        
        if user not in user_pos_items:
            user_pos_items[user] = []
        
        user_pos_items[user].append(item)
    
    return user_pos_items


def recall_at_k(
    ground_truth: List[List[int]],
    predictions: Tensor,
    k: int
) -> float:
    """
    Compute Recall@K.
    
    Args:
        ground_truth: List of lists containing relevant items for each user
        predictions: Binary tensor indicating correct predictions (num_users, k)
        k: Number of recommendations
        
    Returns:
        Recall@K value
    """
    num_correct = torch.sum(predictions, dim=-1)
    num_relevant = torch.tensor([len(gt) for gt in ground_truth], dtype=torch.float32)
    
    # Avoid division by zero
    num_relevant = torch.clamp(num_relevant, min=1)
    
    recall = torch.mean(num_correct / num_relevant)
    
    return recall.item()


def precision_at_k(
    ground_truth: List[List[int]],
    predictions: Tensor,
    k: int
) -> float:
    """
    Compute Precision@K.
    
    Args:
        ground_truth: List of lists containing relevant items for each user
        predictions: Binary tensor indicating correct predictions (num_users, k)
        k: Number of recommendations
        
    Returns:
        Precision@K value
    """
    num_correct = torch.sum(predictions, dim=-1)
    precision = torch.mean(num_correct) / k
    
    return precision.item()


def ndcg_at_k(
    ground_truth: List[List[int]],
    predictions: Tensor,
    k: int
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG)@K.
    
    Args:
        ground_truth: List of lists containing relevant items for each user
        predictions: Binary tensor indicating correct predictions (num_users, k)
        k: Number of recommendations
        
    Returns:
        NDCG@K value
    """
    num_users = len(ground_truth)
    
    # Compute ideal DCG
    test_matrix = torch.zeros((num_users, k))
    for i, items in enumerate(ground_truth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    
    # Discount factors
    discounts = 1.0 / torch.log2(torch.arange(2, k + 2, dtype=torch.float32))
    
    # IDCG
    idcg = torch.sum(test_matrix * discounts, dim=1)
    idcg[idcg == 0.0] = 1.0  # Avoid division by zero
    
    # DCG
    dcg = torch.sum(predictions * discounts, dim=1)
    
    # NDCG
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.0
    
    return torch.mean(ndcg).item()


def health_score(
    users: Tensor,
    top_k_items: Tensor,
    user_tags: Tensor,
    food_tags: Tensor
) -> float:
    """
    Compute health score based on tag matching.
    
    Measures the percentage of recommended foods that share at least
    one common health tag with the user.
    
    Args:
        users: User indices
        top_k_items: Top-K recommended items for each user (num_users, k)
        user_tags: User health tags (num_users, num_tags)
        food_tags: Food health tags (num_foods, num_tags)
        
    Returns:
        Health score (0-1)
    """
    # Get user tags
    user_tags_batch = user_tags[users].cpu()  # (num_users, num_tags)
    
    # Get food tags for recommended items
    recommended_items = top_k_items[users].cpu()  # (num_users, k)
    food_tags_batch = food_tags[recommended_items].cpu()  # (num_users, k, num_tags)
    
    # Expand user tags for broadcasting
    user_tags_expanded = user_tags_batch.unsqueeze(1)  # (num_users, 1, num_tags)
    
    # Check for at least one common tag
    common_tags = torch.logical_and(user_tags_expanded > 0, food_tags_batch > 0)
    has_common_tag = common_tags.sum(dim=2) > 0  # (num_users, k)
    
    # Compute ratio of healthy foods per user
    healthy_ratio = has_common_tag.float().mean(dim=1)  # (num_users,)
    
    # Average across all users
    score = healthy_ratio.mean().item()
    
    return score


def average_health_tags(
    users: Tensor,
    top_k_items: Tensor,
    food_tags: Tensor
) -> float:
    """
    Compute average number of health tags in recommended foods.
    
    Args:
        users: User indices
        top_k_items: Top-K recommended items (num_users, k)
        food_tags: Food health tags (num_foods, num_tags)
        
    Returns:
        Average number of health tags per recommended food
    """
    recommended_items = top_k_items[users].cpu()  # (num_users, k)
    food_tags_batch = food_tags[recommended_items].cpu()  # (num_users, k, num_tags)
    
    # Count tags per food
    tags_per_food = food_tags_batch.sum(dim=2)  # (num_users, k)
    
    # Average per user
    avg_per_user = tags_per_food.float().mean(dim=1)  # (num_users,)
    
    # Average across users
    return avg_per_user.mean().item()


def percentage_recommended_foods(
    users: Tensor,
    top_k_items: Tensor,
    num_foods: int
) -> float:
    """
    Compute percentage of unique foods recommended.
    
    Measures diversity by checking how many different foods
    are recommended across all users.
    
    Args:
        users: User indices
        top_k_items: Top-K recommended items (num_users, k)
        num_foods: Total number of foods
        
    Returns:
        Percentage of foods recommended (0-1)
    """
    recommended_items = top_k_items[users].cpu().flatten().unique()
    percentage = len(recommended_items) / num_foods
    
    return percentage


def get_metrics(
    user_embedding: Tensor,
    item_embedding: Tensor,
    user_tags: Tensor,
    food_tags: Tensor,
    edge_index: Tensor,
    exclude_edge_indices: List[Tensor],
    k: int = 20
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        user_embedding: User embeddings (num_users, dim)
        item_embedding: Item embeddings (num_items, dim)
        user_tags: User health tags
        food_tags: Food health tags
        edge_index: Test edge index
        exclude_edge_indices: Edge indices to exclude (e.g., training edges)
        k: Number of recommendations
        
    Returns:
        Dictionary of metric values
    """
    # Compute ratings
    rating = torch.matmul(user_embedding, item_embedding.t())
    
    # Exclude edges from rating
    for exclude_edge_index in exclude_edge_indices:
        user_pos_items = get_user_positive_items(exclude_edge_index)
        
        exclude_users = []
        exclude_items = []
        
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)
        
        if exclude_users:
            rating[exclude_users, exclude_items] = float('-inf')
    
    # Get top-K recommendations
    _, top_k_items = torch.topk(rating, k=k)
    
    # Get unique users in test set
    users = edge_index[0].unique()
    test_user_pos_items = get_user_positive_items(edge_index)
    
    # Get ground truth for each user
    test_user_pos_items_list = [
        test_user_pos_items.get(user.item(), []) for user in users
    ]
    
    # Compute correctness of predictions
    predictions = []
    for user in users:
        ground_truth_items = test_user_pos_items.get(user.item(), [])
        label = [1 if item.item() in ground_truth_items else 0 
                 for item in top_k_items[user]]
        predictions.append(label)
    
    predictions = torch.tensor(predictions, dtype=torch.float32)
    
    # Compute metrics
    recall = recall_at_k(test_user_pos_items_list, predictions, k)
    precision = precision_at_k(test_user_pos_items_list, predictions, k)
    ndcg = ndcg_at_k(test_user_pos_items_list, predictions, k)
    h_score = health_score(users, top_k_items, user_tags, food_tags)
    avg_tags = average_health_tags(users, top_k_items, food_tags)
    pct_foods = percentage_recommended_foods(users, top_k_items, item_embedding.size(0))
    
    return {
        'recall': recall,
        'precision': precision,
        'ndcg': ndcg,
        'health_score': h_score,
        'avg_health_tags': avg_tags,
        'pct_foods_recommended': pct_foods
    }


def evaluate_model(
    model,
    feature_dict: Dict[str, Tensor],
    user_tags: Tensor,
    food_tags: Tensor,
    edge_index: Tensor,
    pos_edge_index: Tensor,
    neg_edge_index: Tensor,
    exclude_edge_indices: List[Tensor],
    k: int = 20,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Evaluate model on given data.
    
    Args:
        model: MOPI-HFRS model
        feature_dict: Feature dictionary
        user_tags: User health tags
        food_tags: Food health tags
        edge_index: Evaluation edge index
        pos_edge_index: Positive edges
        neg_edge_index: Negative edges
        exclude_edge_indices: Edges to exclude from evaluation
        k: Number of recommendations
        device: Device for computation
        
    Returns:
        Dictionary of metric values
    """
    model.eval()
    
    with torch.no_grad():
        # Get embeddings
        users_emb_final, _, items_emb_final, _ = model(
            feature_dict, edge_index, pos_edge_index, neg_edge_index
        )
        
        # Compute metrics
        metrics = get_metrics(
            users_emb_final,
            items_emb_final,
            user_tags,
            food_tags,
            edge_index,
            exclude_edge_indices,
            k
        )
    
    return metrics


class MetricTracker:
    """
    Track and aggregate metrics during training.
    """
    
    def __init__(self, metric_names: List[str]):
        """
        Initialize metric tracker.
        
        Args:
            metric_names: List of metric names to track
        """
        self.metric_names = metric_names
        self.history = {name: [] for name in metric_names}
        self.best_values = {name: 0.0 for name in metric_names}
    
    def update(self, metrics: Dict[str, float]):
        """
        Update tracker with new metrics.
        
        Args:
            metrics: Dictionary of metric values
        """
        for name in self.metric_names:
            if name in metrics:
                self.history[name].append(metrics[name])
                if metrics[name] > self.best_values[name]:
                    self.best_values[name] = metrics[name]
    
    def get_best(self) -> Dict[str, float]:
        """Get best values for each metric."""
        return self.best_values.copy()
    
    def get_latest(self) -> Dict[str, float]:
        """Get latest values for each metric."""
        return {name: self.history[name][-1] if self.history[name] else 0.0 
                for name in self.metric_names}
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get full history for each metric."""
        return self.history.copy()
    
    def summary(self) -> str:
        """Get summary string of current metrics."""
        latest = self.get_latest()
        return ', '.join([f'{name}: {value:.4f}' for name, value in latest.items()])

