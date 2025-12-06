"""
Loss functions for MOPI-HFRS training.

Implements three objective functions:
1. BPR Loss - User preference learning
2. Health Loss - Personalized health-aware learning
3. Diversity Loss - Nutritional diversity optimization
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional


def jaccard_similarity(user_tags: Tensor, item_tags: Tensor) -> Tensor:
    """
    Compute Jaccard similarity between user and item health tags.
    
    Based on original MOPI-HFRS implementation.
    User tags and item tags should have the same dimension for proper matching.
    
    Args:
        user_tags: User health tag vectors (batch_size, num_tags)
        item_tags: Item health tag vectors (batch_size, num_tags)
        
    Returns:
        Jaccard similarity scores (batch_size,)
    """
    # Compute intersection and union (element-wise min/max)
    intersection = torch.sum(torch.min(user_tags, item_tags), dim=1).float()
    union = torch.sum(torch.max(user_tags, item_tags), dim=1).float()
    
    # Avoid division by zero
    jaccard = intersection / (union + 1e-8)
    
    return jaccard


def bpr_loss(
    users_emb_final: Tensor,
    users_emb_0: Tensor,
    pos_items_emb_final: Tensor,
    pos_items_emb_0: Tensor,
    neg_items_emb_final: Tensor,
    neg_items_emb_0: Tensor,
    lambda_val: float = 1e-6
) -> Tensor:
    """
    Bayesian Personalized Ranking Loss.
    
    Encourages the model to rank positive items higher than negative items.
    Based on: https://arxiv.org/abs/1205.2618
    
    Args:
        users_emb_final: Final user embeddings after propagation
        users_emb_0: Initial user embeddings
        pos_items_emb_final: Final positive item embeddings
        pos_items_emb_0: Initial positive item embeddings
        neg_items_emb_final: Final negative item embeddings
        neg_items_emb_0: Initial negative item embeddings
        lambda_val: L2 regularization coefficient
        
    Returns:
        BPR loss value
    """
    # L2 regularization on initial embeddings
    reg_loss = lambda_val * (
        users_emb_0.norm(2).pow(2) +
        pos_items_emb_0.norm(2).pow(2) +
        neg_items_emb_0.norm(2).pow(2)
    )
    
    # Compute prediction scores
    pos_scores = torch.sum(users_emb_final * pos_items_emb_final, dim=-1)
    neg_scores = torch.sum(users_emb_final * neg_items_emb_final, dim=-1)
    
    # BPR loss: maximize difference between positive and negative scores
    bpr = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
    
    return bpr + reg_loss


def health_loss(
    users_emb_final: Tensor,
    pos_items_emb_final: Tensor,
    neg_items_emb_final: Tensor,
    user_tags: Tensor,
    pos_item_tags: Tensor,
    neg_item_tags: Tensor
) -> Tensor:
    """
    Health-aware loss based on tag matching.
    
    Encourages the model to recommend items that match user's health needs
    based on the Jaccard similarity of health tags.
    
    Based on original MOPI-HFRS equation (11):
    L_health = -sum log((J(t_u, t_i) - J(t_u, t_j)) * sigmoid(y_uij))
    
    Args:
        users_emb_final: User embeddings
        pos_items_emb_final: Positive item embeddings
        neg_items_emb_final: Negative item embeddings
        user_tags: User health tags
        pos_item_tags: Positive item health tags
        neg_item_tags: Negative item health tags
        
    Returns:
        Health loss value
    """
    # Compute prediction scores
    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1)
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1)
    
    # Compute Jaccard similarity for health tags
    pos_jaccard = jaccard_similarity(user_tags, pos_item_tags)
    neg_jaccard = jaccard_similarity(user_tags, neg_item_tags)
    
    # Normalize Jaccard difference to [0, 1] range
    jaccard_diff = ((pos_jaccard - neg_jaccard) + 1) / 2
    
    # Health loss: multiply jaccard weight with sigmoid of score difference
    health_loss = -torch.mean(
        torch.log(torch.mul(jaccard_diff, torch.sigmoid(pos_scores - neg_scores)) + 1e-8)
    )
    
    return health_loss


def diversity_loss(
    users_emb_final: Tensor,
    pos_items_emb_final: Tensor,
    neg_items_emb_final: Tensor,
    user_features: Tensor,
    pos_item_features: Tensor,
    neg_item_features: Tensor,
    k: int = 20
) -> Tensor:
    """
    Diversity loss for nutritional variety.
    
    Based on original MOPI-HFRS equation (12):
    L_diversity = -sum log(sigmoid((sim_pos - sim_neg) * (score_pos - score_neg)))
    
    Encourages diverse recommendations by minimizing similarity
    among top-k recommended items.
    
    Args:
        users_emb_final: User embeddings
        pos_items_emb_final: Positive item embeddings
        neg_items_emb_final: Negative item embeddings
        user_features: User feature vectors
        pos_item_features: Positive item features
        neg_item_features: Negative item features
        k: Number of top items to consider for diversity
        
    Returns:
        Diversity loss value
    """
    def get_top_k_recommendations(user_emb: Tensor, item_emb: Tensor, k: int) -> Tensor:
        """Get top-k item indices for each user."""
        scores = torch.matmul(user_emb, item_emb.T)
        _, top_k_indices = torch.topk(scores, k=k, dim=1)
        return top_k_indices
    
    def get_mean_similarity(user_features_batch: Tensor, item_features_batch: Tensor, k: int) -> Tensor:
        """Compute mean pairwise similarity among top-k items."""
        # Get top-k indices
        top_k_indices = get_top_k_recommendations(user_features_batch, item_features_batch, k)
        top_k_item_embs = item_features_batch[top_k_indices]
        
        # Calculate cosine similarities for all pairs in top-k items
        # Using unsqueeze for broadcasting: (num_users, k, 1, dim) vs (num_users, 1, k, dim)
        similarities = F.cosine_similarity(
            top_k_item_embs.unsqueeze(2),  # Shape: (num_users, k, 1, dim)
            top_k_item_embs.unsqueeze(1),  # Shape: (num_users, 1, k, dim)
            dim=3
        )
        
        # Select upper triangular part (excluding diagonal)
        upper_triangular_indices = torch.triu_indices(k, k, 1, device=similarities.device)
        selected_similarities = similarities[:, upper_triangular_indices[0], upper_triangular_indices[1]]
        
        # Return mean similarity for each user
        return selected_similarities.mean(dim=1)
    
    # Compute mean similarity for positive and negative items
    pos_similarity = get_mean_similarity(user_features, pos_item_features, k)
    neg_similarity = get_mean_similarity(user_features, neg_item_features, k)
    
    # Compute prediction scores
    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1)
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1)
    
    # Diversity loss: sigmoid of (similarity_diff * score_diff)
    loss = -torch.mean(
        torch.log(torch.sigmoid(torch.mul(pos_similarity - neg_similarity, pos_scores - neg_scores)) + 1e-8)
    )
    
    return loss


def combined_loss(
    users_emb_final: Tensor,
    users_emb_0: Tensor,
    pos_items_emb_final: Tensor,
    pos_items_emb_0: Tensor,
    neg_items_emb_final: Tensor,
    neg_items_emb_0: Tensor,
    user_features: Tensor,
    pos_item_features: Tensor,
    neg_item_features: Tensor,
    user_tags: Tensor,
    pos_item_tags: Tensor,
    neg_item_tags: Tensor,
    lambda_val: float = 1e-6,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    k: int = 20
) -> Tuple[Tensor, dict]:
    """
    Combined loss with weighted sum of all objectives.
    
    Args:
        users_emb_final: Final user embeddings
        users_emb_0: Initial user embeddings
        pos_items_emb_final: Final positive item embeddings
        pos_items_emb_0: Initial positive item embeddings
        neg_items_emb_final: Final negative item embeddings
        neg_items_emb_0: Initial negative item embeddings
        user_features: User features
        pos_item_features: Positive item features
        neg_item_features: Negative item features
        user_tags: User health tags
        pos_item_tags: Positive item health tags
        neg_item_tags: Negative item health tags
        lambda_val: L2 regularization coefficient
        alpha: Weight for BPR loss
        beta: Weight for health loss
        gamma: Weight for diversity loss
        k: Top-k for diversity computation
        
    Returns:
        Tuple of (total_loss, loss_dict)
    """
    # Compute individual losses
    l_bpr = bpr_loss(
        users_emb_final, users_emb_0,
        pos_items_emb_final, pos_items_emb_0,
        neg_items_emb_final, neg_items_emb_0,
        lambda_val
    )
    
    l_health = health_loss(
        users_emb_final, pos_items_emb_final, neg_items_emb_final,
        user_tags, pos_item_tags, neg_item_tags
    )
    
    l_diversity = diversity_loss(
        users_emb_final, pos_items_emb_final, neg_items_emb_final,
        user_features, pos_item_features, neg_item_features,
        k
    )
    
    # Weighted sum
    total_loss = alpha * l_bpr + beta * l_health + gamma * l_diversity
    
    loss_dict = {
        'bpr': l_bpr,
        'health': l_health,
        'diversity': l_diversity,
        'total': total_loss
    }
    
    return total_loss, loss_dict


def contrastive_health_loss(
    users_emb: Tensor,
    pos_items_emb: Tensor,
    neg_items_emb: Tensor,
    user_tags: Tensor,
    pos_item_tags: Tensor,
    neg_item_tags: Tensor,
    temperature: float = 0.1
) -> Tensor:
    """
    Contrastive loss variant for health-aware learning.
    
    Uses InfoNCE-style contrastive learning with health tag similarity
    as the supervision signal.
    
    Args:
        users_emb: User embeddings
        pos_items_emb: Positive item embeddings
        neg_items_emb: Negative item embeddings
        user_tags: User health tags
        pos_item_tags: Positive item health tags
        neg_item_tags: Negative item health tags
        temperature: Temperature for softmax
        
    Returns:
        Contrastive loss value
    """
    # Normalize embeddings
    users_emb = F.normalize(users_emb, p=2, dim=1)
    pos_items_emb = F.normalize(pos_items_emb, p=2, dim=1)
    neg_items_emb = F.normalize(neg_items_emb, p=2, dim=1)
    
    # Compute similarities
    pos_sim = torch.sum(users_emb * pos_items_emb, dim=1) / temperature
    neg_sim = torch.sum(users_emb * neg_items_emb, dim=1) / temperature
    
    # Compute health weights
    pos_health = jaccard_similarity(user_tags, pos_item_tags)
    neg_health = jaccard_similarity(user_tags, neg_item_tags)
    
    # Weight by health relevance
    weighted_pos = pos_sim * pos_health
    weighted_neg = neg_sim * (1 - neg_health)
    
    # Contrastive loss
    logits = torch.stack([weighted_pos, weighted_neg], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    
    loss = F.cross_entropy(logits, labels)
    
    return loss

