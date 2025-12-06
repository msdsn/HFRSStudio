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
    
    Args:
        user_tags: User health tag vectors (batch_size, num_tags)
        item_tags: Item health tag vectors (batch_size, num_tags)
        
    Returns:
        Jaccard similarity scores (batch_size,)
    """
    # Compute intersection and union
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
    pos_scores = torch.sum(users_emb_final * pos_items_emb_final, dim=-1)
    neg_scores = torch.sum(users_emb_final * neg_items_emb_final, dim=-1)
    
    # Compute Jaccard similarity for health tags
    pos_jaccard = jaccard_similarity(user_tags, pos_item_tags)
    neg_jaccard = jaccard_similarity(user_tags, neg_item_tags)
    
    # Normalize Jaccard difference to [0, 1]
    jaccard_diff = ((pos_jaccard - neg_jaccard) + 1) / 2
    
    # Health loss: weight BPR by health relevance
    loss = -torch.mean(
        torch.log(
            jaccard_diff * torch.sigmoid(pos_scores - neg_scores) + 1e-8
        )
    )
    
    return loss


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
        scores = torch.matmul(user_emb, item_emb.t())
        _, top_k_indices = torch.topk(scores, k=min(k, item_emb.size(0)), dim=1)
        return top_k_indices
    
    def get_mean_similarity(user_features: Tensor, item_features: Tensor, k: int) -> Tensor:
        """Compute mean pairwise similarity among top-k items."""
        # Get top-k indices
        top_k_indices = get_top_k_recommendations(user_features, item_features, k)
        
        # Get embeddings for top-k items
        batch_size = user_features.size(0)
        actual_k = top_k_indices.size(1)
        
        # Gather top-k item embeddings
        top_k_item_embs = item_features[top_k_indices]  # (batch, k, dim)
        
        # Compute pairwise cosine similarities
        # Normalize embeddings
        top_k_normalized = F.normalize(top_k_item_embs, p=2, dim=2)
        
        # Compute similarity matrix
        similarities = torch.bmm(
            top_k_normalized,
            top_k_normalized.transpose(1, 2)
        )  # (batch, k, k)
        
        # Get upper triangular part (excluding diagonal)
        mask = torch.triu(torch.ones(actual_k, actual_k, device=similarities.device), diagonal=1)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Compute mean similarity
        num_pairs = actual_k * (actual_k - 1) / 2
        mean_sim = (similarities * mask).sum(dim=(1, 2)) / (num_pairs + 1e-8)
        
        return mean_sim
    
    # Compute mean similarity for positive and negative items
    pos_similarity = get_mean_similarity(user_features, pos_item_features, k)
    neg_similarity = get_mean_similarity(user_features, neg_item_features, k)
    
    # Compute prediction scores
    pos_scores = torch.sum(users_emb_final * pos_items_emb_final, dim=-1)
    neg_scores = torch.sum(users_emb_final * neg_items_emb_final, dim=-1)
    
    # Diversity loss: prefer lower similarity (more diverse)
    # Lower similarity in positive items is better
    sim_diff = pos_similarity - neg_similarity
    score_diff = pos_scores - neg_scores
    
    loss = -torch.mean(
        F.logsigmoid(sim_diff * score_diff)
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

