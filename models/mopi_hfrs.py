"""
MOPI-HFRS: Multi-Objective Personalized Interpretable Health-aware Food Recommendation System

Main model implementation combining:
- Feature-based Graph Structure Learning
- Health-aware Signed Graph Learning
- Structure Pooling
- LightGCN for final message passing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.nn import Linear
from typing import Dict, Tuple, Optional

from .lightgcn import LightGCN
from .signed_gcn import SignedGCN
from .graph_generator import GraphGenerator
from .structure_pooling import StructurePooling


class MOPI_HFRS(nn.Module):
    """
    Multi-Objective Personalized Interpretable Health-aware Food Recommendation System.
    
    This model combines three graph structure learning approaches:
    1. Feature-based structure learning using metric learning
    2. Health-aware structure learning using signed graph convolution
    3. Original graph structure
    
    These are fused using attention-based pooling and processed through LightGCN.
    
    Args:
        num_users: Number of user nodes
        num_foods: Number of food nodes
        user_feature_dim: Dimension of user features
        food_feature_dim: Dimension of food features
        embedding_dim: Hidden embedding dimension
        num_layers: Number of GCN layers
        num_heads: Number of attention heads for similarity
        feature_threshold: Threshold for feature similarity
    """
    
    def __init__(
        self,
        num_users: int,
        num_foods: int,
        user_feature_dim: int,
        food_feature_dim: int,
        embedding_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        feature_threshold: float = 0.3
    ):
        super().__init__()
        
        self.num_users = num_users
        self.num_foods = num_foods
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.feature_threshold = feature_threshold
        
        # Feature projection layers (heterogeneous to common space)
        self.user_proj = Linear(user_feature_dim, embedding_dim)
        self.food_proj = Linear(food_feature_dim, embedding_dim)
        
        # Feature-based graph generator
        self.feature_graph_generator = GraphGenerator(
            feature_dim=embedding_dim,
            num_heads=num_heads,
            similarity_threshold=feature_threshold
        )
        
        # Health-aware signed graph learning
        self.signed_layer = SignedGCN(
            num_users=num_users,
            num_foods=num_foods,
            hidden_channels=embedding_dim,
            num_layers=num_layers
        )
        
        # Structure pooling (fuse 3 graphs)
        self.fusion = StructurePooling(num_channels=3)
        
        # LightGCN for final message passing
        self.lightgcn = LightGCN(
            num_users=num_users,
            num_items=num_foods,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            add_self_loops=False
        )
    
    def forward(
        self,
        feature_dict: Dict[str, Tensor],
        edge_index: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass of MOPI-HFRS.
        
        Args:
            feature_dict: Dictionary with 'user' and 'food' features
            edge_index: Full edge index (2, num_edges)
            pos_edge_index: Positive (healthy) edge index
            neg_edge_index: Negative (unhealthy) edge index
            
        Returns:
            Tuple of (users_emb_final, users_emb_0, items_emb_final, items_emb_0)
        """
        # Project features to common embedding space
        user_emb = F.relu(self.user_proj(feature_dict['user']))
        food_emb = F.relu(self.food_proj(feature_dict['food']))
        
        # 1. Generate feature-based similarity mask
        mask_feature = self.feature_graph_generator(
            user_emb, food_emb, edge_index
        )
        
        # 2. Original graph mask (all ones)
        mask_ori = torch.ones_like(mask_feature)
        
        # 3. Generate health-aware signed graph mask
        z = self.signed_layer(pos_edge_index, neg_edge_index)
        mask_semantic = self.signed_layer.discriminate(z, edge_index).float()
        
        # Fuse the three graph structures
        edge_mask = self.fusion([mask_ori, mask_feature, mask_semantic])
        
        # Create new edge index with fused mask
        edge_index_new = edge_index[:, edge_mask]
        
        # Convert to sparse tensor for LightGCN
        # CRITICAL: Add offset to food indices for LightGCN
        # LightGCN concatenates [user_emb, item_emb], so food indices need offset
        sparse_size = self.num_users + self.num_foods
        
        # Add num_users offset to food indices (col)
        edge_index_with_offset = torch.stack([
            edge_index_new[0],  # user indices stay the same
            edge_index_new[1] + self.num_users  # food indices need offset
        ], dim=0)
        
        sparse_edge_index = SparseTensor(
            row=edge_index_with_offset[0],
            col=edge_index_with_offset[1],
            sparse_sizes=(sparse_size, sparse_size)
        )
        
        # Apply LightGCN
        return self.lightgcn(sparse_edge_index)
    
    def get_embeddings(
        self,
        feature_dict: Dict[str, Tensor],
        edge_index: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Get final user and item embeddings.
        
        Args:
            feature_dict: Feature dictionary
            edge_index: Edge index
            pos_edge_index: Positive edges
            neg_edge_index: Negative edges
            
        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        users_emb_final, _, items_emb_final, _ = self.forward(
            feature_dict, edge_index, pos_edge_index, neg_edge_index
        )
        return users_emb_final, items_emb_final
    
    def predict(
        self,
        user_indices: Tensor,
        item_indices: Tensor,
        users_emb: Tensor,
        items_emb: Tensor
    ) -> Tensor:
        """
        Predict scores for user-item pairs.
        
        Args:
            user_indices: User indices
            item_indices: Item indices
            users_emb: User embeddings
            items_emb: Item embeddings
            
        Returns:
            Prediction scores
        """
        user_emb = users_emb[user_indices]
        item_emb = items_emb[item_indices]
        return torch.sum(user_emb * item_emb, dim=-1)
    
    def recommend(
        self,
        user_idx: int,
        users_emb: Tensor,
        items_emb: Tensor,
        k: int = 20,
        exclude_items: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate top-k recommendations for a user.
        
        Args:
            user_idx: User index
            users_emb: User embeddings
            items_emb: Item embeddings
            k: Number of recommendations
            exclude_items: Items to exclude (e.g., already consumed)
            
        Returns:
            Tuple of (top_k_indices, top_k_scores)
        """
        user_emb = users_emb[user_idx]
        
        # Compute scores for all items
        scores = torch.matmul(user_emb, items_emb.t())
        
        # Exclude specified items
        if exclude_items is not None:
            scores[exclude_items] = float('-inf')
        
        # Get top-k
        top_k_scores, top_k_indices = torch.topk(scores, k)
        
        return top_k_indices, top_k_scores


class MOPI_HFRS_Light(nn.Module):
    """
    Lightweight version of MOPI-HFRS without signed graph learning.
    
    Uses only feature-based structure learning and original graph,
    suitable for faster training or when health labels are not available.
    
    Args:
        num_users: Number of users
        num_foods: Number of foods
        user_feature_dim: User feature dimension
        food_feature_dim: Food feature dimension
        embedding_dim: Embedding dimension
        num_layers: Number of GCN layers
        num_heads: Number of attention heads
        feature_threshold: Similarity threshold
    """
    
    def __init__(
        self,
        num_users: int,
        num_foods: int,
        user_feature_dim: int,
        food_feature_dim: int,
        embedding_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        feature_threshold: float = 0.3
    ):
        super().__init__()
        
        self.num_users = num_users
        self.num_foods = num_foods
        self.embedding_dim = embedding_dim
        
        # Feature projections
        self.user_proj = Linear(user_feature_dim, embedding_dim)
        self.food_proj = Linear(food_feature_dim, embedding_dim)
        
        # Feature-based graph generator
        self.feature_graph_generator = GraphGenerator(
            feature_dim=embedding_dim,
            num_heads=num_heads,
            similarity_threshold=feature_threshold
        )
        
        # Structure pooling (2 channels: original + feature)
        self.fusion = StructurePooling(num_channels=2)
        
        # LightGCN
        self.lightgcn = LightGCN(
            num_users=num_users,
            num_items=num_foods,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            add_self_loops=False
        )
    
    def forward(
        self,
        feature_dict: Dict[str, Tensor],
        edge_index: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass without health-aware learning."""
        # Project features
        user_emb = F.relu(self.user_proj(feature_dict['user']))
        food_emb = F.relu(self.food_proj(feature_dict['food']))
        
        # Generate feature-based mask
        mask_feature = self.feature_graph_generator(
            user_emb, food_emb, edge_index
        )
        
        # Original mask
        mask_ori = torch.ones_like(mask_feature)
        
        # Fuse
        edge_mask = self.fusion([mask_ori, mask_feature])
        
        # Create sparse edge index
        edge_index_new = edge_index[:, edge_mask]
        sparse_size = self.num_users + self.num_foods
        
        sparse_edge_index = SparseTensor(
            row=edge_index_new[0],
            col=edge_index_new[1],
            sparse_sizes=(sparse_size, sparse_size)
        )
        
        return self.lightgcn(sparse_edge_index)


def create_model(
    num_users: int,
    num_foods: int,
    user_feature_dim: int,
    food_feature_dim: int,
    config: Optional[dict] = None
) -> MOPI_HFRS:
    """
    Factory function to create MOPI-HFRS model.
    
    Args:
        num_users: Number of users
        num_foods: Number of foods
        user_feature_dim: User feature dimension
        food_feature_dim: Food feature dimension
        config: Optional configuration dictionary
        
    Returns:
        MOPI_HFRS model instance
    """
    default_config = {
        'embedding_dim': 128,
        'num_layers': 3,
        'num_heads': 4,
        'feature_threshold': 0.3
    }
    
    if config:
        default_config.update(config)
    
    return MOPI_HFRS(
        num_users=num_users,
        num_foods=num_foods,
        user_feature_dim=user_feature_dim,
        food_feature_dim=food_feature_dim,
        **default_config
    )

