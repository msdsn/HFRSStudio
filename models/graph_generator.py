"""
Graph Generator module for feature-based structure learning.

This module implements metric learning to generate a similarity graph
based on user and food features, enabling the model to discover
latent relationships based on feature similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class MetricCalculator(nn.Module):
    """
    Applies a learnable transformation for metric learning.
    
    This layer learns importance weights for each feature dimension,
    enabling the model to focus on the most relevant features for
    computing similarity.
    
    Args:
        feature_dim: Dimensionality of input features
    """
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, feature_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply learnable transformation.
        
        Args:
            x: Input features (batch_size, feature_dim)
            
        Returns:
            Weighted features (batch_size, feature_dim)
        """
        return x * self.weight


class GraphGenerator(nn.Module):
    """
    Generates a similarity graph based on feature similarity.
    
    Uses multi-head weighted cosine similarity to compute edge weights
    between users and foods, creating a refined graph structure that
    captures feature-based relationships.
    
    Args:
        feature_dim: Dimensionality of input features
        num_heads: Number of attention heads for multi-head similarity
        similarity_threshold: Minimum similarity for edge creation
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 2,
        similarity_threshold: float = 0.1
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.similarity_threshold = similarity_threshold
        
        # Multi-head metric calculators
        self.metric_layers = nn.ModuleList([
            MetricCalculator(feature_dim) for _ in range(num_heads)
        ])
    
    def forward(
        self,
        left_features: Tensor,
        right_features: Tensor,
        edge_index: Tensor
    ) -> Tensor:
        """
        Compute similarity scores for edges.
        
        Args:
            left_features: Source node features (num_left_nodes, feature_dim)
            right_features: Target node features (num_right_nodes, feature_dim)
            edge_index: Edge indices (2, num_edges)
            
        Returns:
            Similarity scores for each edge (num_edges,)
        """
        # Initialize similarity matrix
        similarity_matrix = torch.zeros(
            edge_index.size(1), 
            device=edge_index.device
        )
        
        # Compute multi-head similarity
        for metric_layer in self.metric_layers:
            # Get weighted features for source and target nodes
            weighted_left = metric_layer(left_features[edge_index[0]])
            weighted_right = metric_layer(right_features[edge_index[1]])
            
            # Compute cosine similarity
            similarity_matrix += F.cosine_similarity(
                weighted_left, weighted_right, dim=1
            )
        
        # Average across heads
        similarity_matrix /= self.num_heads
        
        # Apply threshold
        similarity_matrix = torch.where(
            similarity_matrix < self.similarity_threshold,
            torch.zeros_like(similarity_matrix),
            similarity_matrix
        )
        
        return similarity_matrix
    
    def get_edge_mask(
        self,
        left_features: Tensor,
        right_features: Tensor,
        edge_index: Tensor
    ) -> Tensor:
        """
        Get binary mask for edges above similarity threshold.
        
        Args:
            left_features: Source node features
            right_features: Target node features
            edge_index: Edge indices
            
        Returns:
            Binary mask (1 for similar edges, 0 otherwise)
        """
        similarity = self.forward(left_features, right_features, edge_index)
        return (similarity > 0).float()
    
    def compute_full_similarity(
        self,
        left_features: Tensor,
        right_features: Tensor
    ) -> Tensor:
        """
        Compute full similarity matrix between all pairs.
        
        Warning: This can be memory-intensive for large graphs.
        
        Args:
            left_features: Source node features (num_left, feature_dim)
            right_features: Target node features (num_right, feature_dim)
            
        Returns:
            Similarity matrix (num_left, num_right)
        """
        num_left = left_features.size(0)
        num_right = right_features.size(0)
        
        similarity_matrix = torch.zeros(
            num_left, num_right, 
            device=left_features.device
        )
        
        for metric_layer in self.metric_layers:
            weighted_left = metric_layer(left_features)  # (num_left, dim)
            weighted_right = metric_layer(right_features)  # (num_right, dim)
            
            # Normalize for cosine similarity
            weighted_left = F.normalize(weighted_left, p=2, dim=1)
            weighted_right = F.normalize(weighted_right, p=2, dim=1)
            
            # Compute similarity matrix
            similarity_matrix += torch.mm(weighted_left, weighted_right.t())
        
        similarity_matrix /= self.num_heads
        
        return similarity_matrix


class AdaptiveGraphGenerator(nn.Module):
    """
    Adaptive graph generator with learnable threshold.
    
    This variant learns the similarity threshold during training,
    allowing the model to adaptively determine edge sparsity.
    
    Args:
        feature_dim: Dimensionality of input features
        num_heads: Number of attention heads
        init_threshold: Initial similarity threshold
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 2,
        init_threshold: float = 0.1
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Learnable threshold
        self.threshold = nn.Parameter(torch.tensor(init_threshold))
        
        # Multi-head metric calculators
        self.metric_layers = nn.ModuleList([
            MetricCalculator(feature_dim) for _ in range(num_heads)
        ])
        
        # Optional MLP for non-linear similarity
        self.similarity_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        left_features: Tensor,
        right_features: Tensor,
        edge_index: Tensor,
        use_mlp: bool = False
    ) -> Tensor:
        """
        Compute adaptive similarity scores.
        
        Args:
            left_features: Source node features
            right_features: Target node features
            edge_index: Edge indices
            use_mlp: Whether to use MLP for similarity computation
            
        Returns:
            Similarity scores
        """
        if use_mlp:
            # Concatenate source and target features
            left = left_features[edge_index[0]]
            right = right_features[edge_index[1]]
            combined = torch.cat([left, right], dim=1)
            similarity = self.similarity_mlp(combined).squeeze(-1)
        else:
            # Use multi-head cosine similarity
            similarity = torch.zeros(
                edge_index.size(1),
                device=edge_index.device
            )
            
            for metric_layer in self.metric_layers:
                weighted_left = metric_layer(left_features[edge_index[0]])
                weighted_right = metric_layer(right_features[edge_index[1]])
                similarity += F.cosine_similarity(
                    weighted_left, weighted_right, dim=1
                )
            
            similarity /= self.num_heads
        
        # Apply learnable threshold with sigmoid for differentiability
        threshold = torch.sigmoid(self.threshold)
        similarity = torch.where(
            similarity < threshold,
            torch.zeros_like(similarity),
            similarity
        )
        
        return similarity

