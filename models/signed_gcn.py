"""
SignedGCN implementation for health-aware edge learning.
Based on: https://arxiv.org/abs/1808.06354

This module implements signed graph convolution for distinguishing
healthy (positive) and unhealthy (negative) user-food relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import SignedConv
from typing import Tuple, Optional


class SignedGCN(nn.Module):
    """
    Signed Graph Convolutional Network for health-aware edge learning.
    
    Uses balance theory to aggregate positive and negative edges differently,
    learning separate embeddings for healthy and unhealthy relationships.
    
    Args:
        num_users: Number of user nodes
        num_foods: Number of food nodes
        hidden_channels: Hidden layer dimensionality
        num_layers: Number of SignedConv layers
    """
    
    def __init__(
        self,
        num_users: int,
        num_foods: int,
        hidden_channels: int,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.num_users = num_users
        self.num_foods = num_foods
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        # Node embeddings
        self.users_emb = nn.Embedding(num_users, hidden_channels)
        self.items_emb = nn.Embedding(num_foods, hidden_channels)
        
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)
        
        # First SignedConv layer (first_aggr=True)
        self.conv1 = SignedConv(
            hidden_channels, 
            hidden_channels // 2,
            first_aggr=True
        )
        
        # Subsequent SignedConv layers (first_aggr=False)
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(
                SignedConv(
                    hidden_channels // 2,
                    hidden_channels // 2,
                    first_aggr=False
                )
            )
        
        # Linear layer for edge classification
        self.lin = nn.Linear(2 * hidden_channels, 3)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()
    
    def forward(
        self,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor
    ) -> Tensor:
        """
        Compute node embeddings based on positive and negative edges.
        
        NOTE: Following original SGSL implementation which does NOT add offset
        to food indices. This means food index 0 maps to x[0] (user embedding),
        not x[num_users] (food embedding). This is technically incorrect but
        matches the original paper's implementation.
        
        Args:
            pos_edge_index: Positive (healthy) edge indices (bipartite format)
            neg_edge_index: Negative (unhealthy) edge indices (bipartite format)
            
        Returns:
            Node embeddings tensor
        """
        # Concatenate user and item embeddings
        x = torch.cat([self.users_emb.weight, self.items_emb.weight])
        
        # First layer - using bipartite edge indices directly (matching original)
        z = F.relu(self.conv1(x, pos_edge_index, neg_edge_index))
        
        # Subsequent layers
        for conv in self.convs:
            z = F.relu(conv(z, pos_edge_index, neg_edge_index))
        
        return z
    
    def discriminate(self, z: Tensor, edge_index: Tensor) -> Tensor:
        """
        Classify edges as positive, negative, or non-existent.
        
        Args:
            z: Node embeddings
            edge_index: Edge indices to classify
            
        Returns:
            Edge classification scores (-1: negative, 0: non-existent, 1: positive)
        """
        # Concatenate source and target embeddings
        value = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        value = self.lin(value)
        
        # Get class predictions
        log_softmax_output = torch.log_softmax(value, dim=1)
        class_indices = torch.argmax(log_softmax_output, dim=1)
        
        # Map class indices: 0 -> -1, 1 -> 0, 2 -> 1
        mapping = torch.tensor([-1, 0, 1], device=value.device)
        mapped_output = mapping[class_indices]
        
        return mapped_output
    
    def get_edge_mask(self, z: Tensor, edge_index: Tensor) -> Tensor:
        """
        Get binary mask indicating positive edges.
        
        Args:
            z: Node embeddings
            edge_index: Edge indices
            
        Returns:
            Binary mask (1 for positive edges, 0 otherwise)
        """
        edge_classes = self.discriminate(z, edge_index)
        return (edge_classes == 1).float()
    
    def loss(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor
    ) -> Tensor:
        """
        Compute contrastive loss for signed graph learning.
        
        Args:
            z: Node embeddings
            pos_edge_index: Positive edge indices
            neg_edge_index: Negative edge indices
            
        Returns:
            Loss value
        """
        # Positive edge loss
        pos_value = torch.cat([z[pos_edge_index[0]], z[pos_edge_index[1]]], dim=1)
        pos_out = self.lin(pos_value)
        pos_loss = F.cross_entropy(
            pos_out, 
            torch.full((pos_out.size(0),), 2, dtype=torch.long, device=pos_out.device)
        )
        
        # Negative edge loss
        neg_value = torch.cat([z[neg_edge_index[0]], z[neg_edge_index[1]]], dim=1)
        neg_out = self.lin(neg_value)
        neg_loss = F.cross_entropy(
            neg_out,
            torch.full((neg_out.size(0),), 0, dtype=torch.long, device=neg_out.device)
        )
        
        return pos_loss + neg_loss


class SignedGCNWithFeatures(nn.Module):
    """
    SignedGCN variant that uses input features instead of learned embeddings.
    
    Args:
        in_channels: Input feature dimensionality
        hidden_channels: Hidden layer dimensionality
        num_layers: Number of SignedConv layers
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # First SignedConv layer
        self.conv1 = SignedConv(
            hidden_channels,
            hidden_channels // 2,
            first_aggr=True
        )
        
        # Subsequent layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(
                SignedConv(
                    hidden_channels // 2,
                    hidden_channels // 2,
                    first_aggr=False
                )
            )
        
        # Edge classifier
        self.lin = nn.Linear(2 * hidden_channels, 3)
    
    def forward(
        self,
        x: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor
    ) -> Tensor:
        """
        Forward pass with input features.
        
        Args:
            x: Input node features
            pos_edge_index: Positive edge indices
            neg_edge_index: Negative edge indices
            
        Returns:
            Node embeddings
        """
        # Project input features
        x = F.relu(self.input_proj(x))
        
        # SignedConv layers
        z = F.relu(self.conv1(x, pos_edge_index, neg_edge_index))
        
        for conv in self.convs:
            z = F.relu(conv(z, pos_edge_index, neg_edge_index))
        
        return z
    
    def discriminate(self, z: Tensor, edge_index: Tensor) -> Tensor:
        """Classify edges."""
        value = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        value = self.lin(value)
        
        log_softmax_output = torch.log_softmax(value, dim=1)
        class_indices = torch.argmax(log_softmax_output, dim=1)
        
        mapping = torch.tensor([-1, 0, 1], device=value.device)
        return mapping[class_indices]

