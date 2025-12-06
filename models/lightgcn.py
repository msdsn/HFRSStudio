"""
LightGCN implementation for MOPI-HFRS.
Based on: https://arxiv.org/abs/2002.02126
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from typing import Tuple, Optional


class LightGCN(MessagePassing):
    """
    LightGCN Model for collaborative filtering.
    
    Simplified GCN that removes feature transformation and nonlinear activation,
    using only neighborhood aggregation for collaborative filtering.
    
    Args:
        num_users: Number of user nodes
        num_items: Number of item nodes
        embedding_dim: Dimensionality of embeddings
        num_layers: Number of message passing layers
        add_self_loops: Whether to add self loops during normalization
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        add_self_loops: bool = False
    ):
        super().__init__(aggr='add')
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.add_self_loops = add_self_loops
        
        # Initialize embeddings
        self.users_emb = nn.Embedding(num_users, embedding_dim)
        self.items_emb = nn.Embedding(num_items, embedding_dim)
        
        # Initialize with normal distribution
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)
    
    def forward(
        self,
        edge_index: SparseTensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward propagation of LightGCN.
        
        Args:
            edge_index: Sparse adjacency matrix
            
        Returns:
            Tuple of (users_emb_final, users_emb_0, items_emb_final, items_emb_0)
        """
        # Normalize adjacency matrix
        edge_index_norm = gcn_norm(edge_index, add_self_loops=self.add_self_loops)
        
        # Initial embeddings
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        embs = [emb_0]
        emb_k = emb_0
        
        # Multi-scale diffusion
        for _ in range(self.num_layers):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)
        
        # Average embeddings from all layers
        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)
        
        # Split into user and item embeddings
        users_emb_final, items_emb_final = torch.split(
            emb_final, [self.num_users, self.num_items]
        )
        
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight
    
    def message(self, x_j: Tensor) -> Tensor:
        """Message function: simply pass neighbor features."""
        return x_j
    
    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        """Fused message and aggregation using sparse matrix multiplication."""
        return matmul(adj_t, x)
    
    def get_embedding(self) -> Tuple[Tensor, Tensor]:
        """Get raw embeddings without propagation."""
        return self.users_emb.weight, self.items_emb.weight


class LightGCNWithFeatures(MessagePassing):
    """
    LightGCN variant that uses input features instead of learned embeddings.
    
    This is used when we want to initialize embeddings from external features
    (e.g., from structure learning modules).
    
    Args:
        num_users: Number of user nodes
        num_items: Number of item nodes  
        embedding_dim: Dimensionality of embeddings
        num_layers: Number of message passing layers
        add_self_loops: Whether to add self loops
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        add_self_loops: bool = False
    ):
        super().__init__(aggr='add')
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.add_self_loops = add_self_loops
    
    def forward(
        self,
        user_emb: Tensor,
        item_emb: Tensor,
        edge_index: SparseTensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward propagation with provided embeddings.
        
        Args:
            user_emb: User embeddings (num_users, embedding_dim)
            item_emb: Item embeddings (num_items, embedding_dim)
            edge_index: Sparse adjacency matrix
            
        Returns:
            Tuple of (users_emb_final, users_emb_0, items_emb_final, items_emb_0)
        """
        # Normalize adjacency matrix
        edge_index_norm = gcn_norm(edge_index, add_self_loops=self.add_self_loops)
        
        # Initial embeddings
        emb_0 = torch.cat([user_emb, item_emb])
        embs = [emb_0]
        emb_k = emb_0
        
        # Multi-scale diffusion
        for _ in range(self.num_layers):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)
        
        # Average embeddings from all layers
        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)
        
        # Split into user and item embeddings
        users_emb_final, items_emb_final = torch.split(
            emb_final, [self.num_users, self.num_items]
        )
        
        return users_emb_final, user_emb, items_emb_final, item_emb
    
    def message(self, x_j: Tensor) -> Tensor:
        return x_j
    
    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x)

