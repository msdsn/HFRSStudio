"""
Baseline models for comparison with MOPI-HFRS.

Implements the following models from the paper's comparison table:
- GCN (Graph Convolutional Network)
- GraphSAGE
- GAT (Graph Attention Network)
- LightGCN
- NGCF (Neural Graph Collaborative Filtering)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from typing import Tuple, Optional


class BaseLightGCN(MessagePassing):
    """
    LightGCN baseline model.
    
    Reference: He et al. "LightGCN: Simplifying and Powering Graph Convolution Network 
    for Recommendation" (SIGIR 2020)
    """
    
    def __init__(
        self, 
        num_users: int, 
        num_items: int, 
        embedding_dim: int = 64, 
        num_layers: int = 3,
        add_self_loops: bool = False
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.add_self_loops = add_self_loops
        
        # Embedding layers
        self.users_emb = nn.Embedding(num_users, embedding_dim)
        self.items_emb = nn.Embedding(num_items, embedding_dim)
        
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)
    
    def forward(self, edge_index: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass.
        
        Args:
            edge_index: Edge index tensor of shape (2, num_edges)
            
        Returns:
            Tuple of (users_emb_final, users_emb_0, items_emb_final, items_emb_0)
        """
        # Create sparse tensor with offset for bipartite graph
        edge_index_with_offset = torch.stack([
            edge_index[0],
            edge_index[1] + self.num_users
        ], dim=0)
        
        sparse_size = self.num_users + self.num_items
        sparse_edge_index = SparseTensor(
            row=edge_index_with_offset[0],
            col=edge_index_with_offset[1],
            sparse_sizes=(sparse_size, sparse_size)
        )
        
        # Normalize adjacency
        edge_index_norm = gcn_norm(sparse_edge_index, add_self_loops=self.add_self_loops)
        
        # Initial embeddings
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        embs = [emb_0]
        emb_k = emb_0
        
        # Multi-layer propagation
        for _ in range(self.num_layers):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)
        
        # Mean pooling across layers
        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)
        
        # Split user and item embeddings
        users_emb_final, items_emb_final = torch.split(
            emb_final, [self.num_users, self.num_items]
        )
        
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight
    
    def message(self, x_j: Tensor) -> Tensor:
        return x_j
    
    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x)


class BaseGCN(nn.Module):
    """
    GCN baseline model for recommendation.
    
    Reference: Kipf & Welling "Semi-Supervised Classification with Graph 
    Convolutional Networks" (ICLR 2017)
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layers
        self.users_emb = nn.Embedding(num_users, embedding_dim)
        self.items_emb = nn.Embedding(num_items, embedding_dim)
        
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)
        
        # GCN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(embedding_dim, embedding_dim))
    
    def forward(self, edge_index: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Create bipartite edge index
        edge_index_bipartite = torch.stack([
            edge_index[0],
            edge_index[1] + self.num_users
        ], dim=0)
        
        # Make undirected
        edge_index_undirected = torch.cat([
            edge_index_bipartite,
            edge_index_bipartite.flip(0)
        ], dim=1)
        
        # Initial embeddings
        x = torch.cat([self.users_emb.weight, self.items_emb.weight])
        emb_0 = x.clone()
        
        # GCN layers
        for conv in self.convs:
            x = conv(x, edge_index_undirected)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Split
        users_emb_final, items_emb_final = torch.split(x, [self.num_users, self.num_items])
        users_emb_0, items_emb_0 = torch.split(emb_0, [self.num_users, self.num_items])
        
        return users_emb_final, users_emb_0, items_emb_final, items_emb_0


class BaseGraphSAGE(nn.Module):
    """
    GraphSAGE baseline model for recommendation.
    
    Reference: Hamilton et al. "Inductive Representation Learning on Large Graphs" (NeurIPS 2017)
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layers
        self.users_emb = nn.Embedding(num_users, embedding_dim)
        self.items_emb = nn.Embedding(num_items, embedding_dim)
        
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)
        
        # SAGE layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(embedding_dim, embedding_dim))
    
    def forward(self, edge_index: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Create bipartite edge index
        edge_index_bipartite = torch.stack([
            edge_index[0],
            edge_index[1] + self.num_users
        ], dim=0)
        
        # Make undirected
        edge_index_undirected = torch.cat([
            edge_index_bipartite,
            edge_index_bipartite.flip(0)
        ], dim=1)
        
        # Initial embeddings
        x = torch.cat([self.users_emb.weight, self.items_emb.weight])
        emb_0 = x.clone()
        
        # SAGE layers
        for conv in self.convs:
            x = conv(x, edge_index_undirected)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Split
        users_emb_final, items_emb_final = torch.split(x, [self.num_users, self.num_items])
        users_emb_0, items_emb_0 = torch.split(emb_0, [self.num_users, self.num_items])
        
        return users_emb_final, users_emb_0, items_emb_final, items_emb_0


class BaseGAT(nn.Module):
    """
    GAT baseline model for recommendation.
    
    Reference: Veličković et al. "Graph Attention Networks" (ICLR 2018)
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layers
        self.users_emb = nn.Embedding(num_users, embedding_dim)
        self.items_emb = nn.Embedding(num_items, embedding_dim)
        
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)
        
        # GAT layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GATConv(embedding_dim, embedding_dim // num_heads, heads=num_heads, concat=True))
            else:
                self.convs.append(GATConv(embedding_dim, embedding_dim // num_heads, heads=num_heads, concat=True))
    
    def forward(self, edge_index: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Create bipartite edge index
        edge_index_bipartite = torch.stack([
            edge_index[0],
            edge_index[1] + self.num_users
        ], dim=0)
        
        # Make undirected
        edge_index_undirected = torch.cat([
            edge_index_bipartite,
            edge_index_bipartite.flip(0)
        ], dim=1)
        
        # Initial embeddings
        x = torch.cat([self.users_emb.weight, self.items_emb.weight])
        emb_0 = x.clone()
        
        # GAT layers
        for conv in self.convs:
            x = conv(x, edge_index_undirected)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Split
        users_emb_final, items_emb_final = torch.split(x, [self.num_users, self.num_items])
        users_emb_0, items_emb_0 = torch.split(emb_0, [self.num_users, self.num_items])
        
        return users_emb_final, users_emb_0, items_emb_final, items_emb_0


class NGCF(MessagePassing):
    """
    Neural Graph Collaborative Filtering baseline.
    
    Reference: Wang et al. "Neural Graph Collaborative Filtering" (SIGIR 2019)
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.0
    ):
        super().__init__(aggr='add')
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layers
        self.users_emb = nn.Embedding(num_users, embedding_dim)
        self.items_emb = nn.Embedding(num_items, embedding_dim)
        
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)
        
        # NGCF specific layers
        self.W1 = nn.ModuleList()
        self.W2 = nn.ModuleList()
        for _ in range(num_layers):
            self.W1.append(nn.Linear(embedding_dim, embedding_dim))
            self.W2.append(nn.Linear(embedding_dim, embedding_dim))
    
    def forward(self, edge_index: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Create bipartite sparse tensor
        edge_index_with_offset = torch.stack([
            edge_index[0],
            edge_index[1] + self.num_users
        ], dim=0)
        
        sparse_size = self.num_users + self.num_items
        sparse_edge_index = SparseTensor(
            row=edge_index_with_offset[0],
            col=edge_index_with_offset[1],
            sparse_sizes=(sparse_size, sparse_size)
        )
        
        # Normalize
        edge_index_norm = gcn_norm(sparse_edge_index, add_self_loops=False)
        
        # Initial embeddings
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        embs = [emb_0]
        emb_k = emb_0
        
        # NGCF layers
        for i in range(self.num_layers):
            # Propagate
            neigh_emb = self.propagate(edge_index_norm, x=emb_k)
            
            # NGCF aggregation: W1(e_i + e_N) + W2(e_i * e_N)
            sum_emb = self.W1[i](emb_k + neigh_emb)
            bi_emb = self.W2[i](emb_k * neigh_emb)
            emb_k = F.leaky_relu(sum_emb + bi_emb)
            emb_k = F.dropout(emb_k, p=self.dropout, training=self.training)
            embs.append(emb_k)
        
        # Concatenate all layers
        embs = torch.cat(embs, dim=1)
        
        # Split
        users_emb_final, items_emb_final = torch.split(embs, [self.num_users, self.num_items])
        
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight
    
    def message(self, x_j: Tensor) -> Tensor:
        return x_j
    
    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x)


def get_baseline_model(
    model_name: str,
    num_users: int,
    num_items: int,
    embedding_dim: int = 128,
    num_layers: int = 3,
    **kwargs
) -> nn.Module:
    """
    Factory function to get baseline model by name.
    
    Args:
        model_name: One of 'gcn', 'graphsage', 'gat', 'lightgcn', 'ngcf'
        num_users: Number of users
        num_items: Number of items
        embedding_dim: Embedding dimension
        num_layers: Number of layers
        
    Returns:
        Baseline model instance
    """
    model_name = model_name.lower()
    
    if model_name == 'gcn':
        return BaseGCN(num_users, num_items, embedding_dim, num_layers)
    elif model_name == 'graphsage':
        return BaseGraphSAGE(num_users, num_items, embedding_dim, num_layers)
    elif model_name == 'gat':
        return BaseGAT(num_users, num_items, embedding_dim, num_layers)
    elif model_name == 'lightgcn':
        return BaseLightGCN(num_users, num_items, embedding_dim, num_layers)
    elif model_name == 'ngcf':
        return NGCF(num_users, num_items, embedding_dim, num_layers)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: gcn, graphsage, gat, lightgcn, ngcf")

