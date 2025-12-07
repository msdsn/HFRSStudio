"""
Structure Pooling module for graph fusion.

This module implements attention-based fusion of multiple graph structures:
- Original graph
- Feature-based similarity graph
- Health-aware signed graph

The fused graph is used for final message passing in LightGCN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional


class StructurePooling(nn.Module):
    """
    Attention-based graph structure pooling.
    
    Fuses multiple graph structures using learnable attention weights,
    producing a refined adjacency structure for downstream tasks.
    
    Args:
        num_channels: Number of input graph channels to fuse
    """
    
    def __init__(self, num_channels: int = 3):
        super().__init__()
        
        self.num_channels = num_channels
        
        # Learnable attention weights for each channel
        self.weight = nn.Parameter(torch.ones(num_channels) * 0.1)
    
    def forward(self, edge_mask_list: List[Tensor]) -> Tensor:
        """
        Fuse multiple edge masks using attention.
        
        Following original GraphChannelAttLayer implementation exactly.
        
        NOTE: The original implementation uses row normalization (p=1) which
        makes values very small for large graphs. The threshold 0.5 may need
        adjustment, or we keep edges where the fused mask is positive.
        
        Args:
            edge_mask_list: List of edge masks, each of shape (num_edges,)
            
        Returns:
            Binary fused edge mask (num_edges,)
        """
        # Stack edge masks: (num_channels, num_edges)
        edge_mask = torch.stack(edge_mask_list, dim=0)
        
        # Row normalization of all graphs generated (same as original)
        edge_mask = F.normalize(edge_mask, dim=1, p=1)
        
        # Apply softmax to weights for proper attention
        softmax_weights = F.softmax(self.weight, dim=0)
        
        # Compute weighted sum: (num_channels,) x (num_channels, num_edges) -> (num_edges,)
        weighted_edge_masks = edge_mask * softmax_weights[:, None]
        fused_edge_mask = torch.sum(weighted_edge_masks, dim=0)
        
        # Original threshold from paper
        return fused_edge_mask > 0.5
    
    def forward_soft(self, edge_mask_list: List[Tensor]) -> Tensor:
        """
        Fuse edge masks with soft (continuous) output.
        
        Args:
            edge_mask_list: List of edge masks
            
        Returns:
            Soft fused edge weights (num_edges,)
        """
        edge_mask = torch.stack(edge_mask_list, dim=0)
        edge_mask = F.normalize(edge_mask, dim=1, p=1)
        
        softmax_weights = F.softmax(self.weight, dim=0)
        weighted_edge_masks = edge_mask * softmax_weights[:, None]
        
        return torch.sum(weighted_edge_masks, dim=0)
    
    def get_attention_weights(self) -> Tensor:
        """Get the current attention weights (after softmax)."""
        return F.softmax(self.weight, dim=0)


class GatedStructurePooling(nn.Module):
    """
    Gated structure pooling with learnable gating mechanism.
    
    Uses a gating network to dynamically weight different graph structures
    based on their content, providing more flexible fusion.
    
    Args:
        num_channels: Number of input graph channels
        hidden_dim: Hidden dimension for gating network
    """
    
    def __init__(self, num_channels: int = 3, hidden_dim: int = 64):
        super().__init__()
        
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        
        # Gating network
        self.gate_network = nn.Sequential(
            nn.Linear(num_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_channels),
            nn.Softmax(dim=-1)
        )
        
        # Fallback learnable weights
        self.weight = nn.Parameter(torch.ones(num_channels) / num_channels)
    
    def forward(
        self, 
        edge_mask_list: List[Tensor],
        use_gating: bool = True
    ) -> Tensor:
        """
        Fuse edge masks using gated attention.
        
        Args:
            edge_mask_list: List of edge masks
            use_gating: Whether to use gating network
            
        Returns:
            Binary fused edge mask
        """
        edge_mask = torch.stack(edge_mask_list, dim=0)  # (C, E)
        edge_mask = F.normalize(edge_mask, dim=1, p=1)
        
        if use_gating:
            # Compute per-edge gating weights
            # Transpose to (E, C) for gating network
            edge_mask_t = edge_mask.t()  # (E, C)
            gate_weights = self.gate_network(edge_mask_t)  # (E, C)
            
            # Apply gating: element-wise multiply and sum
            fused = (edge_mask_t * gate_weights).sum(dim=1)  # (E,)
        else:
            # Use simple learnable weights
            softmax_weights = F.softmax(self.weight, dim=0)
            fused = (edge_mask * softmax_weights[:, None]).sum(dim=0)
        
        return fused > 0.5


class HierarchicalStructurePooling(nn.Module):
    """
    Hierarchical structure pooling for multi-level fusion.
    
    First fuses related structures (e.g., feature + health), then
    combines with the original structure for final output.
    
    Args:
        num_channels: Number of input channels
    """
    
    def __init__(self, num_channels: int = 3):
        super().__init__()
        
        self.num_channels = num_channels
        
        # First level: fuse feature and health graphs
        self.level1_weight = nn.Parameter(torch.ones(2) * 0.5)
        
        # Second level: fuse with original graph
        self.level2_weight = nn.Parameter(torch.ones(2) * 0.5)
    
    def forward(self, edge_mask_list: List[Tensor]) -> Tensor:
        """
        Hierarchical fusion of edge masks.
        
        Expects: [original_mask, feature_mask, health_mask]
        
        Args:
            edge_mask_list: List of 3 edge masks
            
        Returns:
            Binary fused edge mask
        """
        assert len(edge_mask_list) == 3, "Expected 3 edge masks"
        
        original_mask, feature_mask, health_mask = edge_mask_list
        
        # Normalize
        original_mask = F.normalize(original_mask.unsqueeze(0), dim=1, p=1).squeeze(0)
        feature_mask = F.normalize(feature_mask.unsqueeze(0), dim=1, p=1).squeeze(0)
        health_mask = F.normalize(health_mask.unsqueeze(0), dim=1, p=1).squeeze(0)
        
        # Level 1: Fuse feature and health
        level1_weights = F.softmax(self.level1_weight, dim=0)
        learned_mask = level1_weights[0] * feature_mask + level1_weights[1] * health_mask
        
        # Level 2: Fuse with original
        level2_weights = F.softmax(self.level2_weight, dim=0)
        fused_mask = level2_weights[0] * original_mask + level2_weights[1] * learned_mask
        
        return fused_mask > 0.5
    
    def get_attention_weights(self) -> dict:
        """Get hierarchical attention weights."""
        return {
            'level1': F.softmax(self.level1_weight, dim=0),
            'level2': F.softmax(self.level2_weight, dim=0)
        }

