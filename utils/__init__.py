"""Utility modules for MOPI-HFRS."""

from .losses import (
    bpr_loss,
    health_loss,
    diversity_loss,
    jaccard_similarity
)
from .metrics import (
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
    health_score,
    average_health_tags,
    percentage_recommended_foods,
    get_metrics
)
from .pareto import MinNormSolver, gradient_normalizers, pareto_loss

__all__ = [
    # Losses
    'bpr_loss',
    'health_loss', 
    'diversity_loss',
    'jaccard_similarity',
    # Metrics
    'recall_at_k',
    'precision_at_k',
    'ndcg_at_k',
    'health_score',
    'average_health_tags',
    'percentage_recommended_foods',
    'get_metrics',
    # Pareto
    'MinNormSolver',
    'gradient_normalizers',
    'pareto_loss'
]

