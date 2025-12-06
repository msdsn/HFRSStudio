"""
Pareto optimization for multi-objective learning.

Implements the Multiple Gradient Descent Algorithm (MGDA) using
Frank-Wolfe algorithm to find Pareto optimal solutions.

Based on: https://arxiv.org/abs/1810.04650
"""

import numpy as np
import torch
from torch import Tensor
from typing import List, Dict, Tuple, Optional
import torch.nn as nn

from .losses import bpr_loss, health_loss, diversity_loss


class MinNormSolver:
    """
    Solver for finding the minimum norm element in the convex hull of gradients.
    
    This is used to find Pareto optimal gradient directions when
    optimizing multiple objectives simultaneously.
    """
    
    MAX_ITER = 250
    STOP_CRIT = 1e-5
    
    @staticmethod
    def _min_norm_element_from2(v1v1: float, v1v2: float, v2v2: float) -> Tuple[float, float]:
        """
        Analytical solution for min_{c} |c*x_1 + (1-c)*x_2|_2^2
        
        Args:
            v1v1: <x1, x1>
            v1v2: <x1, x2>
            v2v2: <x2, x2>
            
        Returns:
            Tuple of (gamma, cost)
        """
        if v1v2 >= v1v1:
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        
        if v1v2 >= v2v2:
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        
        return gamma, cost
    
    @staticmethod
    def _min_norm_2d(vecs: List[List[Tensor]], dps: Dict) -> Tuple[List, Dict]:
        """
        Find the minimum norm solution as combination of two points.
        
        Args:
            vecs: List of gradient vectors
            dps: Dictionary of dot products
            
        Returns:
            Tuple of (solution, updated_dps)
        """
        dmin = 1e8
        sol = None
        
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                # Compute dot products if not cached
                if (i, j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, j)] += torch.mul(
                            vecs[i][k], vecs[j][k]
                        ).sum().data.cpu().numpy()
                    dps[(j, i)] = dps[(i, j)]
                
                if (i, i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, i)] += torch.mul(
                            vecs[i][k], vecs[i][k]
                        ).sum().data.cpu().numpy()
                
                if (j, j) not in dps:
                    dps[(j, j)] = 0.0
                    for k in range(len(vecs[j])):
                        dps[(j, j)] += torch.mul(
                            vecs[j][k], vecs[j][k]
                        ).sum().data.cpu().numpy()
                
                c, d = MinNormSolver._min_norm_element_from2(
                    dps[(i, i)], dps[(i, j)], dps[(j, j)]
                )
                
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        
        return sol, dps
    
    @staticmethod
    def _projection2simplex(y: np.ndarray) -> np.ndarray:
        """
        Project onto probability simplex.
        
        Args:
            y: Input vector
            
        Returns:
            Projected vector on simplex
        """
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0) / m
        
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        
        return np.maximum(y - tmax_f, np.zeros(y.shape))
    
    @staticmethod
    def _next_point(cur_val: np.ndarray, grad: np.ndarray, n: int) -> np.ndarray:
        """
        Compute next point in Frank-Wolfe algorithm.
        
        Args:
            cur_val: Current solution
            grad: Gradient direction
            n: Number of objectives
            
        Returns:
            Next point
        """
        proj_grad = grad - (np.sum(grad) / n)
        
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / proj_grad[proj_grad > 0]
        
        t = 1
        if len(tm1[tm1 > 1e-7]) > 0:
            t = np.min(tm1[tm1 > 1e-7])
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, np.min(tm2[tm2 > 1e-7]))
        
        next_point = proj_grad * t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        
        return next_point
    
    @staticmethod
    def find_min_norm_element(vecs: List[List[Tensor]]) -> Tuple[np.ndarray, float]:
        """
        Find minimum norm element in convex hull of gradients.
        
        Uses projected gradient descent.
        
        Args:
            vecs: List of gradient vectors for each objective
            
        Returns:
            Tuple of (solution_weights, min_norm)
        """
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)
        
        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]
        
        if n < 3:
            return sol_vec, init_sol[2]
        
        iter_count = 0
        grad_mat = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]
        
        while iter_count < MinNormSolver.MAX_ITER:
            grad_dir = -1.0 * np.dot(grad_mat, sol_vec)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
            
            # Re-compute inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i] * sol_vec[j] * dps[(i, j)]
                    v1v2 += sol_vec[i] * new_point[j] * dps[(i, j)]
                    v2v2 += new_point[i] * new_point[j] * dps[(i, j)]
            
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            
            sol_vec = new_sol_vec
            iter_count += 1
        
        return sol_vec, nd
    
    @staticmethod
    def find_min_norm_element_FW(vecs: List[List[Tensor]]) -> Tuple[np.ndarray, float]:
        """
        Find minimum norm element using Frank-Wolfe algorithm.
        
        More efficient variant that uses Frank-Wolfe updates.
        
        Args:
            vecs: List of gradient vectors for each objective
            
        Returns:
            Tuple of (solution_weights, min_norm)
        """
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)
        
        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]
        
        if n < 3:
            return sol_vec, init_sol[2]
        
        iter_count = 0
        grad_mat = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]
        
        nd = init_sol[2]
        
        while iter_count < MinNormSolver.MAX_ITER:
            iter_count += 1
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))
            
            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]
            
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec
            new_sol_vec[t_iter] += 1 - nc
            
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            
            sol_vec = new_sol_vec
        
        return sol_vec, nd


def gradient_normalizers(
    grads: Dict[str, List[Tensor]],
    losses: Dict[str, Tensor],
    normalization_type: str = 'l2'
) -> Dict[str, Tensor]:
    """
    Compute gradient normalizers for each task.
    
    Args:
        grads: Dictionary of gradients for each task
        losses: Dictionary of loss values for each task
        normalization_type: Type of normalization ('l2', 'loss', 'loss+', 'none')
        
    Returns:
        Dictionary of normalizers for each task
    """
    gn = {}
    
    if normalization_type == 'l2':
        for t in grads:
            gn[t] = torch.tensor(
                np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))
            )
    
    elif normalization_type == 'loss':
        for t in grads:
            gn[t] = losses[t]
    
    elif normalization_type == 'loss+':
        for t in grads:
            gn[t] = losses[t] * np.sqrt(
                np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]])
            )
    
    elif normalization_type == 'none':
        for t in grads:
            gn[t] = torch.tensor(1.0)
    
    else:
        raise ValueError(f'Invalid normalization type: {normalization_type}')
    
    return gn


def pareto_loss(
    model: nn.Module,
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
    normalization_type: str = 'l2'
) -> Tuple[Tensor, Dict[str, Tensor], Tensor]:
    """
    Compute Pareto-optimal loss for multi-objective optimization.
    
    Uses MGDA to find optimal weighting of BPR, health, and diversity losses.
    
    Args:
        model: The model being trained
        users_emb_final: Final user embeddings
        users_emb_0: Initial user embeddings
        pos_items_emb_final: Final positive item embeddings
        pos_items_emb_0: Initial positive item embeddings
        neg_items_emb_final: Final negative item embeddings
        neg_items_emb_0: Initial negative item embeddings
        user_features: User features for diversity
        pos_item_features: Positive item features
        neg_item_features: Negative item features
        user_tags: User health tags
        pos_item_tags: Positive item health tags
        neg_item_tags: Negative item health tags
        lambda_val: L2 regularization coefficient
        normalization_type: Gradient normalization type
        
    Returns:
        Tuple of (pareto_loss, loss_dict, actual_total_loss)
    """
    loss_data = {}
    grads = {}
    tasks = ['bpr', 'health', 'diversity']
    
    # Compute gradients for each task
    for task in tasks:
        if task == 'bpr':
            loss = bpr_loss(
                users_emb_final, users_emb_0,
                pos_items_emb_final, pos_items_emb_0,
                neg_items_emb_final, neg_items_emb_0,
                lambda_val
            )
        elif task == 'health':
            loss = health_loss(
                users_emb_final, pos_items_emb_final, neg_items_emb_final,
                user_tags, pos_item_tags, neg_item_tags
            )
        elif task == 'diversity':
            loss = diversity_loss(
                users_emb_final, pos_items_emb_final, neg_items_emb_final,
                user_features, pos_item_features, neg_item_features
            )
        else:
            raise ValueError(f'Unknown task: {task}')
        
        loss_data[task] = loss
        grads[task] = []
        
        # Compute gradients
        loss.backward(retain_graph=True)
        
        for param in model.parameters():
            if param.grad is not None:
                grads[task].append(param.grad.data.detach().clone())
        
        model.zero_grad()
    
    # Normalize gradients
    gn = gradient_normalizers(grads, loss_data, normalization_type)
    
    for task in loss_data:
        for gr_i in range(len(grads[task])):
            grads[task][gr_i] = grads[task][gr_i] / gn[task].to(grads[task][gr_i].device)
    
    # Find Pareto optimal weights
    sol, _ = MinNormSolver.find_min_norm_element_FW([grads[task] for task in tasks])
    sol = {k: sol[i] for i, k in enumerate(tasks)}
    
    model.zero_grad()
    
    # Compute weighted loss
    loss = 0
    actual_loss = 0
    
    for task, l in loss_data.items():
        loss += float(sol[task]) * l
        actual_loss += l
    
    return loss, loss_data, actual_loss


class ParetoMTL:
    """
    Pareto Multi-Task Learning optimizer wrapper.
    
    Provides a convenient interface for Pareto optimization
    during training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tasks: List[str] = ['bpr', 'health', 'diversity'],
        normalization_type: str = 'l2'
    ):
        """
        Initialize Pareto MTL optimizer.
        
        Args:
            model: The model being trained
            tasks: List of task names
            normalization_type: Gradient normalization type
        """
        self.model = model
        self.tasks = tasks
        self.normalization_type = normalization_type
        self.history = {'weights': [], 'losses': []}
    
    def step(
        self,
        loss_dict: Dict[str, Tensor]
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute Pareto optimal loss and weights.
        
        Args:
            loss_dict: Dictionary of losses for each task
            
        Returns:
            Tuple of (pareto_loss, task_weights)
        """
        grads = {}
        
        # Compute gradients for each task
        for task in self.tasks:
            loss = loss_dict[task]
            grads[task] = []
            
            loss.backward(retain_graph=True)
            
            for param in self.model.parameters():
                if param.grad is not None:
                    grads[task].append(param.grad.data.detach().clone())
            
            self.model.zero_grad()
        
        # Normalize gradients
        gn = gradient_normalizers(grads, loss_dict, self.normalization_type)
        
        for task in self.tasks:
            for gr_i in range(len(grads[task])):
                grads[task][gr_i] = grads[task][gr_i] / gn[task].to(grads[task][gr_i].device)
        
        # Find Pareto optimal weights
        sol, _ = MinNormSolver.find_min_norm_element_FW([grads[task] for task in self.tasks])
        weights = {k: sol[i] for i, k in enumerate(self.tasks)}
        
        # Compute weighted loss
        self.model.zero_grad()
        total_loss = sum(float(weights[task]) * loss_dict[task] for task in self.tasks)
        
        # Store history
        self.history['weights'].append(weights)
        self.history['losses'].append({k: v.item() for k, v in loss_dict.items()})
        
        return total_loss, weights

