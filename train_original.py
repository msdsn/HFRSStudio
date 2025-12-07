"""
Original MOPI-HFRS training script - exact reproduction from the paper's code.
This is a direct copy of MOPI-HFRS-main/code/main.py with minimal changes
to make it runnable from the project root.
"""

import sys
import os

# Add the original code directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MOPI-HFRS-main', 'code'))

import argparse
import torch
import torch.optim as optim
import torch_geometric
import numpy as np
import random
from sklearn.model_selection import train_test_split
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import structured_negative_sampling
from tqdm import tqdm
from torch.nn.functional import cosine_similarity
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, SignedConv, Linear
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch import Tensor


# ============================================================================
# min_norm_solvers.py - EXACT COPY
# ============================================================================

class MinNormSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        if v1v2 >= v1v1:
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        gamma = -1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2) )
        cost = v2v2 + gamma*(v1v2 - v2v2)
        return gamma, cost

    def _min_norm_2d(vecs, dps):
        dmin = 1e8
        for i in range(len(vecs)):
            for j in range(i+1,len(vecs)):
                if (i,j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i,j)] += torch.mul(vecs[i][k], vecs[j][k]).sum().data.cpu().numpy()
                    dps[(j, i)] = dps[(i, j)]
                if (i,i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i,i)] += torch.mul(vecs[i][k], vecs[i][k]).sum().data.cpu().numpy()
                if (j,j) not in dps:
                    dps[(j, j)] = 0.0   
                    for k in range(len(vecs[i])):
                        dps[(j, j)] += torch.mul(vecs[j][k], vecs[j][k]).sum().data.cpu().numpy()
                c,d = MinNormSolver._min_norm_element_from2(dps[(i,i)], dps[(i,j)], dps[(j,j)])
                if d < dmin:
                    dmin = d
                    sol = [(i,j),c,d]
        return sol, dps

    def _projection2simplex(y):
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0)/m
        for i in range(m-1):
            tmpsum+= sorted_y[i]
            tmax = (tmpsum - 1)/ (i+1.0)
            if tmax > sorted_y[i+1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))
    
    def _next_point(cur_val, grad, n):
        proj_grad = grad - ( np.sum(grad) / n )
        tm1 = -1.0*cur_val[proj_grad<0]/proj_grad[proj_grad<0]
        tm2 = (1.0 - cur_val[proj_grad>0])/(proj_grad[proj_grad>0])
        
        skippers = np.sum(tm1<1e-7) + np.sum(tm2<1e-7)
        t = 1
        if len(tm1[tm1>1e-7]) > 0:
            t = np.min(tm1[tm1>1e-7])
        if len(tm2[tm2>1e-7]) > 0:
            t = min(t, np.min(tm2[tm2>1e-7]))

        next_point = proj_grad*t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    def find_min_norm_element_FW(vecs):
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n=len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            return sol_vec , init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                grad_mat[i,j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            iter_count += 1
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))

            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc*sol_vec
            new_sol_vec[t_iter] += 1 - nc

            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec
        
        return sol_vec, nd


def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == 'l2':
        for t in grads:
            gn[t] = torch.tensor(np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]])))
    elif normalization_type == 'loss':
        for t in grads:
            gn[t] = losses[t]
    elif normalization_type == 'loss+':
        for t in grads:
            gn[t] = losses[t] * np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))
    elif normalization_type == 'none':
        for t in grads:
            gn[t] = torch.tensor(1.0)
    else:
        print('ERROR: Invalid Normalization Type')
    return gn


# ============================================================================
# RCSYS_utils.py - EXACT COPY (relevant parts)
# ============================================================================

def split_data_new(edge_index, edge_label_index, test_size=0.2, val_size=0.25, seed=42):
    edges = edge_index.numpy().T
    train_edges, test_edges = train_test_split(edges, test_size=test_size, random_state=seed)
    train_edges, val_edges = train_test_split(train_edges, test_size=val_size, random_state=seed)

    train_edge_index = torch.LongTensor(train_edges).T
    val_edge_index = torch.LongTensor(val_edges).T
    test_edge_index = torch.LongTensor(test_edges).T

    def get_pos_neg_edge_indices(edge_label_index, edge_index):
        edge_label_set = set([tuple(edge_label_index[:, i].tolist()) for i in range(edge_label_index.size(1))])
        pos_edge_index = torch.tensor([edge for edge in edge_index.t().tolist() if tuple(edge) in edge_label_set]).t()
        neg_edge_index = torch.tensor([edge for edge in edge_index.t().tolist() if tuple(edge) not in edge_label_set]).t()
        return pos_edge_index, neg_edge_index

    pos_train_edge_index, neg_train_edge_index = get_pos_neg_edge_indices(edge_label_index, train_edge_index)
    pos_val_edge_index, neg_val_edge_index = get_pos_neg_edge_indices(edge_label_index, val_edge_index)
    pos_test_edge_index, neg_test_edge_index = get_pos_neg_edge_indices(edge_label_index, test_edge_index)

    return train_edge_index, val_edge_index, test_edge_index, \
            pos_train_edge_index, neg_train_edge_index, pos_val_edge_index, neg_val_edge_index, \
            pos_test_edge_index, neg_test_edge_index


def sample_mini_batch(batch_size, edge_index, seed=42):
    edges = structured_negative_sampling(edge_index)
    edges = torch.stack(edges, dim=0)
    indices = random.choices(
        [i for i in range(edges[0].shape[0])], k=batch_size)
    batch = edges[:, indices]
    user_indices, pos_item_indices, _ = batch[0], batch[1], batch[2]
    neg_item_indices = torch.randint(0, int(edge_index[1].max()-1), size=(batch_size,), dtype=torch.long)
    return user_indices, pos_item_indices, neg_item_indices


def bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0,
             lambda_val):
    reg_loss = lambda_val * (users_emb_0.norm(2).pow(2) +
                             pos_items_emb_0.norm(2).pow(2) +
                             neg_items_emb_0.norm(2).pow(2))

    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1)
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1)

    bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
    loss = bpr_loss + reg_loss

    return loss


def jaccard_similarity(user_tags, item_tags):
    intersection = torch.sum(torch.min(user_tags, item_tags), dim=1).float()
    union = torch.sum(torch.max(user_tags, item_tags), dim=1).float()
    jaccard = intersection / (union + 1e-8)
    return jaccard


def health_loss(users_emb_final, pos_items_emb_final, neg_items_emb_final, user_tags_batch, pos_item_tags_batch, neg_item_tags_batch):
    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1)
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1)

    pos_jaccard = jaccard_similarity(user_tags_batch, pos_item_tags_batch)
    neg_jaccard = jaccard_similarity(user_tags_batch, neg_item_tags_batch)
    jaccard = ((pos_jaccard - neg_jaccard) + 1 ) / 2
  
    health_loss = -torch.mean(torch.log(torch.mul(jaccard, torch.sigmoid(pos_scores - neg_scores))))

    return health_loss


def diversity_loss(users_emb_final, pos_items_emb_final, neg_items_emb_final, user_features_batch, pos_item_features_batch, neg_item_features_batch, k=20):
    def get_top_k_recommendations(user_emb, item_emb, k=10):
        scores = torch.matmul(user_emb, item_emb.T)
        _, top_k_indices = torch.topk(scores, k=k, dim=1)
        return top_k_indices
    
    def get_mean_similarity(user_features_batch, item_features_batch, k):
        top_k_indices = get_top_k_recommendations(user_features_batch, item_features_batch, k)
        top_k_item_embs = item_features_batch[top_k_indices]

        similarities = cosine_similarity(
            top_k_item_embs.unsqueeze(2),
            top_k_item_embs.unsqueeze(1),
            dim=3
        )

        upper_triangular_indices = torch.triu_indices(k, k, 1)
        selected_similarities = similarities[:, upper_triangular_indices[0], upper_triangular_indices[1]]

        return selected_similarities.mean(dim=1)

    pos_similarity = get_mean_similarity(user_features_batch, pos_item_features_batch, k)
    neg_similarity = get_mean_similarity(user_features_batch, neg_item_features_batch, k)

    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1)
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1)

    loss = -torch.mean(torch.log((torch.sigmoid(torch.mul(pos_similarity - neg_similarity, pos_scores - neg_scores)))))
    return loss


def get_user_positive_items(edge_index):
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)
    return user_pos_items


def RecallPrecision_ATk(groundTruth, r, k):
    num_correct_pred = torch.sum(r, dim=-1)
    user_num_liked = torch.Tensor([len(groundTruth[i])
                                   for i in range(len(groundTruth))])
    recall = torch.mean(num_correct_pred / user_num_liked)
    precision = torch.mean(num_correct_pred) / k
    return recall.item(), precision.item()


def NDCGatK_r(groundTruth, r, k):
    assert len(r) == len(groundTruth)

    test_matrix = torch.zeros((len(r), k))

    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = r * (1. / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return torch.mean(ndcg).item()


def calculate_health_score(users, top_K_items, user_tag, food_tag):
    user_tags = user_tag[users].cpu()
    recommended_items = top_K_items[users].cpu()
    
    food_tags = food_tag[recommended_items].cpu()

    user_tags_expanded = user_tags.unsqueeze(1)
    
    common_tag = torch.logical_and(user_tags_expanded, food_tags).sum(dim=2) > 0

    healthy_foods_ratio = common_tag.float().mean(dim=1)
    
    health_score = healthy_foods_ratio.mean().item()
    
    return health_score


def calculate_average_health_tags(users, top_K_items, food_tags):
    recommended_items = top_K_items[users].cpu()
    
    food_tags_recommended = food_tags[recommended_items]
    
    tags_per_food = food_tags_recommended.sum(dim=2)
    
    avg_tags_per_user = tags_per_food.mean(dim=1)
    
    avg_tags_across_users = avg_tags_per_user.mean().item()

    return avg_tags_across_users


def calculate_percentage_recommended_foods(users, top_K_items, num_foods):
    recommended_items = top_K_items[users].cpu().flatten().unique()
    percentage_recommended = len(recommended_items) / num_foods
    
    return percentage_recommended


def get_metrics(model, user_tags, food_tags, edge_index, exclude_edge_indices, k, users_emb_final, items_emb_final):
    user_embedding = users_emb_final
    item_embedding = items_emb_final

    rating = torch.matmul(user_embedding, item_embedding.T)

    for exclude_edge_index in exclude_edge_indices:
        user_pos_items = get_user_positive_items(exclude_edge_index)
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)

        rating[exclude_users, exclude_items] = -(1 << 10)

    _, top_K_items = torch.topk(rating, k=k)

    users = edge_index[0].unique()
    test_user_pos_items = get_user_positive_items(edge_index)
    test_user_pos_items_list = [
        test_user_pos_items[user.item()] for user in users]

    r = []
    for user in users:
        ground_truth_items = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in ground_truth_items, top_K_items[user]))
        r.append(label)
    r = torch.Tensor(np.array(r).astype('float'))

    recall, precision = RecallPrecision_ATk(test_user_pos_items_list, r, k)
    ndcg = NDCGatK_r(test_user_pos_items_list, r, k)
    health_score = calculate_health_score(users, top_K_items, user_tags, food_tags)
    avg_health_tags_ratio = calculate_average_health_tags(users, top_K_items, food_tags)
    num_foods = item_embedding.size(0)
    percentage_recommended_foods = calculate_percentage_recommended_foods(users, top_K_items, num_foods)

    return recall, precision, ndcg, health_score, avg_health_tags_ratio, percentage_recommended_foods


def eval(model, feature_dict, user_tags, food_tags, edge_index, pos_edge_index, neg_edge_index, 
               exclude_edge_indices, k, lambda_val):
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = \
            model.forward(feature_dict, edge_index, pos_edge_index, neg_edge_index)

    edges = structured_negative_sampling(
        edge_index, contains_neg_self_loops=False)
    user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]
    neg_item_indices = torch.randint(0, int(edge_index[1].max()-1), size=(len(neg_item_indices),), dtype=torch.long)

    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]
    
    loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0,
                    neg_items_emb_final, neg_items_emb_0, lambda_val).item()

    recall, precision, ndcg, health_score, avg_health_tags_ratio, percentage_recommended_foods = \
        get_metrics(model, user_tags, food_tags, edge_index, exclude_edge_indices, k, users_emb_final, items_emb_final)

    return loss, recall, precision, ndcg, health_score, avg_health_tags_ratio, percentage_recommended_foods


def pareto_loss(model, users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, 
                user_features_batch, pos_item_features_batch, neg_item_features_batch, 
                user_tags_batch, pos_item_tags_batch, neg_item_tags_batch, LAMBDA):
            
    loss_data = {}
    grads = {}
    tasks = ['bpr', 'sim', 'health']
    for task in tasks:
        if task == 'bpr':
            loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final,
                    pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, LAMBDA)
        elif task == 'sim':
            loss = diversity_loss(users_emb_final, pos_items_emb_final, neg_items_emb_final,
                                  user_features_batch, pos_item_features_batch, neg_item_features_batch)
        elif task == 'health':
            loss = health_loss(users_emb_final, pos_items_emb_final, neg_items_emb_final, 
                                         user_tags_batch, pos_item_tags_batch, neg_item_tags_batch)
        else:
            raise ValueError('Unknown task') 
        loss_data[task] = loss
        grads[task] = []
        loss.backward(retain_graph=True)
        for param in model.parameters():
            if param.grad is not None:
                grads[task].append(param.grad.data.detach().cpu())
        model.zero_grad()
    gn = gradient_normalizers(grads, loss_data, 'l2')
    for task in loss_data:
        for gr_i in range(len(grads[task])):
            grads[task][gr_i] = grads[task][gr_i] / gn[task].to(grads[task][gr_i].device)
    sol, _ = MinNormSolver.find_min_norm_element_FW([grads[task] for task in tasks])
    sol = {k:sol[i] for i, k in enumerate(tasks)}

    model.zero_grad()
    loss = 0
    actual_loss = 0

    for i, l in loss_data.items():
        loss += float(sol[i]) * l
        actual_loss += l
    
    return loss, loss_data, actual_loss


# ============================================================================
# RCSYS_models.py - EXACT COPY
# ============================================================================

class LightGCN(MessagePassing):
    """LightGCN Model as proposed in https://arxiv.org/abs/2002.02126
    """

    def __init__(self, num_users, num_items, embedding_dim=64, layers=3, add_self_loops=False):
        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.embedding_dim, self.layers = embedding_dim, layers
        self.add_self_loops = add_self_loops

        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)

        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def forward(self, edge_index: SparseTensor):
        edge_index_norm = gcn_norm(edge_index, add_self_loops=self.add_self_loops)

        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        embs = [emb_0]
        emb_k = emb_0

        for i in range(self.layers):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)

        users_emb_final, items_emb_final = torch.split(
            emb_final, [self.num_users, self.num_items])

        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x)


class MetricCalculator(nn.Module):
    """Applies a learnable transformation to the node features."""
    def __init__(self, feature_dim):
        super(MetricCalculator, self).__init__()
        self.weight = nn.Parameter(torch.empty((1, feature_dim)))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, node_features):
        return node_features * self.weight


class GraphGenerator(nn.Module):
    """
    Builds a graph based on similarity between node features from two different sets.
    """
    def __init__(self, feature_dim, num_heads=2, similarity_threshold=0.1):
        super(GraphGenerator, self).__init__()
        self.similarity_threshold = similarity_threshold
        self.metric_layers = nn.ModuleList([MetricCalculator(feature_dim) for _ in range(num_heads)])
        self.num_heads = num_heads

    def forward(self, left_features, right_features, edge_index):
        similarity_matrix = torch.zeros(edge_index.size(1)).to(edge_index.device)
        for metric_layer in self.metric_layers:
            weighted_left = metric_layer(left_features[edge_index[0]])
            weighted_right = metric_layer(right_features[edge_index[1]])
            similarity_matrix += F.cosine_similarity(weighted_left, weighted_right, dim=1)

        similarity_matrix /= self.num_heads
        return torch.where(similarity_matrix < self.similarity_threshold, torch.zeros_like(similarity_matrix), similarity_matrix)


class GraphChannelAttLayer(nn.Module):
    def __init__(self, num_channel):
        super(GraphChannelAttLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel))
        nn.init.constant_(self.weight, 0.1)

    def forward(self, edge_mask_list):
        edge_mask = torch.stack(edge_mask_list, dim=0)
        # Row normalization of all graphs generated
        edge_mask = F.normalize(edge_mask, dim=1, p=1)
        # Hadamard product + summation -> Conv
        softmax_weights = torch.softmax(self.weight, dim=0)
        
        weighted_edge_masks = edge_mask * softmax_weights[:, None]
        
        fused_edge_mask = torch.sum(weighted_edge_masks, dim=0)

        # Changed from > 0.5 to > 0 (bug fix for edge filtering)
        return fused_edge_mask > 0


class SignedGCN(torch.nn.Module):
    def __init__(self, num_users, num_foods, hidden_channels, num_layers):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_users = num_users
        self.num_items = num_foods

        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.hidden_channels)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.hidden_channels)

        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

        self.conv1 = SignedConv(hidden_channels, hidden_channels // 2,
                                first_aggr=True)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(
                SignedConv(hidden_channels // 2, hidden_channels // 2,
                           first_aggr=False))

        self.lin = torch.nn.Linear(2 * hidden_channels, 3)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

    def forward(
        self,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
    ) -> Tensor:
        x = torch.cat([self.users_emb.weight, self.items_emb.weight])
        z = F.relu(self.conv1(x, pos_edge_index, neg_edge_index))
        for conv in self.convs:
            z = F.relu(conv(z, pos_edge_index, neg_edge_index))
        return z

    def discriminate(self, z: Tensor, edge_index: Tensor) -> Tensor:
        value = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        value = self.lin(value)

        log_softmax_output = torch.log_softmax(value, dim=1)
        class_indices = torch.argmax(log_softmax_output, dim=1)

        mapping = torch.tensor([-1, 0, 1]).to(value.device)
        mapped_output = mapping[class_indices]

        return mapped_output


class SGSL(nn.Module):
    def __init__(self, graph, embedding_dim,  feature_threshold=0.3, num_heads=4, num_layer=3):
        super(SGSL, self).__init__()

        self.num_users = graph['user'].num_nodes
        self.num_foods = graph['food'].num_nodes
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layer = num_layer
        self.feature_threshold = feature_threshold

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in graph.node_types:
            self.lin_dict[node_type] = Linear(-1, embedding_dim)
        
        self.feature_graph_generator = GraphGenerator(self.embedding_dim, self.num_heads, self.feature_threshold)
        self.signed_layer = SignedGCN(self.num_users, self.num_foods, self.embedding_dim, self.num_layer)
        self.fusion = GraphChannelAttLayer(3)
        self.lightgcn = LightGCN(self.num_users, self.num_foods, self.embedding_dim, self.num_layer, False)


    def forward(self, feature_dict, edge_index, pos_edge_index, neg_edge_index):
        # Heterogeneous Feature Mapping.
        feature_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in feature_dict.items()
        }

        # Generate the feature graph
        mask_feature = self.feature_graph_generator(feature_dict['user'], feature_dict['food'], edge_index)
        mask_ori = torch.ones_like(mask_feature)

        # Generate the semantic graph
        z = self.signed_layer(pos_edge_index, neg_edge_index)
        mask_semantic = self.signed_layer.discriminate(z, edge_index)

        # Fusion with attention 
        edge_mask = self.fusion([mask_ori, mask_feature, mask_semantic])

        edge_index_new = edge_index[:, edge_mask]
        sparse_size = self.num_users + self.num_foods
        sparse_edge_index = SparseTensor(row=edge_index_new[0], col=edge_index_new[1], sparse_sizes=(
            sparse_size, sparse_size))
        
        # LightGCN on the new graph
        return self.lightgcn(sparse_edge_index)


# ============================================================================
# main.py - EXACT COPY
# ============================================================================

def main(args):
    SEED = args.seed
    BATCH_SIZE = args.batch_size
    LAMBDA = args.LAMBDA
    HIDDEN_DIM = args.hidden_dim
    LAYERS = args.layers
    LR = args.lr
    TH = args.feature_threshold

    torch_geometric.seed_everything(SEED)
    
    # Data loading - use path relative to project root
    data_path = os.path.join(os.path.dirname(__file__), 'MOPI-HFRS_gdrive', 'processed_data', 'benchmark_macro.pt')
    print(f"Loading data from: {data_path}")
    graph = torch.load(data_path, weights_only=False)
    
    num_users, num_foods = graph['user'].num_nodes, graph['food'].num_nodes
    edge_index = graph[('user', 'eats', 'food')].edge_index
    edge_label_index = graph[('user', 'eats', 'food')].edge_label_index
    feature_dict = graph.x_dict

    print(f"Users: {num_users}, Foods: {num_foods}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge label index shape: {edge_label_index.shape}")

    train_edge_index, val_edge_index, test_edge_index, \
    pos_train_edge_index, neg_train_edge_index, pos_val_edge_index, neg_val_edge_index, \
    pos_test_edge_index, neg_test_edge_index = split_data_new(edge_index, edge_label_index)

    print(f"Train edges: {train_edge_index.shape[1]}")
    print(f"Pos train edges: {pos_train_edge_index.shape[1]}")
    print(f"Neg train edges: {neg_train_edge_index.shape[1]}")

    model = SGSL(graph, embedding_dim=HIDDEN_DIM, feature_threshold=TH, num_layer=LAYERS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_edge_index = train_edge_index.to(device)
    val_edge_index = val_edge_index.to(device)
    test_edge_index = test_edge_index.to(device)
    pos_train_edge_index = pos_train_edge_index.to(device)
    neg_train_edge_index = neg_train_edge_index.to(device)
    pos_val_edge_index = pos_val_edge_index.to(device)
    neg_val_edge_index = neg_val_edge_index.to(device)
    pos_test_edge_index = pos_test_edge_index.to(device)
    neg_test_edge_index = neg_test_edge_index.to(device)

    feature_dict = {key: x.to(device) for key, x in feature_dict.items()}
    user_tags = graph['user'].tags.to(device)
    food_tags = graph['food'].tags.to(device)
    user_features = graph['user'].x.to(device)
    food_features = graph['food'].x.to(device) 

    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    train_losses = []
    val_losses = []

    print(f"\n{'='*60}")
    print("Starting Training (Original MOPI-HFRS Code)")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        # forward propagation
        users_emb_final, users_emb_0, items_emb_final, items_emb_0 = \
            model.forward(feature_dict, train_edge_index, pos_train_edge_index, neg_train_edge_index)
        
        # mini batching
        user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(BATCH_SIZE, train_edge_index)
        user_indices = user_indices.to(device)
        pos_item_indices = pos_item_indices.to(device)
        neg_item_indices = neg_item_indices.to(device)
        
        users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
        pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
        neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]

        user_tags_batch = user_tags[user_indices]
        pos_item_tags_batch = food_tags[pos_item_indices]
        neg_item_tags_batch = food_tags[neg_item_indices]

        # Pad user features
        user_features_batch = user_features[user_indices]
        user_features_batch = torch.nn.functional.pad(user_features_batch, (0, food_features.size(1) - user_features_batch.size(1)))

        pos_item_features_batch = food_features[pos_item_indices]
        neg_item_features_batch = food_features[neg_item_indices]

        ### Pareto Loss ### 
        train_loss, loss_data, _ = pareto_loss(model, users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, 
                        user_features_batch, pos_item_features_batch, neg_item_features_batch, 
                        user_tags_batch, pos_item_tags_batch, neg_item_tags_batch, LAMBDA)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch % args.iters_per_eval == 0 and epoch != 0:
            model.eval()
            val_loss, recall, precision, ndcg, health_score, avg_health_tags_ratio, percentage_recommended_foods = \
                eval(model, feature_dict, user_tags, food_tags, val_edge_index, pos_val_edge_index, neg_val_edge_index,
                                [neg_train_edge_index], args.K, LAMBDA)
            
            print(f"Epoch: {epoch}, "
                  f"train_loss: {round(train_loss.item(), 5)}, "
                  f"val_loss: {round(val_loss, 5)}, "
                  f"val_recall@{args.K}: {round(recall, 5)}, "
                  f"val_precision@{args.K}: {round(precision, 5)}, "
                  f"val_ndcg@{args.K}: {round(ndcg, 5)}, "
                  f"val_health_score: {round(health_score, 5)}, "
                  f"avg_health_tags_ratio: {round(avg_health_tags_ratio, 5)}, "
                  f"percentage_recommended_foods: {round(percentage_recommended_foods, 5)}")

            train_losses.append(train_loss.item())
            val_losses.append(val_loss)
            model.train()

        if epoch % args.iters_per_lr_decay == 0 and epoch != 0:
            scheduler.step()

    with torch.no_grad():
        model.eval()
        _, recall, precision, ndcg, health_score, avg_health_tags_ratio, percentage_recommended_foods = \
                eval(model, feature_dict, user_tags, food_tags, test_edge_index, pos_test_edge_index, neg_test_edge_index,
                                [neg_train_edge_index], args.K, LAMBDA)
        
        print(f"\n{'='*60}")
        print("Final Test Results (Original MOPI-HFRS Code)")
        print(f"{'='*60}")
        print(f"test_recall@{args.K}: {round(recall, 5)}, "
              f"test_precision@{args.K}: {round(precision, 5)}, "
              f"test_ndcg@{args.K}: {round(ndcg, 5)}, "
              f"test_health_score: {round(health_score, 5)}, "
              f"avg_health_tags_ratio: {round(avg_health_tags_ratio, 5)}, "
              f"percentage_recommended_foods: {round(percentage_recommended_foods, 5)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Number of hidden dimension.')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size.')
    parser.add_argument('--K', type=int, default=20,
                        help='Number of ranking list.')
    parser.add_argument('--LAMBDA', type=float, default=1e-6,
                        help='Regularization coefficient.')
    parser.add_argument('--iters_per_eval', type=int, default=500,
                        help='Iterations per evaluation.')
    parser.add_argument('--iters_per_lr_decay', type=int, default=200,
                        help='Iterations per learning rate decay.')
    parser.add_argument('--layers', type=int, default=3,
                        help='Number of layers in the model.')
    parser.add_argument('--feature_threshold', type=float, default=0.3,
                        help='Threshold for feature selection.')
    args = parser.parse_args()

    main(args)

