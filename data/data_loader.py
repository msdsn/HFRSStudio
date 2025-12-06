"""
Data loader module for MOPI-HFRS.
Supports both:
1. Loading pre-processed .pt benchmark files
2. Converting CSV files to PyTorch Geometric HeteroData graph
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional, Union
from pathlib import Path


# Health tag columns for users - MACRO only (7 nutrients)
USER_HEALTH_TAGS_MACRO = [
    'user_low_calorie', 'user_high_calorie',
    'user_low_carb',
    'user_low_protein', 'user_high_protein',
    'user_low_saturated_fat',
    'user_low_cholesterol',
    'user_low_sugar',
    'user_high_fiber'
]

# Health tag columns for users - ALL (16 nutrients: 7 macro + 9 micro)
USER_HEALTH_TAGS_ALL = [
    # Macro nutrients (7)
    'user_low_calorie', 'user_high_calorie',
    'user_low_carb',
    'user_low_protein', 'user_high_protein',
    'user_low_saturated_fat',
    'user_low_cholesterol',
    'user_low_sugar',
    'user_high_fiber',
    # Micro nutrients (9)
    'user_low_sodium',
    'user_high_potassium',
    'user_low_phosphorus',
    'user_high_iron',
    'user_high_calcium',
    'user_high_folate_acid',
    'user_high_vitamin_c',
    'user_high_vitamin_d',
    'user_high_vitamin_b12'
]

# Health tag columns for foods - MACRO only
FOOD_HEALTH_TAGS_MACRO = [
    'low_calorie', 'high_calorie',
    'low_carb', 'high_carb',
    'low_protein', 'high_protein',
    'low_saturated_fat', 'high_saturated_fat',
    'low_cholesterol', 'high_cholesterol',
    'low_sugar', 'high_sugar',
    'low_fiber', 'high_fiber'
]

# Health tag columns for foods - ALL
FOOD_HEALTH_TAGS_ALL = [
    # Macro nutrients
    'low_calorie', 'high_calorie',
    'low_carb', 'high_carb',
    'low_protein', 'high_protein',
    'low_saturated_fat', 'high_saturated_fat',
    'low_cholesterol', 'high_cholesterol',
    'low_sugar', 'high_sugar',
    'low_fiber', 'high_fiber',
    # Micro nutrients
    'low_sodium', 'high_sodium',
    'low_potassium', 'high_potassium',
    'low_phosphorus', 'high_phosphorus',
    'low_iron', 'high_iron',
    'low_calcium', 'high_calcium',
    'low_folic_acid', 'high_folic_acid',
    'low_vitamin_c', 'high_vitamin_c',
    'low_vitamin_d', 'high_vitamin_d',
    'low_vitamin_b12', 'high_vitamin_b12'
]

# Default to ALL for backward compatibility
USER_HEALTH_TAGS = USER_HEALTH_TAGS_ALL
FOOD_HEALTH_TAGS = FOOD_HEALTH_TAGS_ALL


def get_health_tags(benchmark_type: str = 'all'):
    """
    Get health tag columns based on benchmark type.
    
    Args:
        benchmark_type: 'macro' for 7 nutrients, 'all' for 16 nutrients
        
    Returns:
        Tuple of (user_tags, food_tags)
    """
    if benchmark_type == 'macro':
        return USER_HEALTH_TAGS_MACRO, FOOD_HEALTH_TAGS_MACRO
    else:
        return USER_HEALTH_TAGS_ALL, FOOD_HEALTH_TAGS_ALL

# Nutrient columns for food features
NUTRIENT_COLUMNS = [
    'calorie', 'protein', 'carb', 'sugar', 'fiber', 'saturated_fat',
    'cholesterol', 'sodium', 'calcium', 'phosphorus', 'potassium',
    'iron', 'folic_acid', 'vitamin_c', 'vitamin_d', 'vitamin_b12'
]

# User demographic columns
USER_DEMOGRAPHIC_COLUMNS = ['gender', 'age', 'race', 'household_income', 'education']


def load_csv_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all CSV files from the data directory.
    
    Args:
        data_dir: Path to directory containing CSV files
        
    Returns:
        Tuple of (user_df, food_df, interaction_df, fndds_df)
    """
    data_path = Path(data_dir)
    
    user_df = pd.read_csv(data_path / 'user_tagging.csv')
    food_df = pd.read_csv(data_path / 'food_tagging.csv')
    interaction_df = pd.read_csv(data_path / 'food_user.csv')
    fndds_df = pd.read_csv(data_path / 'fndds.csv')
    
    print(f"Loaded {len(user_df)} users, {len(food_df)} foods, {len(interaction_df)} interactions")
    
    return user_df, food_df, interaction_df, fndds_df


def create_user_features(
    user_df: pd.DataFrame, 
    normalize: bool = True,
    benchmark_type: str = 'all'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create user feature tensor and health tag tensor.
    
    Args:
        user_df: User dataframe with demographic and health tag columns
        normalize: Whether to normalize demographic features
        benchmark_type: 'macro' for 7 nutrients, 'all' for 16 nutrients
        
    Returns:
        Tuple of (features, tags) tensors
    """
    # Get appropriate health tags based on benchmark type
    user_tags_cols, _ = get_health_tags(benchmark_type)
    
    # Filter to existing columns only
    existing_user_tags = [col for col in user_tags_cols if col in user_df.columns]
    
    # Extract demographic features
    demographic_features = user_df[USER_DEMOGRAPHIC_COLUMNS].fillna(0).values.astype(np.float32)
    
    if normalize:
        scaler = StandardScaler()
        demographic_features = scaler.fit_transform(demographic_features)
    
    # Extract health tags
    health_tags = user_df[existing_user_tags].fillna(0).values.astype(np.float32)
    
    # Combine features (demographic + health tags)
    features = np.concatenate([demographic_features, health_tags], axis=1)
    
    print(f"User features: {features.shape[1]} dims (demo: {len(USER_DEMOGRAPHIC_COLUMNS)}, tags: {len(existing_user_tags)})")
    
    return torch.tensor(features, dtype=torch.float32), torch.tensor(health_tags, dtype=torch.float32)


def create_food_features(
    food_df: pd.DataFrame, 
    normalize: bool = True,
    benchmark_type: str = 'all'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create food feature tensor and health tag tensor.
    
    Args:
        food_df: Food dataframe with nutrient and health tag columns
        normalize: Whether to normalize nutrient features
        benchmark_type: 'macro' for 7 nutrients, 'all' for 16 nutrients
        
    Returns:
        Tuple of (features, tags) tensors
    """
    # Get appropriate health tags based on benchmark type
    _, food_tags_cols = get_health_tags(benchmark_type)
    
    # Filter to existing columns only
    existing_food_tags = [col for col in food_tags_cols if col in food_df.columns]
    
    # Extract nutrient features
    nutrient_features = food_df[NUTRIENT_COLUMNS].fillna(0).values.astype(np.float32)
    
    if normalize:
        scaler = StandardScaler()
        nutrient_features = scaler.fit_transform(nutrient_features)
    
    # Extract health tags
    health_tags = food_df[existing_food_tags].fillna(0).values.astype(np.float32)
    
    # Combine features (nutrients + health tags)
    features = np.concatenate([nutrient_features, health_tags], axis=1)
    
    print(f"Food features: {features.shape[1]} dims (nutrients: {len(NUTRIENT_COLUMNS)}, tags: {len(existing_food_tags)})")
    
    return torch.tensor(features, dtype=torch.float32), torch.tensor(health_tags, dtype=torch.float32)


def create_edge_index(
    interaction_df: pd.DataFrame,
    user_id_map: Dict[int, int],
    food_id_map: Dict[str, int]
) -> torch.Tensor:
    """
    Create edge index from interaction dataframe (vectorized).
    
    Args:
        interaction_df: Interaction dataframe with SEQN and food_id columns
        user_id_map: Mapping from SEQN to user index
        food_id_map: Mapping from food_id to food index
        
    Returns:
        Edge index tensor of shape (2, num_edges)
    """
    # Convert food_id to string for mapping
    interaction_df = interaction_df.copy()
    interaction_df['food_id'] = interaction_df['food_id'].astype(str)
    
    # Vectorized mapping using pandas map
    user_indices = interaction_df['SEQN'].map(user_id_map)
    food_indices = interaction_df['food_id'].map(food_id_map)
    
    # Filter out unmapped values (NaN)
    valid_mask = user_indices.notna() & food_indices.notna()
    user_indices = user_indices[valid_mask].astype(int).values
    food_indices = food_indices[valid_mask].astype(int).values
    
    edge_index = torch.tensor(np.stack([user_indices, food_indices]), dtype=torch.long)
    
    return edge_index


def compute_health_edge_labels(
    edge_index: torch.Tensor,
    user_tags: torch.Tensor,
    food_tags: torch.Tensor,
    matching_threshold: int = 1,
    batch_size: int = 100000
) -> torch.Tensor:
    """
    Compute positive/negative edge labels based on health tag matching (batched for memory efficiency).
    An edge is positive if user and food share at least `matching_threshold` matching tags.
    
    The matching is based on comparing user needs (e.g., 'user_low_sodium') with food properties
    (e.g., 'low_sodium'). A match occurs when both are 1.
    
    Args:
        edge_index: Edge index tensor (2, num_edges)
        user_tags: User health tags tensor (num_users, num_user_tags)
        food_tags: Food health tags tensor (num_foods, num_food_tags)
        matching_threshold: Minimum number of matching tags for positive edge
        batch_size: Batch size for processing to avoid memory issues
        
    Returns:
        Edge label tensor (1 for positive/healthy, 0 for negative/unhealthy)
    """
    num_edges = edge_index.shape[1]
    num_user_tags = user_tags.shape[1]
    num_food_tags = food_tags.shape[1]
    
    # For each user tag, find the corresponding food tag index
    # User tags are like: user_low_calorie, user_high_calorie, user_low_carb, ...
    # Food tags are like: low_calorie, high_calorie, low_carb, high_carb, ...
    # We need to match them correctly
    
    # Simple approach: Use min of tag counts and match directly
    # This works because tags are ordered similarly (both start with calorie, carb, protein, etc.)
    num_matching_tags = min(num_user_tags, num_food_tags)
    
    edge_labels = torch.zeros(num_edges, dtype=torch.long)
    
    # Process in batches to avoid memory issues
    for start_idx in range(0, num_edges, batch_size):
        end_idx = min(start_idx + batch_size, num_edges)
        
        batch_user_indices = edge_index[0, start_idx:end_idx]
        batch_food_indices = edge_index[1, start_idx:end_idx]
        
        # Get tags for batch (use first num_matching_tags columns)
        batch_user_tags = user_tags[batch_user_indices][:, :num_matching_tags]
        batch_food_tags = food_tags[batch_food_indices][:, :num_matching_tags]
        
        # Count matching tags (both user needs and food provides)
        # A match is when both user_tag[i] == 1 AND food_tag[i] == 1
        matching_counts = (batch_user_tags * batch_food_tags).sum(dim=1)
        
        # Set labels
        edge_labels[start_idx:end_idx] = (matching_counts >= matching_threshold).long()
    
    return edge_labels


def create_hetero_graph(
    data_dir: str,
    normalize_features: bool = True,
    benchmark_type: str = 'all'
) -> HeteroData:
    """
    Create a HeteroData graph from CSV files.
    
    Args:
        data_dir: Path to directory containing CSV files
        normalize_features: Whether to normalize numeric features
        benchmark_type: 'macro' for 7 nutrients, 'all' for 16 nutrients
        
    Returns:
        HeteroData graph with user and food nodes, and interaction edges
    """
    print(f"Creating graph with benchmark_type='{benchmark_type}'")
    
    # Load CSV data
    user_df, food_df, interaction_df, fndds_df = load_csv_data(data_dir)
    
    # Create ID mappings
    user_ids = user_df['SEQN'].unique()
    food_ids = food_df['food_id'].astype(str).unique()
    
    user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    food_id_map = {fid: idx for idx, fid in enumerate(food_ids)}
    
    # Reindex dataframes
    user_df = user_df.set_index('SEQN').loc[user_ids].reset_index()
    food_df['food_id'] = food_df['food_id'].astype(str)
    food_df = food_df.set_index('food_id').loc[food_ids].reset_index()
    
    # Create features with appropriate benchmark type
    user_features, user_tags = create_user_features(user_df, normalize=normalize_features, benchmark_type=benchmark_type)
    food_features, food_tags = create_food_features(food_df, normalize=normalize_features, benchmark_type=benchmark_type)
    
    # Create edge index
    edge_index = create_edge_index(interaction_df, user_id_map, food_id_map)
    
    # Compute health-based edge labels
    edge_labels = compute_health_edge_labels(edge_index, user_tags, food_tags)
    
    # Create positive and negative edge indices
    pos_mask = edge_labels == 1
    neg_mask = edge_labels == 0
    
    pos_edge_index = edge_index[:, pos_mask]
    neg_edge_index = edge_index[:, neg_mask]
    
    # Create HeteroData
    data = HeteroData()
    
    # Add user nodes
    data['user'].x = user_features
    data['user'].tags = user_tags
    data['user'].num_nodes = len(user_ids)
    data['user'].node_id = torch.arange(len(user_ids))
    
    # Add food nodes
    data['food'].x = food_features
    data['food'].tags = food_tags
    data['food'].num_nodes = len(food_ids)
    data['food'].node_id = torch.arange(len(food_ids))
    
    # Add edges (user eats food)
    data['user', 'eats', 'food'].edge_index = edge_index
    data['user', 'eats', 'food'].edge_label_index = pos_edge_index
    
    # Store positive and negative edge indices separately
    data['user', 'eats', 'food'].pos_edge_index = pos_edge_index
    data['user', 'eats', 'food'].neg_edge_index = neg_edge_index
    
    # Add reverse edges (food eaten by user)
    data['food', 'eaten_by', 'user'].edge_index = edge_index.flip(0)
    
    # Store metadata
    data.user_id_map = user_id_map
    data.food_id_map = food_id_map
    
    print(f"Created graph with {data['user'].num_nodes} users, {data['food'].num_nodes} foods")
    print(f"Total edges: {edge_index.shape[1]}, Positive: {pos_edge_index.shape[1]}, Negative: {neg_edge_index.shape[1]}")
    
    return data


def split_edges(
    edge_index: torch.Tensor,
    edge_label_index: torch.Tensor,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    """
    Split edges into train/val/test sets.
    
    Args:
        edge_index: Full edge index tensor
        edge_label_index: Positive edge index (for label split)
        train_ratio: Ratio of training edges
        val_ratio: Ratio of validation edges
        seed: Random seed
        
    Returns:
        Dictionary with train/val/test edge indices
    """
    num_edges = edge_index.shape[1]
    indices = np.arange(num_edges)
    
    # Split indices
    train_indices, temp_indices = train_test_split(
        indices, train_size=train_ratio, random_state=seed
    )
    val_size = val_ratio / (1 - train_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices, train_size=val_size, random_state=seed
    )
    
    return {
        'train_edge_index': edge_index[:, train_indices],
        'val_edge_index': edge_index[:, val_indices],
        'test_edge_index': edge_index[:, test_indices],
        'train_indices': torch.tensor(train_indices),
        'val_indices': torch.tensor(val_indices),
        'test_indices': torch.tensor(test_indices)
    }


class HFRSDataset:
    """Dataset class for MOPI-HFRS model that loads from CSV files."""
    
    def __init__(
        self,
        data_dir: str,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        seed: int = 42,
        normalize: bool = True,
        benchmark_type: str = 'all'
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to directory containing CSV files
            train_ratio: Ratio of training edges
            val_ratio: Ratio of validation edges
            seed: Random seed
            normalize: Whether to normalize features
            benchmark_type: 'macro' for 7 nutrients, 'all' for 16 nutrients
        """
        self.data_dir = data_dir
        self.seed = seed
        self.benchmark_type = benchmark_type
        
        # Create graph
        self.graph = create_hetero_graph(data_dir, normalize_features=normalize, benchmark_type=benchmark_type)
        
        # Split edges
        edge_index = self.graph['user', 'eats', 'food'].edge_index
        edge_label_index = self.graph['user', 'eats', 'food'].edge_label_index
        
        self.splits = split_edges(edge_index, edge_label_index, train_ratio, val_ratio, seed)
        
        # Store convenience attributes
        self.num_users = self.graph['user'].num_nodes
        self.num_foods = self.graph['food'].num_nodes
        self.user_features = self.graph['user'].x
        self.food_features = self.graph['food'].x
        self.user_tags = self.graph['user'].tags
        self.food_tags = self.graph['food'].tags
        
    def get_train_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get training edge indices."""
        pos_edge_index = self.graph['user', 'eats', 'food'].pos_edge_index
        neg_edge_index = self.graph['user', 'eats', 'food'].neg_edge_index
        
        # Filter to training set
        train_mask = torch.isin(
            self.graph['user', 'eats', 'food'].edge_index[0] * self.num_foods + 
            self.graph['user', 'eats', 'food'].edge_index[1],
            self.splits['train_edge_index'][0] * self.num_foods + 
            self.splits['train_edge_index'][1]
        )
        
        return self.splits['train_edge_index'], pos_edge_index, neg_edge_index
    
    def get_val_data(self) -> torch.Tensor:
        """Get validation edge indices."""
        return self.splits['val_edge_index']
    
    def get_test_data(self) -> torch.Tensor:
        """Get test edge indices."""
        return self.splits['test_edge_index']
    
    def get_feature_dict(self) -> Dict[str, torch.Tensor]:
        """Get feature dictionary for model input."""
        return {
            'user': self.user_features,
            'food': self.food_features
        }
    
    def to(self, device: torch.device) -> 'HFRSDataset':
        """Move all tensors to device."""
        self.graph = self.graph.to(device)
        self.user_features = self.user_features.to(device)
        self.food_features = self.food_features.to(device)
        self.user_tags = self.user_tags.to(device)
        self.food_tags = self.food_tags.to(device)
        
        for key in self.splits:
            if isinstance(self.splits[key], torch.Tensor):
                self.splits[key] = self.splits[key].to(device)
        
        return self


def save_graph(graph: HeteroData, path: str):
    """Save HeteroData graph to file."""
    torch.save(graph, path)
    print(f"Saved graph to {path}")


def load_graph(path: str) -> HeteroData:
    """Load HeteroData graph from file."""
    graph = torch.load(path, weights_only=False)
    print(f"Loaded graph from {path}")
    return graph


class HFRSDatasetFromPT:
    """
    Dataset class that loads pre-processed .pt benchmark files.
    Compatible with the original MOPI-HFRS benchmark files.
    """
    
    def __init__(
        self,
        pt_file: str,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        seed: int = 42
    ):
        """
        Initialize dataset from .pt file.
        
        Args:
            pt_file: Path to .pt benchmark file (e.g., benchmark_macro.pt)
            train_ratio: Ratio of training edges
            val_ratio: Ratio of validation edges
            seed: Random seed
        """
        self.pt_file = pt_file
        self.seed = seed
        
        # Load graph
        print(f"Loading graph from {pt_file}...")
        self.graph = torch.load(pt_file, weights_only=False)
        
        # Get graph info
        self.num_users = self.graph['user'].num_nodes
        self.num_foods = self.graph['food'].num_nodes
        
        # Get features
        self.user_features = self.graph['user'].x
        self.food_features = self.graph['food'].x
        
        # Get tags
        self.user_tags = self.graph['user'].tags
        self.food_tags = self.graph['food'].tags
        
        # Get edges
        edge_index = self.graph[('user', 'eats', 'food')].edge_index
        edge_label_index = self.graph[('user', 'eats', 'food')].edge_label_index
        
        # Split edges
        self.splits = self._split_edges(edge_index, edge_label_index, train_ratio, val_ratio, seed)
        
        print(f"Loaded: {self.num_users} users, {self.num_foods} foods")
        print(f"Edges: {edge_index.shape[1]} total, {self.splits['train_edge_index'].shape[1]} train")
    
    def _split_edges(
        self,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
        train_ratio: float,
        val_ratio: float,
        seed: int
    ) -> Dict[str, torch.Tensor]:
        """Split edges into train/val/test."""
        num_edges = edge_index.shape[1]
        indices = np.arange(num_edges)
        
        train_indices, temp_indices = train_test_split(
            indices, train_size=train_ratio, random_state=seed
        )
        val_size = val_ratio / (1 - train_ratio)
        val_indices, test_indices = train_test_split(
            temp_indices, train_size=val_size, random_state=seed
        )
        
        # Get positive/negative edge indices from edge_label_index
        edge_label_set = set(
            tuple(edge_label_index[:, i].tolist()) 
            for i in range(edge_label_index.size(1))
        )
        
        pos_mask = torch.tensor([
            tuple(edge_index[:, i].tolist()) in edge_label_set 
            for i in range(edge_index.size(1))
        ])
        
        pos_edge_index = edge_index[:, pos_mask]
        neg_edge_index = edge_index[:, ~pos_mask]
        
        return {
            'train_edge_index': edge_index[:, train_indices],
            'val_edge_index': edge_index[:, val_indices],
            'test_edge_index': edge_index[:, test_indices],
            'pos_edge_index': pos_edge_index,
            'neg_edge_index': neg_edge_index
        }
    
    def get_feature_dict(self) -> Dict[str, torch.Tensor]:
        """Get feature dictionary for model input."""
        return {
            'user': self.user_features,
            'food': self.food_features
        }
    
    def to(self, device: torch.device) -> 'HFRSDatasetFromPT':
        """Move all tensors to device."""
        self.user_features = self.user_features.to(device)
        self.food_features = self.food_features.to(device)
        self.user_tags = self.user_tags.to(device)
        self.food_tags = self.food_tags.to(device)
        
        for key in self.splits:
            if isinstance(self.splits[key], torch.Tensor):
                self.splits[key] = self.splits[key].to(device)
        
        return self


if __name__ == '__main__':
    # Test data loading
    import sys
    
    data_dir = '../MOPI-HFRS_gdrive/processed_data'
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    # Create dataset
    dataset = HFRSDataset(data_dir)
    
    print(f"\nDataset summary:")
    print(f"  Users: {dataset.num_users}")
    print(f"  Foods: {dataset.num_foods}")
    print(f"  User features shape: {dataset.user_features.shape}")
    print(f"  Food features shape: {dataset.food_features.shape}")
    print(f"  Train edges: {dataset.splits['train_edge_index'].shape[1]}")
    print(f"  Val edges: {dataset.splits['val_edge_index'].shape[1]}")
    print(f"  Test edges: {dataset.splits['test_edge_index'].shape[1]}")

