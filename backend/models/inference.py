"""
MOPI-HFRS Model Inference Layer.
Handles model loading, embedding computation, and recommendation generation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from functools import lru_cache
import sys

# Add parent directory to path for importing models
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import get_settings

settings = get_settings()


class RecommendationModel:
    """
    Wrapper for MOPI-HFRS model inference.
    Handles model loading, caching, and recommendation generation.
    """
    
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        """
        Initialize the recommendation model.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run model on (cpu/cuda)
        """
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.dataset = None
        self.food_data = None
        self.user_embeddings = None
        self.food_embeddings = None
        
        self._load_model()
        self._load_food_data()
    
    def _load_model(self):
        """Load model from checkpoint."""
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            
            # Extract model state and metadata
            self.model_state = checkpoint.get('model_state_dict', checkpoint)
            self.model_config = checkpoint.get('config', {})
            
            # Get dataset info from checkpoint
            self.num_users = checkpoint.get('num_users', 8170)
            self.num_foods = checkpoint.get('num_foods', 6769)
            self.user_feature_dim = checkpoint.get('user_feature_dim', 38)
            self.food_feature_dim = checkpoint.get('food_feature_dim', 66)
            
            # Pre-computed embeddings if available
            if 'user_embeddings' in checkpoint:
                self.user_embeddings = checkpoint['user_embeddings'].to(self.device)
            if 'food_embeddings' in checkpoint:
                self.food_embeddings = checkpoint['food_embeddings'].to(self.device)
            
            print(f"Loaded model checkpoint from {self.checkpoint_path}")
            print(f"  Users: {self.num_users}, Foods: {self.num_foods}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Use default embeddings for demo
            self._create_demo_embeddings()
    
    def _create_demo_embeddings(self):
        """Create demo embeddings if model loading fails."""
        embedding_dim = 128
        self.num_users = 8170
        self.num_foods = 6769
        
        # Random embeddings for demo purposes
        torch.manual_seed(42)
        self.user_embeddings = F.normalize(
            torch.randn(self.num_users, embedding_dim), dim=1
        ).to(self.device)
        self.food_embeddings = F.normalize(
            torch.randn(self.num_foods, embedding_dim), dim=1
        ).to(self.device)
        
        print("Using demo embeddings (model not loaded)")
    
    def _load_food_data(self):
        """Load food metadata from CSV files."""
        try:
            # Load from backend/data directory
            data_dir = Path(__file__).parent.parent / "data"
            
            if not data_dir.exists():
                # Fallback to original location
                data_dir = Path(__file__).parent.parent.parent / "MOPI-HFRS_gdrive" / "processed_data"
            
            food_path = data_dir / "food_tagging.csv"
            fndds_path = data_dir / "fndds.csv"
            
            if food_path.exists():
                self.food_df = pd.read_csv(food_path)
                print(f"Loaded food tagging data: {len(self.food_df)} foods")
            else:
                self.food_df = self._create_demo_food_data()
            
            # Load fndds for food names and categories
            if fndds_path.exists():
                self.fndds_df = pd.read_csv(fndds_path)
                # Create a lookup dict for food names (deduplicate by taking first)
                # Use int keys for consistent lookup
                grouped = self.fndds_df.groupby('food_id').first()[['food_desc', 'WWEIA_desc']]
                self.food_names = {int(k): v for k, v in grouped.to_dict('index').items()}
                print(f"Loaded food names: {len(self.food_names)} unique foods")
            else:
                self.fndds_df = None
                self.food_names = {}
                
        except Exception as e:
            print(f"Error loading food data: {e}")
            self.food_df = self._create_demo_food_data()
            self.fndds_df = None
            self.food_names = {}
    
    def _create_demo_food_data(self) -> pd.DataFrame:
        """Create demo food data for testing."""
        foods = [
            {"food_id": "11111000", "food_desc": "Milk, whole", "WWEIA_desc": "Milk, whole", 
             "calorie": 61, "protein": 3.2, "carb": 4.8, "sugar": 5.0, "fiber": 0,
             "low_calorie": 0, "high_protein": 0, "low_sodium": 1},
            {"food_id": "57123000", "food_desc": "Cereal (Cheerios)", "WWEIA_desc": "Ready-to-eat cereal",
             "calorie": 100, "protein": 3.0, "carb": 20, "sugar": 1.0, "fiber": 3.0,
             "low_calorie": 1, "high_fiber": 1, "low_sodium": 0},
            {"food_id": "94000100", "food_desc": "Water, tap", "WWEIA_desc": "Tap water",
             "calorie": 0, "protein": 0, "carb": 0, "sugar": 0, "fiber": 0,
             "low_calorie": 1, "low_sodium": 1, "low_sugar": 1},
        ]
        return pd.DataFrame(foods)
    
    def get_food_info(self, food_idx: int) -> Dict[str, Any]:
        """Get food information by index."""
        if food_idx >= len(self.food_df):
            return {"food_id": str(food_idx), "food_name": f"Food {food_idx}", "category": "Unknown"}
        
        row = self.food_df.iloc[food_idx]
        food_id = row.get('food_id', food_idx)
        
        # Nutrient columns
        nutrient_cols = ['calorie', 'protein', 'carb', 'sugar', 'fiber', 'saturated_fat',
                        'cholesterol', 'sodium', 'calcium', 'phosphorus', 'potassium',
                        'iron', 'folic_acid', 'vitamin_c', 'vitamin_d', 'vitamin_b12']
        
        # Health tag columns
        tag_cols = [c for c in self.food_df.columns if c.startswith(('low_', 'high_'))]
        
        nutrients = {}
        for col in nutrient_cols:
            if col in row:
                nutrients[col] = float(row[col]) if pd.notna(row[col]) else 0.0
        
        health_tags = {}
        for col in tag_cols:
            if col in row and row[col] == 1:
                health_tags[col] = True
        
        # Get food name and category from fndds lookup
        food_name = f"Food #{food_id}"
        category = "General Food"
        
        # Try int lookup (food_id might be int or string)
        lookup_id = int(food_id) if isinstance(food_id, (int, float)) else food_id
        try:
            lookup_id = int(lookup_id)
        except (ValueError, TypeError):
            pass
            
        if self.food_names and lookup_id in self.food_names:
            name_info = self.food_names[lookup_id]
            food_name = name_info.get('food_desc', food_name)
            category = name_info.get('WWEIA_desc', category)
        
        return {
            "food_id": str(food_id),
            "food_name": food_name,
            "category": category,
            "nutrients": nutrients,
            "health_tags": health_tags
        }
    
    def compute_user_embedding(self, user_profile: Dict[str, Any]) -> torch.Tensor:
        """
        Compute user embedding from profile.
        For new users, we create an embedding based on similar existing users.
        
        Args:
            user_profile: User profile dictionary with health_tags
            
        Returns:
            User embedding tensor
        """
        # Convert health tags to tensor
        health_tags = user_profile.get('health_tags', {})
        
        # Create a pseudo user by averaging similar users' embeddings
        # For simplicity, we use random embedding for new users
        embedding_dim = self.user_embeddings.shape[1] if self.user_embeddings is not None else 128
        
        # Initialize with normalized random embedding
        torch.manual_seed(hash(str(user_profile.get('id', 'default'))) % (2**32))
        user_emb = F.normalize(torch.randn(embedding_dim), dim=0).to(self.device)
        
        return user_emb
    
    def recommend(
        self,
        user_embedding: torch.Tensor,
        k: int = 20,
        exclude_food_ids: Optional[List[str]] = None,
        health_filter: Optional[Dict[str, bool]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate top-k recommendations for a user.
        
        Args:
            user_embedding: User embedding tensor
            k: Number of recommendations
            exclude_food_ids: Food IDs to exclude
            health_filter: Health tags to filter by
            
        Returns:
            List of recommended food items
        """
        # Compute scores
        scores = torch.matmul(user_embedding, self.food_embeddings.t())
        num_foods = len(scores)
        
        # Apply health filter if provided (only for foods within embedding range)
        if health_filter and len(health_filter) > 0:
            for tag, required in health_filter.items():
                if required:
                    tag_col = tag.replace('user_', '')
                    if tag_col in self.food_df.columns:
                        # Only use mask for foods that have embeddings
                        mask = (self.food_df[tag_col] == 1).values[:num_foods]
                        # Boost scores for matching foods
                        scores[mask] *= 1.5
        
        # Get top-k
        top_scores, top_indices = torch.topk(scores, min(k * 2, len(scores)))
        
        recommendations = []
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            if len(recommendations) >= k:
                break
            
            food_info = self.get_food_info(idx)
            
            # Skip if in exclusion list
            if exclude_food_ids and food_info['food_id'] in exclude_food_ids:
                continue
            
            food_info['score'] = score
            food_info['food_idx'] = idx
            recommendations.append(food_info)
        
        return recommendations
    
    def get_food_embedding(self, food_idx: int) -> torch.Tensor:
        """Get embedding for a specific food."""
        return self.food_embeddings[food_idx]
    
    def compute_health_match_score(
        self,
        food_info: Dict[str, Any],
        user_health_tags: Dict[str, bool]
    ) -> Tuple[float, List[str], List[str]]:
        """
        Compute how well a food matches user's health requirements.
        
        Args:
            food_info: Food information dict
            user_health_tags: User's health tags
            
        Returns:
            Tuple of (score, matching_tags, conflicting_tags)
        """
        food_tags = food_info.get('health_tags', {})
        
        matching = []
        conflicting = []
        
        for user_tag, required in user_health_tags.items():
            if not required:
                continue
            
            # Convert user tag to food tag (remove 'user_' prefix)
            food_tag = user_tag.replace('user_', '')
            
            # Check for match
            if food_tag in food_tags and food_tags[food_tag]:
                matching.append(food_tag)
            
            # Check for conflict (e.g., user needs low_sodium, food is high_sodium)
            opposite_tag = food_tag.replace('low_', 'HIGH_').replace('high_', 'low_').replace('HIGH_', 'high_')
            if opposite_tag in food_tags and food_tags[opposite_tag]:
                conflicting.append(opposite_tag)
        
        # Calculate score
        total_tags = len(matching) + len(conflicting)
        score = len(matching) / max(total_tags, 1)
        
        return score, matching, conflicting


# Singleton instance
_model_instance: Optional[RecommendationModel] = None


def get_recommendation_model() -> RecommendationModel:
    """Get or create the singleton recommendation model instance."""
    global _model_instance
    
    if _model_instance is None:
        checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "best_model.pt"
        
        # Fallback paths
        if not checkpoint_path.exists():
            checkpoint_path = Path(__file__).parent.parent.parent / "checkpoints" / "best_model.pt"
        
        _model_instance = RecommendationModel(
            checkpoint_path=str(checkpoint_path),
            device=settings.device
        )
    
    return _model_instance


def reset_model():
    """Reset the model instance (for testing)."""
    global _model_instance
    _model_instance = None
