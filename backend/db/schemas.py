"""
Pydantic schemas for database models and API requests/responses.
"""

from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime


# Health tag names matching NHANES format
HEALTH_TAG_NAMES = [
    "low_calorie", "high_calorie",
    "low_carb", "high_carb", 
    "low_protein", "high_protein",
    "low_saturated_fat", "high_saturated_fat",
    "low_sugar", "high_sugar",
    "low_cholesterol", "high_cholesterol",
    "low_fiber", "high_fiber",
    "low_sodium", "high_sodium",
    "low_potassium", "high_potassium",
    "low_phosphorus", "high_phosphorus",
    "low_iron", "high_iron",
    "low_calcium", "high_calcium",
    "low_folic_acid", "high_folic_acid",
    "low_vitamin_c", "high_vitamin_c",
    "low_vitamin_d", "high_vitamin_d",
    "low_vitamin_b12", "high_vitamin_b12"
]


class UserProfile(BaseModel):
    """User profile from database."""
    id: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    
    # Demographics (NHANES style)
    gender: Optional[str] = None
    age: Optional[int] = None
    race: Optional[str] = None
    education: Optional[str] = None
    household_income: Optional[int] = None
    
    # Health tags (NHANES format)
    health_tags: Dict[str, bool] = {}
    
    # Preferences
    dietary_restrictions: List[str] = []
    allergies: List[str] = []
    cuisine_preferences: List[str] = []
    
    # Status
    onboarding_completed: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class HealthProfile(BaseModel):
    """Health profile for onboarding."""
    # Demographics
    gender: str  # male, female
    age: int
    race: Optional[str] = None
    education: Optional[str] = None
    household_income: Optional[int] = None
    
    # Health conditions (will be converted to health tags)
    has_high_blood_pressure: bool = False
    has_diabetes: bool = False
    has_high_cholesterol: bool = False
    has_kidney_disease: bool = False
    has_heart_disease: bool = False
    is_overweight: bool = False
    is_underweight: bool = False
    has_anemia: bool = False
    is_pregnant: bool = False
    
    # Dietary preferences
    dietary_restrictions: List[str] = []  # vegetarian, vegan, halal, kosher, etc.
    allergies: List[str] = []  # nuts, dairy, gluten, shellfish, etc.
    
    def to_health_tags(self) -> Dict[str, bool]:
        """Convert health conditions to NHANES-style health tags."""
        tags = {}
        
        # Blood pressure -> low sodium
        if self.has_high_blood_pressure:
            tags["user_low_sodium"] = True
        
        # Diabetes -> low sugar, low carb
        if self.has_diabetes:
            tags["user_low_sugar"] = True
            tags["user_low_carb"] = True
        
        # High cholesterol -> low cholesterol, low saturated fat
        if self.has_high_cholesterol:
            tags["user_low_cholesterol"] = True
            tags["user_low_saturated_fat"] = True
        
        # Kidney disease -> low phosphorus, low potassium, low sodium
        if self.has_kidney_disease:
            tags["user_low_phosphorus"] = True
            tags["user_low_potassium"] = True
            tags["user_low_sodium"] = True
        
        # Heart disease -> low sodium, low saturated fat, low cholesterol
        if self.has_heart_disease:
            tags["user_low_sodium"] = True
            tags["user_low_saturated_fat"] = True
            tags["user_low_cholesterol"] = True
        
        # Overweight -> low calorie
        if self.is_overweight:
            tags["user_low_calorie"] = True
        
        # Underweight -> high calorie, high protein
        if self.is_underweight:
            tags["user_high_calorie"] = True
            tags["user_high_protein"] = True
        
        # Anemia -> high iron, high vitamin b12, high folic acid
        if self.has_anemia:
            tags["user_high_iron"] = True
            tags["user_high_vitamin_b12"] = True
            tags["user_high_folic_acid"] = True
        
        # Pregnancy -> high folic acid, high iron, high calcium
        if self.is_pregnant:
            tags["user_high_folic_acid"] = True
            tags["user_high_iron"] = True
            tags["user_high_calcium"] = True
        
        return tags


class UpdateProfileRequest(BaseModel):
    """Request to update user profile."""
    full_name: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    race: Optional[str] = None
    education: Optional[str] = None
    household_income: Optional[int] = None
    health_tags: Optional[Dict[str, bool]] = None
    dietary_restrictions: Optional[List[str]] = None
    allergies: Optional[List[str]] = None
    cuisine_preferences: Optional[List[str]] = None


class FoodItem(BaseModel):
    """Food item from the dataset."""
    food_id: str
    food_name: str
    category: str
    description: Optional[str] = None
    ingredients: List[str] = []
    
    # Nutrients
    calories: float = 0
    protein: float = 0
    carbs: float = 0
    sugar: float = 0
    fiber: float = 0
    saturated_fat: float = 0
    cholesterol: float = 0
    sodium: float = 0
    calcium: float = 0
    phosphorus: float = 0
    potassium: float = 0
    iron: float = 0
    folic_acid: float = 0
    vitamin_c: float = 0
    vitamin_d: float = 0
    vitamin_b12: float = 0
    
    # Health tags
    health_tags: Dict[str, bool] = {}


class FoodHistoryEntry(BaseModel):
    """Food history entry."""
    id: str
    user_id: str
    food_id: str
    food_name: str
    rating: Optional[int] = None
    created_at: datetime


class RecommendationLog(BaseModel):
    """Recommendation log entry."""
    id: str
    user_id: str
    recommendations: List[Dict[str, Any]]
    agent_outputs: Dict[str, Any]
    created_at: datetime
