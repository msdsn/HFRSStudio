/**
 * API client for HFRS backend
 */

const API_URL = import.meta.env.VITE_API_URL || '/api';

// Types
export interface UserProfile {
  id: string;
  email: string;
  full_name?: string;
  health_profile?: HealthProfile;
  preferences?: UserPreferences;
  onboarding_completed: boolean;
  created_at: string;
  updated_at: string;
  // Direct user fields
  age?: number;
  gender?: string;
  health_tags?: Record<string, boolean>;
  dietary_restrictions?: string[];
  allergies?: string[];
}

export interface HealthProfile {
  conditions?: string[];
  dietary_restrictions?: string[];
  allergies?: string[];
  health_goals?: string[];
  age?: number;
  weight?: number;
  height?: number;
  activity_level?: string;
  gender?: string;
  // Health conditions
  has_high_blood_pressure?: boolean;
  has_diabetes?: boolean;
  has_high_cholesterol?: boolean;
  has_kidney_disease?: boolean;
  has_heart_disease?: boolean;
  is_overweight?: boolean;
  is_underweight?: boolean;
  has_anemia?: boolean;
  is_pregnant?: boolean;
}

export interface UserPreferences {
  cuisines?: string[];
  disliked_foods?: string[];
  favorite_foods?: string[];
  meal_types?: string[];
  cooking_time?: string;
  spice_level?: string;
}

export interface FoodExplanation {
  headline?: string;
  description?: string;
  health_benefits?: string[];
  serving_suggestion?: string;
  fun_fact?: string;
}

export interface FoodRecommendation {
  food_id: number;
  food_name: string;
  category: string;
  score: number;
  final_score?: number;
  health_status: 'include' | 'caution' | 'exclude';
  explanation?: FoodExplanation;
  nutrients?: Record<string, number>;
  strengths?: string[];
  concerns?: string[];
}

export interface AgentOutput {
  agent: string;
  status: 'pending' | 'running' | 'complete' | 'error';
  output?: string;
  duration_ms?: number;
  error?: string;
  success?: boolean;
  analysis?: string;
  confidence?: number;
}

export interface RecommendationResponse {
  recommendations: FoodRecommendation[];
  agent_outputs?: Record<string, AgentOutput>;
  workflow_summary?: string;
  request_id?: string;
}

export interface RecommendationLog {
  id: string;
  user_id: string;
  recommendations: FoodRecommendation[];
  agent_outputs?: Record<string, AgentOutput>;
  created_at: string;
}

export interface RecommendationHistoryResponse {
  history: RecommendationLog[];
  total: number;
}

// API Client Class
class ApiClient {
  private token: string | null = null;

  setToken(token: string | null) {
    this.token = token;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${API_URL}${endpoint}`;
    
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...((options.headers as Record<string, string>) || {}),
    };

    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `API Error: ${response.status}`);
    }

    return response.json();
  }

  // Auth endpoints
  async getProfile(): Promise<UserProfile> {
    return this.request<UserProfile>('/users/me');
  }

  async updateProfile(data: Partial<UserProfile>): Promise<UserProfile> {
    return this.request<UserProfile>('/users/me', {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async updateHealthProfile(data: HealthProfile): Promise<UserProfile> {
    return this.request<UserProfile>('/users/me/health-profile', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updatePreferences(data: UserPreferences): Promise<UserProfile> {
    return this.request<UserProfile>('/users/me/preferences', {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async completeOnboarding(): Promise<UserProfile> {
    return this.request<UserProfile>('/users/me/complete-onboarding', {
      method: 'POST',
    });
  }

  // Recommendations endpoints
  async generateRecommendations(params: {
    num_recommendations?: number;
    include_explanations?: boolean;
  }): Promise<RecommendationResponse> {
    const queryParams = new URLSearchParams();
    if (params.num_recommendations) {
      queryParams.set('num_recommendations', params.num_recommendations.toString());
    }
    if (params.include_explanations !== undefined) {
      queryParams.set('include_explanations', params.include_explanations.toString());
    }

    const endpoint = `/recommendations/generate${queryParams.toString() ? `?${queryParams}` : ''}`;
    return this.request<RecommendationResponse>(endpoint, {
      method: 'POST',
    });
  }

  async getRecommendationHistory(limit?: number): Promise<RecommendationHistoryResponse> {
    const queryParams = limit ? `?limit=${limit}` : '';
    return this.request<RecommendationHistoryResponse>(`/recommendations/history${queryParams}`);
  }

  // Health check
  async healthCheck(): Promise<{ status: string; service: string }> {
    return this.request('/health');
  }
}

export const api = new ApiClient();
