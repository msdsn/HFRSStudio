/**
 * Onboarding Step 4: Complete
 */

import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { CheckCircle, Loader2 } from 'lucide-react';
import { OnboardingLayout } from './OnboardingLayout';
import { Button } from '../../components/ui/Button';
import { useAuthStore } from '../../stores/auth';
import { api } from '../../lib/api';
import type { HealthProfile } from '../../lib/api';

export function CompletePage() {
  const navigate = useNavigate();
  const { updateUser } = useAuthStore();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    // Auto-submit on mount
    handleComplete();
  }, []);

  const handleComplete = async () => {
    setIsSubmitting(true);
    setError('');

    try {
      // Get stored data
      const healthData = JSON.parse(sessionStorage.getItem('onboarding_health') || '{}');
      const prefsData = JSON.parse(sessionStorage.getItem('onboarding_preferences') || '{}');

      // Build health profile
      const healthProfile: HealthProfile = {
        gender: healthData.gender || 'other',
        age: parseInt(healthData.age) || 30,
        has_high_blood_pressure: healthData.conditions?.has_high_blood_pressure || false,
        has_diabetes: healthData.conditions?.has_diabetes || false,
        has_high_cholesterol: healthData.conditions?.has_high_cholesterol || false,
        has_kidney_disease: healthData.conditions?.has_kidney_disease || false,
        has_heart_disease: healthData.conditions?.has_heart_disease || false,
        is_overweight: healthData.conditions?.is_overweight || false,
        is_underweight: healthData.conditions?.is_underweight || false,
        has_anemia: healthData.conditions?.has_anemia || false,
        is_pregnant: healthData.conditions?.is_pregnant || false,
        dietary_restrictions: prefsData.dietaryRestrictions || [],
        allergies: prefsData.allergies || [],
      };

      // Submit to API
      const updatedProfile = await api.updateHealthProfile(healthProfile);
      
      // Update local state
      updateUser(updatedProfile);

      // Clear session storage
      sessionStorage.removeItem('onboarding_health');
      sessionStorage.removeItem('onboarding_preferences');

      setIsComplete(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save profile');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <OnboardingLayout
      currentStep={3}
      totalSteps={4}
      stepTitle={isComplete ? "You're all set!" : "Setting up your profile"}
      stepDescription={isComplete ? "Your personalized recommendations await" : "Please wait..."}
    >
      <div className="text-center py-8">
        {isSubmitting ? (
          <div className="space-y-4">
            <Loader2 className="w-16 h-16 text-emerald-600 animate-spin mx-auto" />
            <p className="text-slate-600">Creating your personalized profile...</p>
          </div>
        ) : error ? (
          <div className="space-y-4">
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
              {error}
            </div>
            <Button onClick={handleComplete}>Try Again</Button>
          </div>
        ) : isComplete ? (
          <div className="space-y-6">
            <div className="inline-flex items-center justify-center w-20 h-20 bg-emerald-100 rounded-full">
              <CheckCircle className="w-10 h-10 text-emerald-600" />
            </div>
            
            <div>
              <h3 className="text-xl font-semibold text-slate-900 mb-2">
                Profile Created Successfully!
              </h3>
              <p className="text-slate-600 max-w-sm mx-auto">
                Our AI agents are ready to provide you with personalized,
                health-aware food recommendations.
              </p>
            </div>

            <div className="bg-emerald-50 rounded-xl p-4 text-left">
              <h4 className="font-medium text-emerald-900 mb-2">What happens next?</h4>
              <ul className="text-sm text-emerald-700 space-y-1">
                <li>✓ 5 AI agents analyze your profile</li>
                <li>✓ Personalized recommendations generated</li>
                <li>✓ Health-aware explanations provided</li>
                <li>✓ Continuous learning from your feedback</li>
              </ul>
            </div>

            <Button
              onClick={() => navigate('/dashboard')}
              size="lg"
              className="w-full"
            >
              Go to Dashboard
            </Button>
          </div>
        ) : null}
      </div>
    </OnboardingLayout>
  );
}
