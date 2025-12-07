/**
 * Onboarding Step 2: Health Profile
 */

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { OnboardingLayout } from './OnboardingLayout';
import { Button } from '../../components/ui/Button';
import { Input } from '../../components/ui/Input';

interface HealthCondition {
  id: string;
  label: string;
  description: string;
}

const healthConditions: HealthCondition[] = [
  { id: 'has_high_blood_pressure', label: 'High Blood Pressure', description: 'Need low sodium foods' },
  { id: 'has_diabetes', label: 'Diabetes', description: 'Need low sugar, low carb foods' },
  { id: 'has_high_cholesterol', label: 'High Cholesterol', description: 'Need low fat foods' },
  { id: 'has_kidney_disease', label: 'Kidney Disease', description: 'Need low phosphorus, potassium' },
  { id: 'has_heart_disease', label: 'Heart Disease', description: 'Need heart-healthy foods' },
  { id: 'is_overweight', label: 'Overweight', description: 'Need low calorie foods' },
  { id: 'is_underweight', label: 'Underweight', description: 'Need high calorie, protein foods' },
  { id: 'has_anemia', label: 'Anemia', description: 'Need iron-rich foods' },
  { id: 'is_pregnant', label: 'Pregnant', description: 'Need prenatal nutrition' },
];

export function HealthProfilePage() {
  const navigate = useNavigate();
  
  const [formData, setFormData] = useState({
    gender: '',
    age: '',
    conditions: {} as Record<string, boolean>,
  });

  const handleConditionToggle = (conditionId: string) => {
    setFormData((prev) => ({
      ...prev,
      conditions: {
        ...prev.conditions,
        [conditionId]: !prev.conditions[conditionId],
      },
    }));
  };

  const handleSubmit = () => {
    // Store in session storage for now
    sessionStorage.setItem('onboarding_health', JSON.stringify(formData));
    navigate('/onboarding/preferences');
  };

  const isValid = formData.gender && formData.age;

  return (
    <OnboardingLayout
      currentStep={1}
      totalSteps={4}
      stepTitle="Your Health Profile"
      stepDescription="Help us understand your health needs"
    >
      <div className="space-y-6">
        {/* Basic info */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1.5">
              Gender
            </label>
            <select
              value={formData.gender}
              onChange={(e) => setFormData({ ...formData, gender: e.target.value })}
              className="w-full h-10 rounded-lg border border-slate-300 bg-white px-3 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
            >
              <option value="">Select...</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="other">Other</option>
            </select>
          </div>
          
          <Input
            label="Age"
            type="number"
            value={formData.age}
            onChange={(e) => setFormData({ ...formData, age: e.target.value })}
            placeholder="25"
            min="1"
            max="120"
          />
        </div>

        {/* Health conditions */}
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-3">
            Do you have any of these health conditions?
          </label>
          <div className="grid gap-2">
            {healthConditions.map((condition) => (
              <button
                key={condition.id}
                type="button"
                onClick={() => handleConditionToggle(condition.id)}
                className={`flex items-center justify-between p-3 rounded-lg border text-left transition-colors ${
                  formData.conditions[condition.id]
                    ? 'border-emerald-500 bg-emerald-50'
                    : 'border-slate-200 hover:border-slate-300'
                }`}
              >
                <div>
                  <span className="font-medium text-slate-900">{condition.label}</span>
                  <p className="text-xs text-slate-500">{condition.description}</p>
                </div>
                <div
                  className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                    formData.conditions[condition.id]
                      ? 'border-emerald-500 bg-emerald-500'
                      : 'border-slate-300'
                  }`}
                >
                  {formData.conditions[condition.id] && (
                    <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  )}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Navigation */}
        <div className="flex space-x-3 pt-4">
          <Button
            variant="outline"
            onClick={() => navigate('/onboarding')}
            className="flex-1"
          >
            Back
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={!isValid}
            className="flex-1"
          >
            Continue
          </Button>
        </div>
      </div>
    </OnboardingLayout>
  );
}
