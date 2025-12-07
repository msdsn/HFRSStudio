/**
 * Onboarding Step 3: Dietary Preferences
 */

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { OnboardingLayout } from './OnboardingLayout';
import { Button } from '../../components/ui/Button';

const dietaryOptions = [
  { id: 'vegetarian', label: '🥬 Vegetarian', description: 'No meat' },
  { id: 'vegan', label: '🌱 Vegan', description: 'No animal products' },
  { id: 'halal', label: '🕌 Halal', description: 'Islamic dietary laws' },
  { id: 'kosher', label: '✡️ Kosher', description: 'Jewish dietary laws' },
  { id: 'gluten-free', label: '🌾 Gluten-Free', description: 'No gluten' },
  { id: 'dairy-free', label: '🥛 Dairy-Free', description: 'No dairy products' },
  { id: 'low-carb', label: '🥩 Low-Carb', description: 'Reduced carbohydrates' },
  { id: 'keto', label: '🥑 Keto', description: 'Very low carb, high fat' },
];

const allergyOptions = [
  { id: 'nuts', label: '🥜 Nuts' },
  { id: 'peanuts', label: '🥜 Peanuts' },
  { id: 'shellfish', label: '🦐 Shellfish' },
  { id: 'fish', label: '🐟 Fish' },
  { id: 'eggs', label: '🥚 Eggs' },
  { id: 'milk', label: '🥛 Milk' },
  { id: 'soy', label: '🫘 Soy' },
  { id: 'wheat', label: '🌾 Wheat' },
];

export function PreferencesPage() {
  const navigate = useNavigate();
  
  const [dietaryRestrictions, setDietaryRestrictions] = useState<string[]>([]);
  const [allergies, setAllergies] = useState<string[]>([]);

  const toggleDietary = (id: string) => {
    setDietaryRestrictions((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  const toggleAllergy = (id: string) => {
    setAllergies((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  const handleSubmit = () => {
    // Store in session storage
    sessionStorage.setItem(
      'onboarding_preferences',
      JSON.stringify({ dietaryRestrictions, allergies })
    );
    navigate('/onboarding/complete');
  };

  return (
    <OnboardingLayout
      currentStep={2}
      totalSteps={4}
      stepTitle="Dietary Preferences"
      stepDescription="Tell us about your diet and any allergies"
    >
      <div className="space-y-8">
        {/* Dietary restrictions */}
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-3">
            Dietary restrictions (select all that apply)
          </label>
          <div className="grid grid-cols-2 gap-2">
            {dietaryOptions.map((option) => (
              <button
                key={option.id}
                type="button"
                onClick={() => toggleDietary(option.id)}
                className={`p-3 rounded-lg border text-left transition-colors ${
                  dietaryRestrictions.includes(option.id)
                    ? 'border-emerald-500 bg-emerald-50'
                    : 'border-slate-200 hover:border-slate-300'
                }`}
              >
                <span className="font-medium text-slate-900">{option.label}</span>
                <p className="text-xs text-slate-500">{option.description}</p>
              </button>
            ))}
          </div>
        </div>

        {/* Allergies */}
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-3">
            Food allergies (select all that apply)
          </label>
          <div className="flex flex-wrap gap-2">
            {allergyOptions.map((option) => (
              <button
                key={option.id}
                type="button"
                onClick={() => toggleAllergy(option.id)}
                className={`px-4 py-2 rounded-full border text-sm font-medium transition-colors ${
                  allergies.includes(option.id)
                    ? 'border-red-500 bg-red-50 text-red-700'
                    : 'border-slate-200 hover:border-slate-300 text-slate-700'
                }`}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>

        {/* Navigation */}
        <div className="flex space-x-3 pt-4">
          <Button
            variant="outline"
            onClick={() => navigate('/onboarding/health')}
            className="flex-1"
          >
            Back
          </Button>
          <Button onClick={handleSubmit} className="flex-1">
            Continue
          </Button>
        </div>
      </div>
    </OnboardingLayout>
  );
}
