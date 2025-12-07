/**
 * Onboarding Step 1: Welcome
 */

import { useNavigate } from 'react-router-dom';
import { Sparkles, Heart, Brain, Utensils } from 'lucide-react';
import { OnboardingLayout } from './OnboardingLayout';
import { Button } from '../../components/ui/Button';

const features = [
  {
    icon: Heart,
    title: 'Personalized Health',
    description: 'Recommendations tailored to your unique health profile',
  },
  {
    icon: Brain,
    title: 'AI-Powered Analysis',
    description: '5 specialized AI agents analyze every recommendation',
  },
  {
    icon: Utensils,
    title: 'Smart Suggestions',
    description: 'Discover foods that match your taste and health needs',
  },
];

export function WelcomePage() {
  const navigate = useNavigate();

  return (
    <OnboardingLayout
      currentStep={0}
      totalSteps={4}
      stepTitle="Welcome to HFRS"
      stepDescription="Your personal AI-powered nutrition assistant"
    >
      <div className="space-y-8">
        {/* Hero section */}
        <div className="text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-emerald-100 rounded-2xl mb-4">
            <Sparkles className="w-8 h-8 text-emerald-600" />
          </div>
          <p className="text-slate-600 max-w-md mx-auto">
            We'll help you discover delicious foods that are perfect for your health goals.
            Let's set up your profile in just a few steps.
          </p>
        </div>

        {/* Features */}
        <div className="grid gap-4">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="flex items-start space-x-4 p-4 rounded-xl bg-slate-50 border border-slate-100"
            >
              <div className="flex-shrink-0 w-10 h-10 bg-emerald-100 rounded-lg flex items-center justify-center">
                <feature.icon className="w-5 h-5 text-emerald-600" />
              </div>
              <div>
                <h3 className="font-medium text-slate-900">{feature.title}</h3>
                <p className="text-sm text-slate-500">{feature.description}</p>
              </div>
            </div>
          ))}
        </div>

        {/* CTA */}
        <div className="pt-4">
          <Button
            onClick={() => navigate('/onboarding/health')}
            className="w-full"
            size="lg"
          >
            Get Started
          </Button>
          <p className="text-center text-xs text-slate-500 mt-3">
            Takes about 2 minutes to complete
          </p>
        </div>
      </div>
    </OnboardingLayout>
  );
}
