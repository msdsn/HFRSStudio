/**
 * Layout for onboarding wizard pages
 */

import type { ReactNode } from 'react';
import { Salad, ChevronRight } from 'lucide-react';

interface OnboardingLayoutProps {
  children: ReactNode;
  currentStep: number;
  totalSteps: number;
  stepTitle: string;
  stepDescription: string;
}

const steps = [
  { name: 'Welcome', icon: '👋' },
  { name: 'Health Profile', icon: '❤️' },
  { name: 'Preferences', icon: '🍽️' },
  { name: 'Complete', icon: '✨' },
];

export function OnboardingLayout({
  children,
  currentStep,
  totalSteps,
  stepTitle,
  stepDescription,
}: OnboardingLayoutProps) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-white to-teal-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-slate-100 sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="bg-emerald-600 p-1.5 rounded-lg">
              <Salad className="h-5 w-5 text-white" />
            </div>
            <span className="font-bold text-slate-800">HFRS</span>
          </div>
          
          {/* Progress indicator */}
          <div className="flex items-center space-x-2">
            {steps.map((step, index) => (
              <div key={step.name} className="flex items-center">
                <div
                  className={`flex items-center justify-center w-8 h-8 rounded-full text-sm font-medium transition-colors ${
                    index < currentStep
                      ? 'bg-emerald-600 text-white'
                      : index === currentStep
                      ? 'bg-emerald-100 text-emerald-700 ring-2 ring-emerald-600'
                      : 'bg-slate-100 text-slate-400'
                  }`}
                >
                  {index < currentStep ? '✓' : step.icon}
                </div>
                {index < steps.length - 1 && (
                  <ChevronRight className={`w-4 h-4 mx-1 ${
                    index < currentStep ? 'text-emerald-600' : 'text-slate-300'
                  }`} />
                )}
              </div>
            ))}
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-2xl mx-auto px-4 py-12">
        {/* Step info */}
        <div className="text-center mb-8">
          <p className="text-sm font-medium text-emerald-600 mb-2">
            Step {currentStep + 1} of {totalSteps}
          </p>
          <h1 className="text-3xl font-bold text-slate-900 mb-2">{stepTitle}</h1>
          <p className="text-slate-600">{stepDescription}</p>
        </div>

        {/* Form content */}
        <div className="bg-white rounded-2xl shadow-xl shadow-slate-200/50 border border-slate-100 p-8">
          {children}
        </div>
      </main>
    </div>
  );
}
