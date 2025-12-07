/**
 * Food recommendation card component
 */

import { useState } from 'react';
import { ChevronDown, ChevronUp, Heart, AlertTriangle, Sparkles } from 'lucide-react';
import type { FoodRecommendation } from '../../lib/api';
import { cn } from '../../lib/utils';

interface FoodCardProps {
  food: FoodRecommendation;
  rank: number;
}

export function FoodCard({ food, rank }: FoodCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const getHealthStatusColor = () => {
    switch (food.health_status) {
      case 'include':
        return 'bg-emerald-100 text-emerald-700';
      case 'caution':
        return 'bg-yellow-100 text-yellow-700';
      case 'exclude':
        return 'bg-red-100 text-red-700';
      default:
        return 'bg-slate-100 text-slate-700';
    }
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden hover:shadow-lg transition-shadow">
      {/* Header */}
      <div className="p-4">
        <div className="flex items-start justify-between">
          <div className="flex items-start space-x-3">
            {/* Rank badge */}
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-emerald-600 text-white flex items-center justify-center font-bold text-sm">
              {rank}
            </div>
            
            <div className="flex-1 min-w-0">
              <h3 className="font-semibold text-slate-900 truncate">
                {food.food_name}
              </h3>
              <p className="text-sm text-slate-500">{food.category}</p>
            </div>
          </div>

          {/* Score */}
          <div className="text-right">
            <div className="flex items-center space-x-1">
              <Sparkles className="h-4 w-4 text-amber-500" />
              <span className="font-semibold text-slate-900">
                {((food.final_score || food.score) * 100).toFixed(0)}%
              </span>
            </div>
            <span className="text-xs text-slate-500">match</span>
          </div>
        </div>

        {/* Headline explanation */}
        {food.explanation?.headline && (
          <p className="mt-3 text-sm text-emerald-700 bg-emerald-50 px-3 py-2 rounded-lg">
            {food.explanation.headline}
          </p>
        )}

        {/* Health status */}
        <div className="mt-3 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            {food.health_status === 'include' && (
              <span className={cn('px-2 py-1 rounded-full text-xs font-medium flex items-center', getHealthStatusColor())}>
                <Heart className="h-3 w-3 mr-1" />
                Healthy Choice
              </span>
            )}
            {food.health_status === 'caution' && (
              <span className={cn('px-2 py-1 rounded-full text-xs font-medium flex items-center', getHealthStatusColor())}>
                <AlertTriangle className="h-3 w-3 mr-1" />
                Use Caution
              </span>
            )}
          </div>

          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="text-sm text-slate-600 hover:text-slate-900 flex items-center"
          >
            {isExpanded ? 'Less' : 'More'}
            {isExpanded ? (
              <ChevronUp className="h-4 w-4 ml-1" />
            ) : (
              <ChevronDown className="h-4 w-4 ml-1" />
            )}
          </button>
        </div>
      </div>

      {/* Expanded content */}
      {isExpanded && (
        <div className="px-4 pb-4 border-t border-slate-100 pt-4 space-y-4">
          {/* Explanation */}
          {food.explanation?.description && (
            <div>
              <h4 className="text-sm font-medium text-slate-900 mb-1">Why this food?</h4>
              <p className="text-sm text-slate-600">{food.explanation.description}</p>
            </div>
          )}

          {/* Health benefits */}
          {food.explanation?.health_benefits && food.explanation.health_benefits.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-slate-900 mb-1">Health Benefits</h4>
              <ul className="text-sm text-slate-600 space-y-1">
                {food.explanation.health_benefits.map((benefit: string, i: number) => (
                  <li key={i} className="flex items-start">
                    <span className="text-emerald-500 mr-2">✓</span>
                    {benefit}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Strengths & Concerns */}
          <div className="grid grid-cols-2 gap-4">
            {food.strengths && food.strengths.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-emerald-700 mb-1">Strengths</h4>
                <ul className="text-xs text-slate-600 space-y-0.5">
                  {food.strengths.slice(0, 3).map((s: string, i: number) => (
                    <li key={i}>+ {s}</li>
                  ))}
                </ul>
              </div>
            )}
            
            {food.concerns && food.concerns.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-orange-700 mb-1">Concerns</h4>
                <ul className="text-xs text-slate-600 space-y-0.5">
                  {food.concerns.slice(0, 3).map((c: string, i: number) => (
                    <li key={i}>• {c}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Serving suggestion */}
          {food.explanation?.serving_suggestion && (
            <div className="bg-slate-50 p-3 rounded-lg">
              <h4 className="text-sm font-medium text-slate-900 mb-1">💡 Serving Tip</h4>
              <p className="text-sm text-slate-600">{food.explanation.serving_suggestion}</p>
            </div>
          )}

          {/* Fun fact */}
          {food.explanation?.fun_fact && (
            <p className="text-xs text-slate-500 italic">
              🎲 {food.explanation.fun_fact}
            </p>
          )}

          {/* Nutrients preview */}
          {food.nutrients && Object.keys(food.nutrients).length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-slate-900 mb-2">Key Nutrients</h4>
              <div className="flex flex-wrap gap-2">
                {Object.entries(food.nutrients).slice(0, 6).map(([key, value]: [string, number]) => (
                  <span
                    key={key}
                    className="px-2 py-1 bg-slate-100 text-slate-700 rounded text-xs"
                  >
                    {key}: {value.toFixed(1)}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
