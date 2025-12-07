/**
 * Agent card component for workflow visualizer
 */

import type { ReactNode } from 'react';
import { Loader2, CheckCircle, AlertCircle, ChevronDown, ChevronUp } from 'lucide-react';
import { cn } from '../../lib/utils';

export type AgentStatus = 'pending' | 'running' | 'completed' | 'error';

interface AgentCardProps {
  name: string;
  emoji: string;
  description: string;
  status: AgentStatus;
  analysis?: string;
  confidence?: number;
  isExpanded?: boolean;
  onToggle?: () => void;
  children?: ReactNode;
}

export function AgentCard({
  name,
  emoji,
  description,
  status,
  analysis,
  confidence,
  isExpanded,
  onToggle,
  children,
}: AgentCardProps) {
  const getStatusIcon = () => {
    switch (status) {
      case 'running':
        return <Loader2 className="h-5 w-5 animate-spin text-blue-500" />;
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-emerald-500" />;
      case 'error':
        return <AlertCircle className="h-5 w-5 text-red-500" />;
      default:
        return <div className="h-5 w-5 rounded-full border-2 border-slate-300" />;
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'running':
        return 'border-blue-300 bg-blue-50';
      case 'completed':
        return 'border-emerald-300 bg-emerald-50';
      case 'error':
        return 'border-red-300 bg-red-50';
      default:
        return 'border-slate-200 bg-slate-50';
    }
  };

  return (
    <div
      className={cn(
        'rounded-xl border-2 transition-all duration-300',
        getStatusColor(),
        status === 'running' && 'ring-2 ring-blue-200 ring-offset-2'
      )}
    >
      {/* Header */}
      <button
        onClick={onToggle}
        className="w-full p-4 flex items-center justify-between text-left"
        disabled={status === 'pending'}
      >
        <div className="flex items-center space-x-3">
          <span className="text-2xl">{emoji}</span>
          <div>
            <h3 className="font-semibold text-slate-900">{name}</h3>
            <p className="text-sm text-slate-500">{description}</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-3">
          {confidence !== undefined && status === 'completed' && (
            <span className="text-sm text-slate-600">
              {Math.round(confidence * 100)}% confidence
            </span>
          )}
          {getStatusIcon()}
          {status !== 'pending' && (
            isExpanded ? (
              <ChevronUp className="h-4 w-4 text-slate-400" />
            ) : (
              <ChevronDown className="h-4 w-4 text-slate-400" />
            )
          )}
        </div>
      </button>

      {/* Expanded content */}
      {isExpanded && status !== 'pending' && (
        <div className="px-4 pb-4 border-t border-slate-200/50 mt-2 pt-3">
          {status === 'running' && (
            <div className="flex items-center space-x-2 text-blue-600">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span className="text-sm">Analyzing...</span>
            </div>
          )}
          
          {status === 'completed' && analysis && (
            <div className="space-y-2">
              <p className="text-sm text-slate-700 whitespace-pre-wrap">
                {analysis.length > 500 ? analysis.slice(0, 500) + '...' : analysis}
              </p>
              {children}
            </div>
          )}
          
          {status === 'error' && (
            <p className="text-sm text-red-600">
              An error occurred during analysis
            </p>
          )}
        </div>
      )}
    </div>
  );
}
