/**
 * Workflow visualizer component showing all AI agents and their progress
 */

import { useState } from 'react';
import { AgentCard } from './AgentCard';
import type { AgentStatus } from './AgentCard';
import type { AgentOutput } from '../../lib/api';

interface AgentConfig {
  id: string;
  name: string;
  emoji: string;
  description: string;
}

const agents: AgentConfig[] = [
  {
    id: 'model',
    name: 'MOPI-HFRS Model',
    emoji: '🧠',
    description: 'Generating initial recommendations using graph neural network',
  },
  {
    id: 'nutritionist',
    name: 'Nutritionist Agent',
    emoji: '🥗',
    description: 'Analyzing nutritional content and health tags',
  },
  {
    id: 'personalizer',
    name: 'Personalizer Agent',
    emoji: '🎯',
    description: 'Matching foods to your personal preferences',
  },
  {
    id: 'health_advisor',
    name: 'Health Advisor Agent',
    emoji: '❤️',
    description: 'Evaluating health compatibility and contraindications',
  },
  {
    id: 'critic',
    name: 'Critic Agent',
    emoji: '🔍',
    description: 'Quality control and final filtering',
  },
  {
    id: 'explainer',
    name: 'Explainer Agent',
    emoji: '💬',
    description: 'Generating user-friendly explanations',
  },
];

interface WorkflowVisualizerProps {
  agentOutputs: Record<string, AgentOutput>;
  currentStep: string;
  isComplete: boolean;
}

export function WorkflowVisualizer({
  agentOutputs,
  currentStep,
  isComplete,
}: WorkflowVisualizerProps) {
  const [expandedAgent, setExpandedAgent] = useState<string | null>(null);

  const getAgentStatus = (agentId: string): AgentStatus => {
    // If complete, all agents are completed
    if (isComplete || currentStep === 'complete') {
      return 'completed';
    }
    
    // Map step names to agent order
    const stepOrder = ['model_prediction', 'nutritionist', 'personalizer', 'health_advisor', 'critic', 'explainer', 'finalize'];
    const currentStepIndex = stepOrder.indexOf(currentStep);
    const agentIndex = agents.findIndex((a) => a.id === agentId);
    
    // Model is special case (index 0)
    if (agentId === 'model') {
      if (currentStep === 'model_prediction' || currentStep === '') return 'running';
      if (currentStepIndex > 0) return 'completed';
      return 'pending';
    }
    
    // Check output first
    const output = agentOutputs[agentId];
    if (output) {
      if (output.error) return 'error';
      if (output.success) return 'completed';
    }
    
    // Determine status based on current step
    if (currentStep === agentId) return 'running';
    if (currentStepIndex > agentIndex) return 'completed';
    if (currentStepIndex === agentIndex) return 'running';
    
    return 'pending';
  };

  const toggleAgent = (agentId: string) => {
    setExpandedAgent(expandedAgent === agentId ? null : agentId);
  };

  // Calculate overall progress
  const completedCount = agents.filter(
    (a) => getAgentStatus(a.id) === 'completed'
  ).length;
  const progress = (completedCount / agents.length) * 100;

  return (
    <div className="space-y-4">
      {/* Progress bar */}
      <div className="bg-slate-100 rounded-full h-2 overflow-hidden">
        <div
          className="bg-emerald-500 h-full transition-all duration-500 ease-out"
          style={{ width: `${progress}%` }}
        />
      </div>
      
      <p className="text-sm text-slate-600 text-center">
        {isComplete
          ? '✨ All agents completed analysis'
          : `${completedCount} of ${agents.length} agents completed`}
      </p>

      {/* Agent cards with connecting lines */}
      <div className="relative">
        {/* Vertical line */}
        <div className="absolute left-6 top-8 bottom-8 w-0.5 bg-slate-200 hidden md:block" />
        
        <div className="space-y-3">
          {agents.map((agent, index) => {
            const status = getAgentStatus(agent.id);
            const output = agentOutputs[agent.id];
            
            return (
              <div key={agent.id} className="relative">
                {/* Connector dot */}
                {index > 0 && (
                  <div
                    className={`absolute left-6 -top-1.5 w-2 h-2 rounded-full transform -translate-x-1/2 hidden md:block ${
                      status === 'completed'
                        ? 'bg-emerald-500'
                        : status === 'running'
                        ? 'bg-blue-500'
                        : 'bg-slate-300'
                    }`}
                  />
                )}
                
                <AgentCard
                  name={agent.name}
                  emoji={agent.emoji}
                  description={agent.description}
                  status={status}
                  analysis={output?.analysis}
                  confidence={output?.confidence}
                  isExpanded={expandedAgent === agent.id}
                  onToggle={() => toggleAgent(agent.id)}
                />
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
