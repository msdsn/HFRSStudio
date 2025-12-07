/**
 * Main recommendations page with workflow visualizer
 */

import { useState, useCallback } from 'react';
import { Sparkles, RefreshCw, Eye, EyeOff, Loader2 } from 'lucide-react';
import { DashboardLayout } from '../../components/layout/DashboardLayout';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { WorkflowVisualizer } from '../../components/workflow/WorkflowVisualizer';
import { FoodCard } from '../../components/food/FoodCard';
import { api } from '../../lib/api';
import type { FoodRecommendation, AgentOutput } from '../../lib/api';

type ViewMode = 'workflow' | 'results';

export function RecommendationsPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [recommendations, setRecommendations] = useState<FoodRecommendation[]>([]);
  const [agentOutputs, setAgentOutputs] = useState<Record<string, AgentOutput>>({});
  const [workflowSummary, setWorkflowSummary] = useState('');
  const [currentStep, setCurrentStep] = useState('');
  const [isComplete, setIsComplete] = useState(false);
  const [, setViewMode] = useState<ViewMode>('workflow');
  const [showWorkflow, setShowWorkflow] = useState(true);
  const [error, setError] = useState('');

  const handleGenerateRecommendations = useCallback(async () => {
    console.log('[HFRS] Starting recommendation generation...');
    setIsLoading(true);
    setError('');
    setRecommendations([]);
    setAgentOutputs({});
    setWorkflowSummary('');
    setCurrentStep('model_prediction');
    setIsComplete(false);
    setViewMode('workflow');

    try {
      // Simulate workflow progress for better UX
      const steps = ['model_prediction', 'nutritionist', 'personalizer', 'health_advisor', 'critic', 'explainer', 'finalize'];
      let stepIndex = 0;
      
      const progressInterval = setInterval(() => {
        if (stepIndex < steps.length - 1) {
          stepIndex++;
          console.log('[HFRS] Progress step:', steps[stepIndex]);
          setCurrentStep(steps[stepIndex]);
        }
      }, 3000); // Update every 3 seconds

      console.log('[HFRS] Calling API...');
      // Call standard API
      const result = await api.generateRecommendations({
        num_recommendations: 10,
        include_explanations: true,
      });
      
      console.log('[HFRS] API Response:', result);
      console.log('[HFRS] Recommendations count:', result?.recommendations?.length);
      console.log('[HFRS] Agent outputs:', result?.agent_outputs);
      console.log('[HFRS] Workflow summary:', result?.workflow_summary);
      
      // Clear interval
      clearInterval(progressInterval);
      
      // Set all results
      if (result?.recommendations) {
        console.log('[HFRS] Setting recommendations...');
        setRecommendations(result.recommendations);
      } else {
        console.log('[HFRS] No recommendations in result!');
      }
      
      if (result?.agent_outputs) {
        console.log('[HFRS] Setting agent outputs...');
        setAgentOutputs(result.agent_outputs);
      }
      
      if (result?.workflow_summary) {
        console.log('[HFRS] Setting workflow summary...');
        setWorkflowSummary(result.workflow_summary);
      }
      
      setCurrentStep('complete');
      setIsComplete(true);
      console.log('[HFRS] Workflow complete, switching to results view...');
      
      // Show results after a brief delay
      setTimeout(() => {
        console.log('[HFRS] Switching viewMode to results');
        setViewMode('results');
      }, 500);
      
    } catch (err) {
      console.error('[HFRS] Recommendation error:', err);
      setError(err instanceof Error ? err.message : 'Failed to generate recommendations');
    } finally {
      console.log('[HFRS] Finally block, setting isLoading=false');
      setIsLoading(false);
    }
  }, []);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-900">AI Food Recommendations</h1>
            <p className="text-slate-600">
              Personalized recommendations powered by 5 AI agents
            </p>
          </div>
          
          <div className="flex items-center space-x-3">
            {isComplete && recommendations.length > 0 && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowWorkflow(!showWorkflow)}
              >
                {showWorkflow ? (
                  <>
                    <EyeOff className="h-4 w-4 mr-2" />
                    Hide Workflow
                  </>
                ) : (
                  <>
                    <Eye className="h-4 w-4 mr-2" />
                    Show Workflow
                  </>
                )}
              </Button>
            )}
            
            <Button
              onClick={handleGenerateRecommendations}
              disabled={isLoading}
              isLoading={isLoading}
            >
              {isLoading ? (
                'Analyzing...'
              ) : recommendations.length > 0 ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Regenerate
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4 mr-2" />
                  Get Recommendations
                </>
              )}
            </Button>
          </div>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
            {error}
          </div>
        )}

        {/* Initial state */}
        {!isLoading && recommendations.length === 0 && !error && (
          <Card>
            <CardContent className="py-12 text-center">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-emerald-100 rounded-full mb-4">
                <Sparkles className="h-8 w-8 text-emerald-600" />
              </div>
              <h3 className="text-lg font-semibold text-slate-900 mb-2">
                Ready to find your perfect foods?
              </h3>
              <p className="text-slate-600 max-w-md mx-auto mb-6">
                Click the button above to start the AI analysis. Our 5 specialized agents 
                will work together to find the healthiest foods for you.
              </p>
              <div className="flex justify-center space-x-4 text-3xl">
                <span title="MOPI-HFRS Model">🧠</span>
                <span title="Nutritionist">🥗</span>
                <span title="Personalizer">🎯</span>
                <span title="Health Advisor">❤️</span>
                <span title="Critic">🔍</span>
                <span title="Explainer">💬</span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Workflow & Results */}
        {(isLoading || recommendations.length > 0) && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Workflow visualizer */}
            {showWorkflow && (
              <div className="lg:col-span-1">
                <Card className="sticky top-4">
                  <CardHeader>
                    <CardTitle className="text-lg">AI Workflow</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <WorkflowVisualizer
                      agentOutputs={agentOutputs}
                      currentStep={currentStep}
                      isComplete={isComplete}
                    />
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Results */}
            <div className={showWorkflow ? 'lg:col-span-2' : 'lg:col-span-3'}>
              {isLoading && recommendations.length === 0 ? (
                <Card>
                  <CardContent className="py-12 text-center">
                    <Loader2 className="h-12 w-12 animate-spin text-emerald-600 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-slate-900 mb-2">
                      AI Agents Working...
                    </h3>
                    <p className="text-slate-600">
                      Watch the workflow panel to see each agent's progress
                    </p>
                  </CardContent>
                </Card>
              ) : (
                <div className="space-y-4">
                  {/* Summary */}
                  {workflowSummary && (
                    <Card className="bg-gradient-to-r from-emerald-50 to-teal-50 border-emerald-200">
                      <CardContent className="py-4">
                        <p className="text-emerald-800 whitespace-pre-wrap">
                          {workflowSummary}
                        </p>
                      </CardContent>
                    </Card>
                  )}

                  {/* Food cards */}
                  <div className="grid gap-4">
                    {recommendations.map((food, index) => (
                      <FoodCard
                        key={food.food_id}
                        food={food}
                        rank={index + 1}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
