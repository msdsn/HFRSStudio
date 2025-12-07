/**
 * Recommendation history page
 */

import { useState, useEffect } from 'react';
import { Calendar, ChevronRight, Loader2 } from 'lucide-react';
import { DashboardLayout } from '../../components/layout/DashboardLayout';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/Card';
import { api } from '../../lib/api';
import type { RecommendationLog } from '../../lib/api';

export function HistoryPage() {
  const [history, setHistory] = useState<RecommendationLog[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedLog, setSelectedLog] = useState<RecommendationLog | null>(null);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const data = await api.getRecommendationHistory(20);
      setHistory(data.history || []);
    } catch (error) {
      console.error('Failed to load history:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Recommendation History</h1>
          <p className="text-slate-600">View your past food recommendations</p>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-emerald-600" />
          </div>
        ) : history.length === 0 ? (
          <Card>
            <CardContent className="py-12 text-center">
              <Calendar className="h-12 w-12 text-slate-300 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-slate-900 mb-2">No history yet</h3>
              <p className="text-slate-600">
                Your recommendation history will appear here after you get your first recommendations.
              </p>
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-4">
            {history.map((log) => (
              <Card 
                key={log.id}
                className="cursor-pointer hover:shadow-md transition-shadow"
                onClick={() => setSelectedLog(log)}
              >
                <CardContent className="py-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-slate-500 mb-1">
                        {formatDate(log.created_at)}
                      </p>
                      <p className="font-medium text-slate-900">
                        {Array.isArray(log.recommendations) ? log.recommendations.length : 0} recommendations
                      </p>
                      <div className="flex flex-wrap gap-1 mt-2">
                        {Array.isArray(log.recommendations) && log.recommendations.slice(0, 3).map((rec: any, i: number) => (
                          <span
                            key={i}
                            className="px-2 py-0.5 bg-slate-100 text-slate-600 rounded text-xs"
                          >
                            {rec.food_name || 'Unknown'}
                          </span>
                        ))}
                        {Array.isArray(log.recommendations) && log.recommendations.length > 3 && (
                          <span className="px-2 py-0.5 text-slate-500 text-xs">
                            +{log.recommendations.length - 3} more
                          </span>
                        )}
                      </div>
                    </div>
                    <ChevronRight className="h-5 w-5 text-slate-400" />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        {/* Detail modal/drawer could go here */}
        {selectedLog && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <Card className="w-full max-w-2xl max-h-[80vh] overflow-y-auto">
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle>Recommendation Details</CardTitle>
                <button
                  onClick={() => setSelectedLog(null)}
                  className="text-slate-400 hover:text-slate-600"
                >
                  ✕
                </button>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-sm text-slate-500">
                  Generated on {formatDate(selectedLog.created_at)}
                </p>
                
                <div className="space-y-3">
                  {Array.isArray(selectedLog.recommendations) && selectedLog.recommendations.map((rec: any, i: number) => (
                    <div key={i} className="p-3 bg-slate-50 rounded-lg">
                      <p className="font-medium text-slate-900">{rec.food_name || 'Unknown'}</p>
                      <p className="text-sm text-slate-600">{rec.category || 'Uncategorized'}</p>
                      {rec.explanation?.headline && (
                        <p className="text-sm text-emerald-600 mt-1">{rec.explanation.headline}</p>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
