/**
 * Dashboard home page
 */

import { useNavigate } from 'react-router-dom';
import { Sparkles, Heart, TrendingUp, Clock, ArrowRight } from 'lucide-react';
import { DashboardLayout } from '../../components/layout/DashboardLayout';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { useAuthStore } from '../../stores/auth';

export function DashboardHome() {
  const navigate = useNavigate();
  const { user } = useAuthStore();

  // Get health summary from user profile
  const healthTags = user?.health_tags || {};
  const activeHealthNeeds = Object.entries(healthTags)
    .filter(([, value]) => value)
    .map(([key]) => key.replace('user_', '').replace(/_/g, ' '));

  return (
    <DashboardLayout>
      <div className="space-y-8">
        {/* Welcome section */}
        <div className="bg-gradient-to-r from-emerald-500 to-teal-500 rounded-2xl p-6 lg:p-8 text-white">
          <h1 className="text-2xl lg:text-3xl font-bold mb-2">
            Welcome back, {user?.full_name?.split(' ')[0] || 'there'}! 👋
          </h1>
          <p className="text-emerald-100 mb-6 max-w-xl">
            Ready to discover healthy foods tailored just for you? Our AI agents are 
            standing by to analyze and recommend the perfect meals.
          </p>
          <Button
            onClick={() => navigate('/recommendations')}
            className="bg-white text-emerald-600 hover:bg-emerald-50"
          >
            <Sparkles className="mr-2 h-4 w-4" />
            Get Recommendations
          </Button>
        </div>

        {/* Stats cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center space-x-4">
                <div className="p-3 bg-emerald-100 rounded-xl">
                  <Heart className="h-6 w-6 text-emerald-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-slate-900">
                    {activeHealthNeeds.length || 0}
                  </p>
                  <p className="text-sm text-slate-500">Health Focus Areas</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center space-x-4">
                <div className="p-3 bg-blue-100 rounded-xl">
                  <TrendingUp className="h-6 w-6 text-blue-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-slate-900">5</p>
                  <p className="text-sm text-slate-500">AI Agents Active</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center space-x-4">
                <div className="p-3 bg-purple-100 rounded-xl">
                  <Clock className="h-6 w-6 text-purple-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-slate-900">~30s</p>
                  <p className="text-sm text-slate-500">Avg. Analysis Time</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Health profile summary */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Your Health Profile</span>
              <Button variant="ghost" size="sm" onClick={() => navigate('/profile')}>
                Edit <ArrowRight className="ml-1 h-4 w-4" />
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {activeHealthNeeds.length > 0 ? (
              <div className="space-y-3">
                <p className="text-sm text-slate-600 mb-3">
                  Based on your profile, we focus on these nutritional aspects:
                </p>
                <div className="flex flex-wrap gap-2">
                  {activeHealthNeeds.map((need) => (
                    <span
                      key={need}
                      className="px-3 py-1 bg-emerald-50 text-emerald-700 rounded-full text-sm font-medium"
                    >
                      {need}
                    </span>
                  ))}
                </div>
              </div>
            ) : (
              <div className="text-center py-4">
                <p className="text-slate-600 mb-3">
                  Complete your health profile to get personalized recommendations
                </p>
                <Button variant="outline" onClick={() => navigate('/profile')}>
                  Complete Profile
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* AI Agents info */}
        <Card>
          <CardHeader>
            <CardTitle>How Our AI Works</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              {[
                { name: 'Nutritionist', emoji: '🥗', desc: 'Analyzes nutrients' },
                { name: 'Health Advisor', emoji: '❤️', desc: 'Checks health fit' },
                { name: 'Personalizer', emoji: '🎯', desc: 'Matches preferences' },
                { name: 'Critic', emoji: '🔍', desc: 'Quality control' },
                { name: 'Explainer', emoji: '💬', desc: 'Creates explanations' },
              ].map((agent) => (
                <div key={agent.name} className="text-center p-4 bg-slate-50 rounded-xl">
                  <div className="text-3xl mb-2">{agent.emoji}</div>
                  <p className="font-medium text-slate-900 text-sm">{agent.name}</p>
                  <p className="text-xs text-slate-500">{agent.desc}</p>
                </div>
              ))}
            </div>
            <p className="text-center text-sm text-slate-500 mt-4">
              All 5 agents work together to provide you with the best recommendations
            </p>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}
