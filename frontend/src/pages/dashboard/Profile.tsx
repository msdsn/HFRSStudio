/**
 * Profile management page
 */

import { useState } from 'react';
import { Save, User, Heart, Utensils, AlertTriangle } from 'lucide-react';
import { DashboardLayout } from '../../components/layout/DashboardLayout';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Input } from '../../components/ui/Input';
import { useAuthStore } from '../../stores/auth';
import { api } from '../../lib/api';

const healthConditions = [
  { id: 'user_low_sodium', label: 'Low Sodium', icon: '🧂' },
  { id: 'user_low_sugar', label: 'Low Sugar', icon: '🍬' },
  { id: 'user_low_calorie', label: 'Low Calorie', icon: '🔥' },
  { id: 'user_high_protein', label: 'High Protein', icon: '💪' },
  { id: 'user_low_cholesterol', label: 'Low Cholesterol', icon: '❤️' },
  { id: 'user_high_fiber', label: 'High Fiber', icon: '🌾' },
  { id: 'user_high_iron', label: 'High Iron', icon: '🩸' },
  { id: 'user_high_calcium', label: 'High Calcium', icon: '🦴' },
];

export function ProfilePage() {
  const { user, updateUser } = useAuthStore();
  
  const [fullName, setFullName] = useState(user?.full_name || '');
  const [age, setAge] = useState(user?.age?.toString() || '');
  const [gender, setGender] = useState(user?.gender || '');
  const [healthTags, setHealthTags] = useState<Record<string, boolean>>(user?.health_tags || {});
  const [isLoading, setIsLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  const toggleHealthTag = (tagId: string) => {
    setHealthTags((prev) => ({
      ...prev,
      [tagId]: !prev[tagId],
    }));
  };

  const handleSave = async () => {
    setIsLoading(true);
    setSuccess(false);

    try {
      const updated = await api.updateProfile({
        full_name: fullName,
        age: parseInt(age) || undefined,
        gender,
        health_tags: healthTags,
      });

      updateUser(updated);
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (error) {
      console.error('Failed to update profile:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <DashboardLayout>
      <div className="max-w-3xl mx-auto space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Profile Settings</h1>
          <p className="text-slate-600">Manage your personal information and health preferences</p>
        </div>

        {success && (
          <div className="bg-emerald-50 border border-emerald-200 text-emerald-700 px-4 py-3 rounded-lg">
            Profile updated successfully!
          </div>
        )}

        {/* Basic Info */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <User className="h-5 w-5 text-slate-400" />
              <span>Basic Information</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Input
              label="Full Name"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              placeholder="John Doe"
            />
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1.5">
                  Gender
                </label>
                <select
                  value={gender}
                  onChange={(e) => setGender(e.target.value)}
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
                value={age}
                onChange={(e) => setAge(e.target.value)}
                placeholder="25"
                min="1"
                max="120"
              />
            </div>

            <div className="pt-2">
              <p className="text-sm text-slate-500">
                Email: <span className="text-slate-700">{user?.email}</span>
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Health Needs */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Heart className="h-5 w-5 text-red-400" />
              <span>Health Needs</span>
            </CardTitle>
            <CardDescription>
              Select the nutritional requirements that apply to you
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {healthConditions.map((condition) => (
                <button
                  key={condition.id}
                  type="button"
                  onClick={() => toggleHealthTag(condition.id)}
                  className={`p-3 rounded-xl border text-center transition-colors ${
                    healthTags[condition.id]
                      ? 'border-emerald-500 bg-emerald-50'
                      : 'border-slate-200 hover:border-slate-300'
                  }`}
                >
                  <span className="text-2xl mb-1 block">{condition.icon}</span>
                  <span className="text-sm font-medium text-slate-700">{condition.label}</span>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Dietary Info */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Utensils className="h-5 w-5 text-orange-400" />
              <span>Dietary Restrictions</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {(user?.dietary_restrictions || []).map((restriction) => (
                <span
                  key={restriction}
                  className="px-3 py-1 bg-orange-50 text-orange-700 rounded-full text-sm"
                >
                  {restriction}
                </span>
              ))}
              {(user?.dietary_restrictions?.length || 0) === 0 && (
                <p className="text-slate-500 text-sm">No dietary restrictions set</p>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Allergies */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <AlertTriangle className="h-5 w-5 text-yellow-500" />
              <span>Allergies</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {(user?.allergies || []).map((allergy) => (
                <span
                  key={allergy}
                  className="px-3 py-1 bg-red-50 text-red-700 rounded-full text-sm"
                >
                  {allergy}
                </span>
              ))}
              {(user?.allergies?.length || 0) === 0 && (
                <p className="text-slate-500 text-sm">No allergies set</p>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Save button */}
        <div className="flex justify-end">
          <Button onClick={handleSave} isLoading={isLoading}>
            <Save className="mr-2 h-4 w-4" />
            Save Changes
          </Button>
        </div>
      </div>
    </DashboardLayout>
  );
}
