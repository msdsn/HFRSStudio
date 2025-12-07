/**
 * Main App component with routing
 */

import { useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useAuthStore } from './stores/auth';

// Auth pages
import { LoginPage } from './pages/auth/Login';
import { RegisterPage } from './pages/auth/Register';
import { ForgotPasswordPage } from './pages/auth/ForgotPassword';

// Onboarding pages
import { WelcomePage } from './pages/onboarding/Welcome';
import { HealthProfilePage } from './pages/onboarding/HealthProfile';
import { PreferencesPage } from './pages/onboarding/Preferences';
import { CompletePage } from './pages/onboarding/Complete';

// Dashboard pages
import { DashboardHome } from './pages/dashboard/Home';
import { ProfilePage } from './pages/dashboard/Profile';
import { HistoryPage } from './pages/dashboard/History';

// Recommendations
import { RecommendationsPage } from './pages/recommendations/Main';

// Loading component
function LoadingScreen() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-600 mx-auto"></div>
        <p className="mt-4 text-slate-600">Loading...</p>
      </div>
    </div>
  );
}

// Protected route wrapper
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading, user } = useAuthStore();

  if (isLoading) {
    return <LoadingScreen />;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  // Check if user needs onboarding
  if (user && !user.onboarding_completed) {
    return <Navigate to="/onboarding" replace />;
  }

  return <>{children}</>;
}

// Onboarding route wrapper
function OnboardingRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuthStore();

  if (isLoading) {
    return <LoadingScreen />;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}

// Public route wrapper (redirect if authenticated)
function PublicRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading, user } = useAuthStore();

  if (isLoading) {
    return <LoadingScreen />;
  }

  if (isAuthenticated) {
    if (user && !user.onboarding_completed) {
      return <Navigate to="/onboarding" replace />;
    }
    return <Navigate to="/dashboard" replace />;
  }

  return <>{children}</>;
}

function App() {
  const { checkAuth, isLoading } = useAuthStore();

  useEffect(() => {
    checkAuth();
  }, [checkAuth]);

  if (isLoading) {
    return <LoadingScreen />;
  }

  return (
    <BrowserRouter>
      <Routes>
        {/* Public routes */}
        <Route
          path="/login"
          element={
            <PublicRoute>
              <LoginPage />
            </PublicRoute>
          }
        />
        <Route
          path="/register"
          element={
            <PublicRoute>
              <RegisterPage />
            </PublicRoute>
          }
        />
        <Route
          path="/forgot-password"
          element={
            <PublicRoute>
              <ForgotPasswordPage />
            </PublicRoute>
          }
        />

        {/* Onboarding routes */}
        <Route
          path="/onboarding"
          element={
            <OnboardingRoute>
              <WelcomePage />
            </OnboardingRoute>
          }
        />
        <Route
          path="/onboarding/health"
          element={
            <OnboardingRoute>
              <HealthProfilePage />
            </OnboardingRoute>
          }
        />
        <Route
          path="/onboarding/preferences"
          element={
            <OnboardingRoute>
              <PreferencesPage />
            </OnboardingRoute>
          }
        />
        <Route
          path="/onboarding/complete"
          element={
            <OnboardingRoute>
              <CompletePage />
            </OnboardingRoute>
          }
        />

        {/* Protected routes */}
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <DashboardHome />
            </ProtectedRoute>
          }
        />
        <Route
          path="/recommendations"
          element={
            <ProtectedRoute>
              <RecommendationsPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/profile"
          element={
            <ProtectedRoute>
              <ProfilePage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/history"
          element={
            <ProtectedRoute>
              <HistoryPage />
            </ProtectedRoute>
          }
        />

        {/* Default redirect */}
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
