/**
 * Authentication store using Zustand
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { supabase } from '../lib/supabase';
import { api } from '../lib/api';
import type { UserProfile } from '../lib/api';

interface AuthState {
  user: UserProfile | null;
  accessToken: string | null;
  refreshToken: string | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  isAnonymous: boolean;
  
  // Actions
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, fullName?: string) => Promise<void>;
  loginAnonymously: () => Promise<void>;
  linkEmail: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshAuth: () => Promise<void>;
  updateUser: (data: Partial<UserProfile>) => void;
  checkAuth: () => Promise<void>;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      accessToken: null,
      refreshToken: null,
      isLoading: true,
      isAuthenticated: false,
      isAnonymous: false,

      login: async (email: string, password: string) => {
        set({ isLoading: true });
        try {
          // Use Supabase for auth
          const { data, error } = await supabase.auth.signInWithPassword({
            email,
            password,
          });

          if (error) throw error;

          const accessToken = data.session?.access_token || null;
          const refreshToken = data.session?.refresh_token || null;

          // Set token for API calls
          api.setToken(accessToken);

          // Get user profile
          const profile = await api.getProfile();

          set({
            user: profile,
            accessToken,
            refreshToken,
            isAuthenticated: true,
            isAnonymous: false,
            isLoading: false,
          });
        } catch (error) {
          set({ isLoading: false });
          throw error;
        }
      },

      register: async (email: string, password: string, fullName?: string) => {
        set({ isLoading: true });
        try {
          // Use Supabase for registration
          const { data, error } = await supabase.auth.signUp({
            email,
            password,
            options: {
              data: {
                full_name: fullName,
              },
            },
          });

          if (error) throw error;

          const accessToken = data.session?.access_token || null;
          const refreshToken = data.session?.refresh_token || null;

          // Set token for API calls
          api.setToken(accessToken);

          // Create initial profile if we have a session
          if (accessToken) {
            const profile = await api.getProfile();
            
            set({
              user: profile,
              accessToken,
              refreshToken,
              isAuthenticated: true,
              isAnonymous: false,
              isLoading: false,
            });
          } else {
            // Email confirmation required
            set({ isLoading: false });
          }
        } catch (error) {
          set({ isLoading: false });
          throw error;
        }
      },

      loginAnonymously: async () => {
        set({ isLoading: true });
        try {
          // Use Supabase for anonymous auth
          const { data, error } = await supabase.auth.signInAnonymously();

          if (error) throw error;

          const accessToken = data.session?.access_token || null;
          const refreshToken = data.session?.refresh_token || null;

          // Set token for API calls
          api.setToken(accessToken);

          // Get or create user profile
          const profile = await api.getProfile();

          set({
            user: profile,
            accessToken,
            refreshToken,
            isAuthenticated: true,
            isAnonymous: true,
            isLoading: false,
          });
        } catch (error) {
          set({ isLoading: false });
          throw error;
        }
      },

      linkEmail: async (email: string, password: string) => {
        set({ isLoading: true });
        try {
          // Update user with email and password
          const { data, error } = await supabase.auth.updateUser({
            email,
            password,
          });

          if (error) throw error;

          // Get fresh session
          const { data: sessionData } = await supabase.auth.getSession();
          
          const accessToken = sessionData.session?.access_token || null;
          const refreshToken = sessionData.session?.refresh_token || null;

          // Update token for API calls
          api.setToken(accessToken);

          // Get updated profile
          const profile = await api.getProfile();

          set({
            user: profile,
            accessToken,
            refreshToken,
            isAuthenticated: true,
            isAnonymous: false,
            isLoading: false,
          });
        } catch (error) {
          set({ isLoading: false });
          throw error;
        }
      },

      logout: async () => {
        try {
          await supabase.auth.signOut();
        } catch (error) {
          console.error('Logout error:', error);
        }

        api.setToken(null);
        set({
          user: null,
          accessToken: null,
          refreshToken: null,
          isAuthenticated: false,
          isAnonymous: false,
          isLoading: false,
        });
      },

      refreshAuth: async () => {
        const { refreshToken } = get();
        if (!refreshToken) return;

        try {
          const { data, error } = await supabase.auth.refreshSession({
            refresh_token: refreshToken,
          });

          if (error) throw error;

          const newAccessToken = data.session?.access_token || null;
          const newRefreshToken = data.session?.refresh_token || null;

          api.setToken(newAccessToken);

          set({
            accessToken: newAccessToken,
            refreshToken: newRefreshToken,
          });
        } catch (error) {
          console.error('Token refresh failed:', error);
          get().logout();
        }
      },

      updateUser: (data: Partial<UserProfile>) => {
        const { user } = get();
        if (user) {
          set({ user: { ...user, ...data } });
        }
      },

      checkAuth: async () => {
        set({ isLoading: true });
        try {
          const { data: { session } } = await supabase.auth.getSession();

          if (session) {
            api.setToken(session.access_token);
            const profile = await api.getProfile();

            // Check if user is anonymous (no email)
            const isAnonymous = !session.user.email;

            set({
              user: profile,
              accessToken: session.access_token,
              refreshToken: session.refresh_token || null,
              isAuthenticated: true,
              isAnonymous,
              isLoading: false,
            });
          } else {
            set({
              user: null,
              accessToken: null,
              refreshToken: null,
              isAuthenticated: false,
              isAnonymous: false,
              isLoading: false,
            });
          }
        } catch (error) {
          console.error('Auth check failed:', error);
          set({
            user: null,
            accessToken: null,
            refreshToken: null,
            isAuthenticated: false,
            isAnonymous: false,
            isLoading: false,
          });
        }
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        accessToken: state.accessToken,
        refreshToken: state.refreshToken,
        isAnonymous: state.isAnonymous,
      }),
    }
  )
);
