/**
 * Layout for authentication pages
 */

import type { ReactNode } from 'react';
import { Salad } from 'lucide-react';

interface AuthLayoutProps {
  children: ReactNode;
  title: string;
  subtitle?: string;
}

export function AuthLayout({ children, title, subtitle }: AuthLayoutProps) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-white to-teal-50 flex flex-col justify-center py-12 px-4 sm:px-6 lg:px-8">
      <div className="sm:mx-auto sm:w-full sm:max-w-md">
        {/* Logo */}
        <div className="flex justify-center">
          <div className="flex items-center space-x-2">
            <div className="bg-emerald-600 p-2 rounded-xl">
              <Salad className="h-8 w-8 text-white" />
            </div>
            <span className="text-2xl font-bold text-slate-800">HFRS</span>
          </div>
        </div>
        
        {/* Title */}
        <h2 className="mt-6 text-center text-3xl font-bold tracking-tight text-slate-900">
          {title}
        </h2>
        {subtitle && (
          <p className="mt-2 text-center text-sm text-slate-600">
            {subtitle}
          </p>
        )}
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
        <div className="bg-white py-8 px-4 shadow-xl shadow-slate-200/50 rounded-2xl sm:px-10 border border-slate-100">
          {children}
        </div>
      </div>
      
      {/* Footer */}
      <p className="mt-8 text-center text-xs text-slate-500">
        Health-aware Food Recommendation System
        <br />
        Powered by MOPI-HFRS & AI Agents
      </p>
    </div>
  );
}
