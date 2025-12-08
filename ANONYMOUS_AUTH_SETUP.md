# Anonymous Authentication Setup Guide

This document describes the anonymous sign-in feature implementation and deployment steps.

## Overview

The app now supports anonymous (guest) authentication, allowing users to explore the application without creating an account. Users can later convert their anonymous account to a permanent account by linking an email and password.

## Features

### 1. Anonymous Sign-in
- Users can start using the app immediately without registration
- Large, prominent "🎉 Continue as Guest" button on Login and Register pages
- Anonymous users get full app functionality
- All user data and preferences are saved

### 2. Email Linking
- Anonymous users can link an email and password anytime
- Linking converts the anonymous account to a permanent account
- All existing data and preferences are preserved
- Email linking UI is prominently displayed in Profile page for anonymous users

## Technical Implementation

### Backend Changes

1. **New Environment Variable Required:**
   ```
   SUPABASE_ANON=your_supabase_anon_key
   ```
   This is the Supabase Anonymous/Public key (different from Service key)

2. **New API Endpoints:**
   - `POST /api/auth/anonymous` - Create anonymous session
   - `POST /api/auth/link-email` - Link email to anonymous account (updates both Auth and profiles table)

3. **Files Modified:**
   - `backend/config.py` - Added `supabase_anon` setting
   - `backend/db/supabase.py` - Added `get_supabase_anon_client()` function
   - `backend/api/auth.py` - Added anonymous auth endpoints

### Frontend Changes

1. **Auth Store Enhanced:**
   - Added `isAnonymous` state
   - Added `loginAnonymously()` action
   - Added `linkEmail()` action
   - Anonymous state is persisted across sessions

2. **UI Updates:**
   - Login page: Added guest login button
   - Register page: Added guest login button  
   - Profile page: Added email linking card for anonymous users
   - Input component: Added icon support

3. **Files Modified:**
   - `frontend/src/stores/auth.ts`
   - `frontend/src/pages/auth/Login.tsx`
   - `frontend/src/pages/auth/Register.tsx`
   - `frontend/src/pages/dashboard/Profile.tsx`
   - `frontend/src/components/ui/Input.tsx`

## Deployment Steps

### 1. Supabase Configuration

**CRITICAL:** Enable anonymous sign-ins in Supabase:

1. Go to Supabase Dashboard
2. Navigate to: **Authentication** → **Providers**
3. Find **Anonymous** provider
4. Toggle **"Enable Anonymous sign-ins"** to ON
5. Save changes

### 2. Backend Deployment

Add the new environment variable to your Railway/Docker deployment:

```bash
# Railway
railway variables set SUPABASE_ANON=your_supabase_anon_key

# Docker
# Add to .env file:
SUPABASE_ANON=your_supabase_anon_key
```

**Finding your Supabase Anon Key:**
1. Go to Supabase Dashboard
2. Navigate to: **Settings** → **API**
3. Copy the **anon public** key (NOT the service_role key)

### 3. Frontend Deployment

No changes needed - frontend already uses `VITE_SUPABASE_KEY` which should be the anon key.

### 4. Database (No Changes Required)

The existing database schema supports anonymous users automatically. Supabase handles anonymous user IDs.

## Testing Checklist

After deployment, test the following:

### Anonymous Sign-in Flow
- [ ] Click "Continue as Guest" on Login page
- [ ] Verify redirect to onboarding
- [ ] Complete onboarding as anonymous user
- [ ] Verify recommendations work
- [ ] Check that preferences are saved
- [ ] Refresh page - verify session persists

### Email Linking Flow
- [ ] As anonymous user, go to Profile page
- [ ] Verify yellow "Link Your Account" card is visible
- [ ] Enter email and password
- [ ] Click "Link Email to Account"
- [ ] Verify success message
- [ ] Verify card disappears after linking
- [ ] Check that email is updated in Profile page
- [ ] Logout and login with the linked email
- [ ] Verify all previous data is preserved
- [ ] Verify email is correctly shown in profiles table

### Register Flow (Guest Option)
- [ ] Visit Register page
- [ ] Verify "Continue as Guest" button is present
- [ ] Test guest login from Register page

## Security Considerations

1. **RLS Policies:** Ensure Row Level Security policies allow anonymous users to access their own data
2. **Rate Limiting:** Consider implementing rate limits for anonymous users
3. **Data Retention:** Consider implementing cleanup for abandoned anonymous accounts
4. **Anonymous Limitations:** Anonymous sessions have the same capabilities as authenticated users

## Troubleshooting

### "Anonymous sign-in failed"
- Check that Anonymous provider is enabled in Supabase Dashboard
- Verify `SUPABASE_ANON` environment variable is set correctly
- Check Supabase logs for detailed error messages

### Email linking fails
- Verify the user is actually anonymous (check `isAnonymous` state)
- Check that the email is not already in use
- Ensure password meets minimum requirements (6+ characters)

### Anonymous session not persisting
- Check browser localStorage (should contain auth-storage)
- Verify Supabase client configuration in frontend
- Check that `autoRefreshToken` and `persistSession` are enabled

## Files Reference

### Backend Files
- `backend/config.py` - Configuration with new SUPABASE_ANON setting
- `backend/db/supabase.py` - Supabase client initialization
- `backend/api/auth.py` - Authentication endpoints
- `backend/requirements.txt` - Python dependencies

### Frontend Files
- `frontend/src/stores/auth.ts` - Authentication state management
- `frontend/src/pages/auth/Login.tsx` - Login page with guest option
- `frontend/src/pages/auth/Register.tsx` - Register page with guest option
- `frontend/src/pages/dashboard/Profile.tsx` - Profile with email linking
- `frontend/src/components/ui/Input.tsx` - Enhanced input component
- `frontend/src/lib/supabase.ts` - Supabase client configuration

## Support

For issues or questions:
1. Check Supabase logs in Dashboard → Logs
2. Check browser console for frontend errors
3. Check Railway/Docker logs for backend errors
4. Verify all environment variables are set correctly

