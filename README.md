# HFRS - Health-aware Food Recommendation System

AI-powered personalized food recommendation system based on the MOPI-HFRS paper, featuring a multi-agent LangGraph workflow for intelligent analysis and explanation.

## Features

- **MOPI-HFRS Model**: Graph neural network for personalized food recommendations
- **5 AI Agents**: Collaborative multi-agent system for comprehensive analysis
  - 🥗 Nutritionist Agent: Nutritional content analysis
  - 🎯 Personalizer Agent: Personal preference matching
  - ❤️ Health Advisor Agent: Health compatibility evaluation
  - 🔍 Critic Agent: Quality control and filtering
  - 💬 Explainer Agent: User-friendly explanations
- **LangGraph Workflow**: State machine orchestration of agents
- **Real-time Streaming**: Live updates as agents complete analysis
- **Supabase Auth**: Secure user authentication and data storage
- **Modern UI**: React + Tailwind CSS with workflow visualizer

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend (React + Tailwind)                  │
│  Auth → Onboarding → Dashboard → Recommendations + Visualizer   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                          │
│  Auth API │ User API │ Recommendation API (Model + Agents)      │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌──────────────────┐ ┌────────────────┐ ┌─────────────────────────┐
│   MOPI-HFRS      │ │   Supabase     │ │    LangGraph Workflow   │
│   (PyTorch)      │ │   (Auth+DB)    │ │    (5 AI Agents)        │
└──────────────────┘ └────────────────┘ └─────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+
- Supabase account
- Gemini API key (primary) and/or OpenAI API key (fallback)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run server
uvicorn main:app --reload
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.example .env.local
# Edit .env.local with your Supabase keys

# Run development server
npm run dev
```

### Supabase Setup

Run these SQL commands in your Supabase SQL editor:

```sql
-- profiles table
create table profiles (
  id uuid references auth.users primary key,
  created_at timestamp default now(),
  updated_at timestamp default now(),
  email text,
  full_name text,
  gender text,
  age integer,
  race text,
  education text,
  household_income integer,
  health_tags jsonb default '{}',
  dietary_restrictions text[] default '{}',
  allergies text[] default '{}',
  cuisine_preferences text[] default '{}',
  onboarding_completed boolean default false
);

-- food_history table
create table food_history (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references profiles(id),
  food_id text,
  food_name text,
  rating integer,
  created_at timestamp default now()
);

-- recommendations_log table
create table recommendations_log (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references profiles(id),
  recommendations jsonb,
  agent_outputs jsonb,
  created_at timestamp default now()
);

-- Enable RLS
alter table profiles enable row level security;
alter table food_history enable row level security;
alter table recommendations_log enable row level security;

-- RLS policies
create policy "Users can read own profile" on profiles for select using (auth.uid() = id);
create policy "Users can update own profile" on profiles for update using (auth.uid() = id);
create policy "Users can insert own profile" on profiles for insert with check (auth.uid() = id);
```

## Model Checkpoint

Place your trained model checkpoint at `backend/checkpoints/best_model.pt`.

The checkpoint should contain:
- `model_state_dict`: Model weights
- `num_users`: Number of users
- `num_foods`: Number of foods
- `user_embeddings` (optional): Pre-computed user embeddings
- `food_embeddings` (optional): Pre-computed food embeddings

## Deployment

### Railway Deployment

#### Prerequisites
1. Railway account (https://railway.app)
2. Railway CLI installed (`npm i -g @railway/cli`)
3. Git repository pushed to GitHub/GitLab

#### Backend Deployment

1. **Create Railway Project:**
   ```bash
   railway login
   railway init
   ```

2. **Set Environment Variables:**
   ```bash
   railway variables set SUPABASE_URL=your_supabase_url
   railway variables set SUPABASE_SERVICE=your_supabase_service_key
   railway variables set GEMINI_KEY=your_gemini_api_key
   railway variables set OPENAI_KEY=your_openai_api_key
   railway variables set DEVICE=cpu
   railway variables set DEBUG=false
   ```

3. **Deploy Backend:**
   - In Railway dashboard, create a new service
   - Connect your GitHub repository
   - Set root directory to `backend/`
   - Set Dockerfile path to `backend/Dockerfile`
   - Railway will automatically detect and deploy

4. **Upload Data Files:**
   - Use Railway's volume feature or upload `data/` and `checkpoints/` directories
   - Or use Railway CLI:
     ```bash
     railway up --service backend
     ```

#### Frontend Deployment

1. **Build Environment Variables:**
   ```bash
   railway variables set VITE_API_URL=https://your-backend.railway.app/api
   railway variables set VITE_SUPABASE_URL=your_supabase_url
   railway variables set VITE_SUPABASE_KEY=your_supabase_anon_key
   ```

2. **Deploy Frontend:**
   - Create a new service in Railway
   - Set root directory to `frontend/`
   - Set Dockerfile path to `frontend/Dockerfile`
   - Railway will build and deploy

#### Alternative: Single Service Deployment

You can also deploy both frontend and backend together:

1. Create a single Railway service
2. Use the backend Dockerfile (it serves frontend static files)
3. Build frontend first, then copy to backend:
   ```dockerfile
   # In backend/Dockerfile, add:
   COPY ../frontend/dist ./static
   ```

### Docker Compose (Local Development)

```bash
# Copy .env.example to .env and fill in values
cp .env.example .env

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Manual Docker Build

**Backend:**
```bash
cd backend
docker build -t hfrs-backend .
docker run -p 8000:8000 --env-file .env hfrs-backend
```

**Frontend:**
```bash
cd frontend
docker build -t hfrs-frontend .
docker run -p 80:80 hfrs-frontend
```

## Project Structure

```
HFRSStudio/
├── backend/
│   ├── Dockerfile            # Backend container
│   ├── main.py              # FastAPI app
│   ├── config.py            # Settings
│   ├── api/                  # API endpoints
│   ├── agents/               # AI agents
│   ├── models/               # ML inference
│   ├── workflows/            # LangGraph
│   ├── db/                   # Database
│   └── utils/                # Utilities
├── frontend/
│   ├── Dockerfile            # Frontend container
│   ├── nginx.conf            # Nginx config
│   ├── src/
│   │   ├── components/       # UI components
│   │   ├── pages/            # Page components
│   │   ├── stores/           # Zustand stores
│   │   └── lib/              # Utilities
│   └── ...
├── docker-compose.yml        # Local development
├── railway.json              # Railway config
├── models/                   # MOPI-HFRS model code
├── data/                     # Data loaders
└── utils/                    # Training utilities
```

## References

- [MOPI-HFRS Paper](https://doi.org/10.1145/3690624.3709382) - KDD '25
- [NHANES Dataset](https://wwwn.cdc.gov/nchs/nhanes/default.aspx)

## License

MIT
