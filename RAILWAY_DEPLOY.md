# Railway Deployment Guide

Bu rehber HFRS projesini Railway'e deploy etmek için adım adım talimatlar içerir.

## Deployment Stratejisi

**Tek Container Yaklaşımı:** Frontend ve Backend aynı container'da çalışır.
- Frontend: React (Vite) static files → `/app/static`
- Backend: FastAPI → Port 8000 (Railway'in PORT env var'ı)
- Nginx gerekmez, FastAPI StaticFiles ile serve edilir

## Ön Hazırlık

1. **Railway Hesabı Oluştur:**
   - https://railway.app adresinden hesap oluştur
   - GitHub/GitLab ile bağlan

2. **Railway CLI (Opsiyonel):**
   ```bash
   npm i -g @railway/cli
   railway login
   ```

## Deployment Adımları

### 1. Yeni Project Oluştur

1. Railway dashboard'a git
2. "New Project" → "Deploy from GitHub repo"
3. Repository'yi seç (HFRSStudio)

### 2. Service Ayarları

**Root Directory:** Boş bırak (root kullanılacak)

**Settings** → **Build**:
- **Builder:** Dockerfile
- **Dockerfile Path:** `Dockerfile`

**Settings** → **Deploy**:
- **Start Command:** Dockerfile'dan otomatik alınacak (`uvicorn main:app...`)

### 3. Environment Variables

**Settings** → **Variables** bölümüne şunları ekle:

#### Backend Variables (Runtime)
```
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE=your_supabase_service_role_key
GEMINI_KEY=your_google_gemini_api_key
OPENAI_KEY=your_openai_api_key
DEVICE=cpu
DEBUG=false
PRIMARY_LLM=gemini
FALLBACK_LLM=openai
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2048
```

#### Frontend Variables (Build-time)

**ÖNEMLİ:** Railway'de build ARG olarak geçmek için şu şekilde ekleyin:

**Service Settings** → **Variables** → **Add Variable**:

```
VITE_API_URL=/api
VITE_SUPABASE_URL=your_supabase_project_url
VITE_SUPABASE_KEY=your_supabase_anon_key
```

**Not:** 
- Railway otomatik olarak `PORT` environment variable'ını set eder
- `VITE_API_URL=/api` relative path kullanır (aynı domain)
- Build ARG'ları Railway tarafından otomatik geçilir

### 4. Build Configuration

Railway'de `Dockerfile` kullanıldığı için:
- Frontend build stage otomatik çalışır
- Build sırasında VITE_ değişkenleri embed edilir
- Data/checkpoint dosyaları Google Drive'dan download edilir

### 5. CORS Ayarları

Backend'in `config.py` dosyasında CORS origins'i güncelleyin:

Railway'de environment variable olarak:
```
CORS_ORIGINS=https://your-app.railway.app,http://localhost:5173
```

Ya da `config.py`'de:
```python
cors_origins: list[str] = [
    "http://localhost:5173",
    "http://localhost:8000",
    "https://your-app.railway.app"
]
```

### 6. Deploy

1. **Deploy** butonuna tıkla
2. Build loglarını izle:
   - Frontend build
   - Backend dependencies install
   - PyTorch CPU version install
   - Data download from Google Drive
3. Deploy tamamlandığında URL'yi kopyala (örn: `https://hfrs.railway.app`)

## Post-Deployment

### 1. Supabase Ayarları

1. Supabase dashboard → **Settings** → **API**
2. **Allowed Origins** listesine Railway URL'ini ekle:
   - `https://your-app.railway.app`
3. **Redirect URLs** listesine ekle:
   - `https://your-app.railway.app/**`

### 2. Test Endpoints

Deployment sonrası test edin:

```bash
# Health check
curl https://your-app.railway.app/api/health

# Frontend
curl https://your-app.railway.app/
```

Tarayıcıda:
- Ana sayfa: `https://your-app.railway.app/`
- API docs: `https://your-app.railway.app/docs`

### 3. Test Akışı

1. Frontend URL'ine git
2. Kayıt ol / Giriş yap
3. Onboarding'i tamamla
4. Recommendation'ları test et

## Local Testing (Production Mode)

Railway'e deploy etmeden önce local'de test edin:

```bash
# .env dosyanızı oluşturun
cp .env.example .env
# .env dosyasını doldurun

# Production build
docker compose --profile production up --build

# Test
open http://localhost:8000
```

## Troubleshooting

### Build Hatası

**Frontend build fails:**
- Environment variables doğru mu? (VITE_ prefix)
- `frontend/package.json` dependencies tamam mı?

**Backend build fails:**
- PyTorch installation: CPU version kullanıldığından emin ol
- Google Drive download: Network izni var mı?

**Data files download hatası:**
- Google Drive public access var mı?
- gdown paketi install edildi mi?

### Runtime Hatası

**Model yüklenemiyor:**
- Checkpoint dosyası `/app/checkpoints` altında mı?
- Google Drive'dan düzgün download edildi mi? (Build logs'a bakın)

**Frontend 404:**
- `/app/static` klasörü var mı?
- Frontend build başarılı oldu mu?
- `main.py` StaticFiles mount doğru mu?

**CORS hatası:**
- Backend CORS_ORIGINS'e Railway URL eklendi mi?
- Supabase allowed origins güncellendi mi?

**Environment variables bulunamıyor:**
- Frontend: Build-time variable olarak eklendi mi?
- Backend: Runtime variable olarak eklendi mi?

**API 500 errors:**
- Supabase credentials doğru mu?
- LLM API keys geçerli mi?
- Logs'u kontrol et: Railway dashboard → **Logs**

### VITE Environment Variables

Vite, build time'da environment variable'ları işler:

**Yanlış:** Runtime'da inject etmek
```yaml
environment:
  - VITE_API_URL=/api  # ❌ Çalışmaz
```

**Doğru:** Build ARG olarak geçmek
```dockerfile
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build  # ✅ Build sırasında embed edilir
```

Railway'de bu otomatik yapılır, sadece Variables'a ekleyin.

## Monitoring

1. **Logs:** Railway dashboard → Service → **Logs**
2. **Metrics:** Service → **Metrics** (CPU, Memory, Network)
3. **Health Check:** Backend `/api/health` endpoint'i otomatik check edilir
4. **Deployments:** Deployment history ve rollback

## Cost Optimization

- **CPU-only model:** GPU kullanmıyor, daha ucuz
- **Single container:** İki container yerine tek container (half cost)
- **Data storage:** Google Drive'dan download, Railway disk kullanmıyor
- **Sleep on idle:** Railway free tier'da idle servisler sleep'e gider

## Scaling

Railway'de otomatik scaling yok, manuel:

1. Service → **Settings** → **Resources**
2. RAM/CPU limits ayarla
3. Multiple replicas için paralel deploy

## Development vs Production

**Development (Local):**
```bash
docker compose --profile development up
# Frontend: http://localhost (port 80, nginx)
# Backend: http://localhost:8000
```

**Production (Railway):**
```bash
docker compose --profile production up
# Everything: http://localhost:8000
```

## Production Checklist

- [ ] `.env` dosyası oluşturuldu ve dolduruldu
- [ ] Railway project oluşturuldu
- [ ] Environment variables (backend + frontend) set edildi
- [ ] CORS origins yapılandırıldı (backend)
- [ ] Supabase allowed origins güncellendi
- [ ] Build başarıyla tamamlandı
- [ ] Health check çalışıyor (`/api/health`)
- [ ] Frontend yükleniyor (`/`)
- [ ] Authentication çalışıyor
- [ ] Recommendations generate ediliyor
- [ ] Logs temiz (kritik error yok)
- [ ] Custom domain (opsiyonel) bağlandı

## Custom Domain (Opsiyonel)

1. Railway Service → **Settings** → **Networking**
2. **Custom Domain** → Domain ekle
3. DNS records'u güncelle (Railway talimatları)
4. CORS ve Supabase ayarlarını yeni domain ile güncelle

## Support

- Railway Docs: https://docs.railway.app
- HFRS Issues: GitHub repository issues
- Railway Discord: https://discord.gg/railway
