# Railway Deployment Guide

Bu rehber HFRS projesini Railway'e deploy etmek için adım adım talimatlar içerir.

## Ön Hazırlık

1. **Railway Hesabı Oluştur:**
   - https://railway.app adresinden hesap oluştur
   - GitHub/GitLab ile bağlan

2. **Railway CLI (Opsiyonel):**
   ```bash
   npm i -g @railway/cli
   railway login
   ```

## Backend Deployment

### 1. Yeni Service Oluştur

1. Railway dashboard'a git
2. "New Project" → "Deploy from GitHub repo"
3. Repository'yi seç
4. Yeni bir service oluştur: "Backend"

### 2. Service Ayarları

1. **Settings** → **Root Directory:** `backend`
2. **Settings** → **Dockerfile Path:** `backend/Dockerfile`
3. **Settings** → **Start Command:** (boş bırak, Dockerfile'dan alınacak)

### 3. Environment Variables

**Settings** → **Variables** bölümüne şunları ekle:

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

**Not:** Railway otomatik olarak `PORT` environment variable'ını set eder.

### 4. Data Files Upload

Model checkpoint ve data dosyalarını yüklemek için iki seçenek:

**Seçenek 1: Railway Volumes (Önerilen)**
1. Service → **Volumes** → **Add Volume**
2. Mount path: `/app/data` ve `/app/checkpoints`
3. Dosyaları upload et

**Seçenek 2: Git Repository**
- `backend/data/` ve `backend/checkpoints/` klasörlerini commit et
- Railway build sırasında kopyalanacak

### 5. Deploy

1. **Deployments** → **Deploy** butonuna tıkla
2. Build loglarını izle
3. Deploy tamamlandığında URL'yi kopyala (örn: `https://hfrs-backend.railway.app`)

## Frontend Deployment

### 1. Yeni Service Oluştur

1. Aynı project içinde **New Service** → **Deploy from GitHub repo**
2. Aynı repository'yi seç
3. Service adı: "Frontend"

### 2. Service Ayarları

1. **Settings** → **Root Directory:** `frontend`
2. **Settings** → **Dockerfile Path:** `frontend/Dockerfile`

### 3. Environment Variables

**Build-time variables** (Vite için):

```
VITE_API_URL=https://your-backend-service.railway.app/api
VITE_SUPABASE_URL=your_supabase_project_url
VITE_SUPABASE_KEY=your_supabase_anon_key
```

**Not:** Backend URL'ini backend service'in Railway URL'i ile değiştir.

### 4. Deploy

1. **Deploy** butonuna tıkla
2. Build tamamlandığında frontend URL'ini al

### 5. Nginx Config Güncelleme (Opsiyonel)

Eğer frontend'den backend'e proxy yapmak istiyorsan:

1. `frontend/nginx.conf` dosyasını düzenle
2. `proxy_pass http://backend:8000;` satırını backend'in Railway URL'i ile değiştir
3. Ya da environment variable kullan:

```nginx
location /api {
    proxy_pass ${BACKEND_URL};
    ...
}
```

## Post-Deployment

### 1. CORS Ayarları

Backend'in `config.py` dosyasında CORS origins'e frontend URL'ini ekle:

```python
cors_origins: list[str] = [
    "http://localhost:5173",
    "https://your-frontend.railway.app"
]
```

Ya da environment variable olarak:

```
CORS_ORIGINS=https://your-frontend.railway.app,http://localhost:5173
```

### 2. Supabase Ayarları

1. Supabase dashboard → **Settings** → **API**
2. **Allowed Origins** listesine frontend URL'ini ekle
3. **Redirect URLs** listesine frontend URL'ini ekle

### 3. Test

1. Frontend URL'ine git
2. Kayıt ol / Giriş yap
3. Onboarding'i tamamla
4. Recommendation'ları test et

## Troubleshooting

### Backend Build Hatası

- **Torch installation hatası:** CPU version kullanıldığından emin ol
- **Data files bulunamadı:** Volume mount'ları kontrol et
- **Port hatası:** Railway otomatik PORT set eder, manuel ayarlama yapma

### Frontend Build Hatası

- **Environment variables:** Build-time variables olduğundan emin ol (VITE_ prefix)
- **API connection:** Backend URL'inin doğru olduğundan emin

### Runtime Hataları

- **Model yüklenemiyor:** Checkpoint dosyasının doğru yerde olduğundan emin
- **LLM API hatası:** API key'lerin doğru olduğundan emin
- **Database hatası:** Supabase credentials'ları kontrol et

## Monitoring

1. **Logs:** Railway dashboard → Service → **Logs**
2. **Metrics:** Service → **Metrics** (CPU, Memory, Network)
3. **Health Check:** Backend `/api/health` endpoint'i otomatik check edilir

## Scaling

Railway'de otomatik scaling yok, manuel olarak:

1. Service → **Settings** → **Scaling**
2. Instance sayısını artır
3. Resource limits'i ayarla

## Cost Optimization

- **CPU-only model:** GPU kullanmıyor, daha ucuz
- **Sleep on idle:** Railway free tier'da idle servisler sleep'e gider
- **Data storage:** Sadece gerekli dosyaları upload et

## Production Checklist

- [ ] Environment variables set edildi
- [ ] CORS origins yapılandırıldı
- [ ] Supabase allowed origins güncellendi
- [ ] Data files upload edildi
- [ ] Health checks çalışıyor
- [ ] Frontend backend'e bağlanıyor
- [ ] Authentication çalışıyor
- [ ] Recommendations generate ediliyor
- [ ] Logs temiz (error yok)

