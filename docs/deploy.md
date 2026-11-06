# Deploy: API'ni internetga chiqarish ($0/oy)

Ustoz testni o'tkazishi uchun bizdan ikki narsa kerak: **base URL** va **test uchun API kalit**. Bu spetsifikatsiya checklistidagi oxirgi band.

| Qism | Qayerda | Bepul tarif (2026-11-06 holatiga) |
|---|---|---|
| Postgres + pgvector (indeks) | [Neon](https://neon.com/pricing) | Har loyihaga 1 GB (bizga ~355 MB kerak), oyiga 100 CU-soat, 5 daqiqa ishlatilmasa uxlaydi |
| API (Docker) | [Render](https://render.com/docs/free) | 512 MB RAM, 0,1 CPU, oyiga 750 soat; 15 daqiqa trafik bo'lmasa uxlaydi, uyg'onishi ~1 daqiqa |
| Uyg'oq ushlash | GitHub Actions (`.github/workflows/keepalive.yml`) | Ochiq (public) repo uchun cheklovsiz |

Nega Hugging Face emas: Docker Space'lar endi faqat pullik PRO obunada. Bundan tashqari, Space'dan faqat 80, 443 va 8080 portlarga chiqish mumkin, Neon'ning Postgres porti esa 5432.

Lokal sinov (`--memory 512m --cpus 0.1`, ya'ni Render'ning cheklovlari bilan): ishga tushgandan keyin 59 MB RAM, batch ishlayotganda eng ko'pi 76 MB. 4 savollik batch "iliq" holatda 11 soniyada bajarildi.

## Qadamlar

### 1. Neon (siz)
1. https://neon.com → GitHub akkaunti bilan ro'yxatdan o'ting.
2. **New project**: nomi `lex-rag`, region **AWS Europe Central 1 (Frankfurt)**. Bu O'zbekistonga va Render'ga eng yaqin region.
3. **Connect** tugmasini bosing. "Connection pooling" ni **o'chiring**, chunki bizga to'g'ridan-to'g'ri (direct) ulanish kerak. `postgresql://...neon.tech/neondb?sslmode=require` ko'rinishidagi satrni nusxalang.
4. Uni `lex-rag\.env` fayliga yozing: `LEXRAG_NEON_URL=postgresql://...` Chatga yubormang.

Nega pooled emas: Neon'ning pooled ulanishi PgBouncer'ning transaction rejimida ishlaydi. Bu rejimda `SET hnsw.ef_search` kabi sessiya sozlamalari keyingi tranzaksiyada yo'qoladi. Bizga bor-yo'g'i ~8 ta ulanish kerak, direct ulanish bunga bemalol yetadi.

### 2. Indeksni ko'chirish (men)
```bash
uv run python scripts/copy_index.py --to "$LEXRAG_NEON_URL"
```
Vektorlar qayta hisoblanmaydi, shuning uchun bu bosqich $0 turadi.

### 3. GitHub (siz, keyin men)
1. GitHub'da yangi repo yarating: `lex-rag`. **Public** tavsiya etiladi: ping workflow'i bepul ishlaydi va repo portfolioga ham yaraydi. `.env` va `data/` repoga tushmaydi.
2. Repo havolasini menga bering. Men remote qo'shaman, siz esa push qilasiz: `! git push -u origin main` (brauzer orqali GitHub login so'raladi).

### 4. Render (siz)
1. https://render.com → GitHub bilan kiring.
2. **New → Blueprint** → `lex-rag` repoini tanlang. `render.yaml` ni Render o'zi o'qiydi.
3. Uchta maxfiy qiymatni kiriting:
   - `LEXRAG_DATABASE_URL`: Neon'dagi satr, `LEXRAG_NEON_URL` bilan bir xil;
   - `OPENROUTER_API_KEY`: `.env` dagi bilan bir xil;
   - `LEXRAG_API_KEYS`: kalitlar vergul bilan ajratiladi. Ustoz uchun alohida kalit bo'lishi kerak, men uni yaratib, `.env` ga yozib beraman.
4. Deploy tugagach, `https://lex-rag-XXXX.onrender.com` ko'rinishidagi URL paydo bo'ladi.

### 5. Ping (siz)
GitHub repo → Settings → Secrets and variables → Actions → **Variables** → `LEXRAG_URL` = Render URL.

### 6. Tekshiruv (men)
`/v1/health`, `/v1/meta` va kichik batch tekshiriladi. Keyin ustozga beriladi: base URL va uning kaliti.

## Cheklovlar (ustozga aytish kerak)
- Bepul Render 0,1 CPU'da ishlaydi, shuning uchun bitta savol lokaldagidan 1–3 soniya sekinroq javob olishi mumkin.
- Ping kechiksa, Render uxlab qolishi mumkin. Unda birinchi so'rov ~1 daqiqa kutadi, keyingilari tez ishlaydi.
- OpenRouter kalitining limiti $1,50. 100 savollik test taxminan $0,10 turadi.

## Holat (2026-11-06)

- API: https://lex-rag-mdca.onrender.com, hujjatlar sahifasi `/docs`. Indeks Neon'da: 385 hujjat, 20 789 chunk, 353 MB.
- Ping: GitHub Actions `keep-alive`, `LEXRAG_URL` repository variable orqali ishlaydi. Birinchi ishga tushirish muvaffaqiyatli o'tdi.
- Internet orqali sinov: 4 savollik batch 11,5 s. Bir xil batch qayta yuborilganda 0,5 s (saqlangan natija). Ikkala kalit ham qabul qilinadi, noto'g'ri kalit 403, kalitsiz so'rov 401.

### Deploy'dan keyin topilgan xato: Neon uxlaganda ulanishlar uziladi

Deploy'dan bir necha daqiqa o'tib `/v1/health` 503 qaytardi: `terminating connection due to administrator command`. Neon'ning bepul hisoblash qismi 5 daqiqa so'rov kelmasa uxlaydi va **barcha ochiq ulanishlarni yopadi**. API esa yopilgan ulanishni qayta ishlatmoqchi bo'lgan. Ustozning testida bu tanaffusdan keyingi birinchi batch'dagi barcha savollarni `RETRIEVAL_FAILED` qilardi.

Ping bu muammoni hal qilmaydi, chunki u ataylab bazaga tegmaydi. Agar ping bazani ham uyg'oq ushlasa, Neon'ning oyiga 100 CU-soatlik bepul hisoblash vaqti tez tugab qolardi.

Tuzatish (`fc70665`):
- ulanishlar puli (`ConnectionPool(check=...)`) har bir ulanishni berishdan oldin tekshiradi;
- har bir ishchi oqimning pipeline'i savoldan oldin `SELECT 1` yuboradi. Ulanish o'lik bo'lsa, qaytadan ulanadi va `hnsw.ef_search` sessiya sozlamasini qayta qo'yadi;
- `tests/test_reconnect.py` ulanishni `pg_terminate_backend` bilan majburan uzib, xuddi shu vaziyatni takrorlaydi.

Narxi: Neon uyg'onishi uchun tanaffusdan keyingi birinchi savolga ~0,5–2 s qo'shiladi.
