# Faza 3: indeksni lex.uz bilan yangilab turish (Airflow)

Indeks endi bir martalik nusxa emas. Kunlik va haftalik ishlar lex.uz'dagi o'zgarishlarni topadi, faqat o'zgargan hujjatlarni qayta yuklaydi, parse qiladi va indekslaydi. O'zgarmagan matn uchun embedding qayta hisoblanmaydi, ya'ni pul sarflanmaydi.

Kod: `src/lexrag/update.py` (mantiq), `dags/lexrag_pipeline.py` (Airflow), `airflow/` (Docker image). Qo'lda ishga tushirish: `lexrag update --mode daily|full`.

## Zanjir

```
lexrag_daily   (@daily)   RSS → yangi va o'zgartirilgan hujjatlar ─┐
lexrag_weekly  (@weekly)  butun ro'yxat → barcha hujjatlar ────────┴─▶ asset lexrag_raw
lexrag_parse   (raw kelganda)      HTML → JSON, faqat o'zgarganlar ───▶ asset lexrag_parsed
lexrag_index   (parsed kelganda)   JSON → Postgres, inkremental ───────▶ asset lexrag_index
```

**Data-triggered (asset) nima.** Oddiy DAG soat bo'yicha ishga tushadi (`@daily`). `lexrag_parse` va `lexrag_index` esa vaqt bo'yicha emas, **ma'lumot paydo bo'lganda** ishga tushadi. Oldingi qadam muvaffaqiyatli tugab, asset'ni "yangiladim" deb belgilasa, Airflow keyingi DAG'ni o'zi boshlaydi. O'zgargan hujjatlar ID'lari asset hodisasining `extra` maydoni orqali keyingi qadamga uzatiladi.

**Bo'sh kun.** RSS'da bizga tegishli hech narsa bo'lmasa, birinchi qadam `skipped` holatida tugaydi. O'tkazib yuborilgan task asset hodisasini chiqarmaydi, shuning uchun parse va index umuman ishga tushmaydi.

## Kunlik rejim: RSS'dan nimani olamiz

lex.uz'ning `/uz/rss` lentasida oxirgi qabul qilingan hujjatlar bor: qonunlar, farmonlar, qarorlar, buyruqlar. Har bir yozuvning `description` maydoni hujjat turi bilan boshlanadi.

| RSS'dagi yozuv | Nima qilamiz |
|---|---|
| Qaror, farmon, buyruq | E'tiborsiz qoldiramiz, chunki bu hujjatlar bizning qamrovimizda yo'q |
| Yangi mazmunli qonun yoki kodeks | Yuklab olamiz. Agar lotincha varianti boshqa ID bilan bo'lsa, o'shani olamiz |
| O'zgartiruvchi qonun ("…o'zgartirishlar kiritish to'g'risida") | Sahifasini ochamiz va matnidagi havolalardan **qaysi hujjatlarni o'zgartirayotganini** topamiz. Ular bizda bo'lsa, qayta yuklaymiz |

**Asosiy g'oya.** O'zgartiruvchi qonun matnida o'zgartirilayotgan hujjatga havola bor. Masalan, sinovda ishlatilgan haqiqiy qonun (`-8547667`) ikkita kodeksga havola beradi: `/uz/docs/-3523891` va `/uz/docs/-4674902`. Biz o'zgartiruvchi qonunning o'zini indekslamaymiz, chunki uning mazmuni asosiy hujjatning amaldagi matniga allaqachon kiritilgan. Faqat asosiy hujjatni yangilaymiz. Havola kirillcha yoki ruscha ID bilan kelsa ham, hujjatning barcha ID'lari (`twins`) orqali topiladi.

Ko'rib chiqilgan RSS yozuvlari `data/update_state.json` faylida saqlanadi. Ertasiga faqat yangi yozuvlar ko'riladi.

## Haftalik rejim: xavfsizlik tarmog'i

Ba'zi o'zgarishlar RSS orqali kelmaydi:
- O'zgartirish bugun e'lon qilinib, keyinroq kuchga kirishi mumkin. lex.uz amaldagi matnni kuchga kirish kunida yangilaydi, bizning kunlik ishimiz esa uni e'lon qilingan kuni yuklagan bo'ladi.
- Kunlik ish bir kun ishlamay qolishi mumkin.
- Hujjat kuchini yo'qotishi mumkin.

Shuning uchun haftada bir marta butun ro'yxat qayta olinadi va qamrovdagi barcha hujjatlar qayta yuklanadi. Kuchini yo'qotgan hujjatlar indeksdan o'chiriladi (`prune`).

## Nega bu arzon

| Qadam | O'zgarmagan hujjat uchun |
|---|---|
| Yuklash | Qayta yuklanadi (lex.uz'ga ~1 so'rov/soniya) |
| Parse | `content_hash` bir xil bo'lsa, keyingi qadamga uzatilmaydi |
| Index | `embed_hash` bir xil chunk'lar eski vektorini oladi, ya'ni $0 |

Kodeksdagi bitta modda o'zgarsa, faqat o'sha moddaning vektori qayta hisoblanadi (Faza 2, inkremental indekslash).

## lex.uz'ga hurmat

- Klient so'rovlar orasida ~1 soniya kutadi (Faza 1).
- Barcha lex.uz task'lari bitta slotli `lexuz` pool'ida ishlaydi. Kunlik va haftalik ish bir vaqtda saytga murojaat qilmaydi.
- Kunlik ish o'nlab so'rov bilan cheklanadi: 1 ta RSS, bir nechta o'zgartiruvchi qonun va ularning maqsad hujjatlari.

## Ishga tushirish

```bash
# Airflowsiz, xuddi shu qadamlar:
uv run lexrag update --mode daily
uv run lexrag update --mode full

# Airflow (~1,5 GB RAM kerak):
docker compose --profile airflow up -d --build      # → http://127.0.0.1:8080
docker compose --profile airflow run --rm airflow airflow dags test lexrag_daily
```

Airflow standart holatda lokal indeksni yangilaydi. Production'dagi Neon indeksini yangilash uchun `.env` ga `LEXRAG_AIRFLOW_DATABASE_URL=<Neon URL>` qo'shiladi.

**Cheklov:** bepul infratuzilmada Airflow'ni 24/7 ushlab turadigan joy yo'q. Airflow kompyuterda yoqilgan paytda ishlaydi.

## Ustoz so'rashi mumkin

**Indeks qanday yangilanadi?** Har kuni lex.uz RSS lentasi tekshiriladi. Yangi qonun bo'lsa, u qo'shiladi. O'zgartiruvchi qonun bo'lsa, u o'zgartirayotgan hujjatlar qayta yuklanadi. Har hafta esa barcha hujjatlar to'liq qayta tekshiriladi.

**O'zgartiruvchi qonun qaysi hujjatni o'zgartirayotganini qanday bilasiz?** Uning matnidagi havolalardan: lex.uz o'zgartirilayotgan hujjatga va hatto aniq moddasiga havola qo'yadi.

**Nega Airflow?** Qadamlar ma'lumot orqali bog'langan (assets). Har bir run, xato va qayta urinish UI'da ko'rinadi. Bitta slotli pool lex.uz'ga yukni cheklaydi. Xuddi shu qadamlar `lexrag update` CLI orqali Airflowsiz ham ishlaydi.

**Nega o'zgarmagan hujjat qayta indekslanmaydi?** Matn hashi solishtiriladi. Hash bir xil bo'lsa, parse qadami uni keyingi bosqichga uzatmaydi. Indeks qadamida esa chunk darajasidagi hash vektorni qayta ishlatadi.

**Kuchini yo'qotgan hujjat nima bo'ladi?** Haftalik to'liq tekshiruvda u qamrovdan chiqadi va indeksdan o'chiriladi.
