# Faza 6: ustozning spetsifikatsiyasi bo'yicha API

API ustozning "RAG API — test spetsifikatsiyasi (lex.uz)" hujjatini (2026-11-03) maydonma-maydon bajaradi. Spetsifikatsiyaning 7-bo'limidagi checklistning har bir bandi `tests/test_api.py` da alohida test bilan tekshiriladi.

Kod: `src/lexrag/api/` — `schemas.py` (so'rov va javob formatlari), `store.py` (batch'larni saqlash), `app.py` (endpointlar).

```bash
uv run uvicorn lexrag.api.app:app --port 8000
```

## Endpointlar

| Endpoint | Nima qiladi |
|---|---|
| `POST /v1/rag/batch` | 1–10 savol → har biriga javob, iqtiboslar, kontekstlar |
| `GET /v1/rag/batch/{batch_id}` | Saqlangan batch'ni qaytaradi (async rejim yoki qayta tekshirish) |
| `POST /v1/rag/query` | Bitta savol, debug uchun |
| `GET /v1/meta` | Model, embedding, indeks versiyasi, lex.uz snapshot sanasi |
| `GET /v1/health` | Servis va indeks ishlayaptimi |

## So'rov qanday ishlanadi

```
POST /v1/rag/batch
  │ 1. Kalit tekshiruvi (Bearer) ─────────── yo'q: 401, noto'g'ri: 403
  │ 2. Rate limit (kalit bo'yicha, soniyada 5) ── oshsa: 429 + Retry-After
  │ 3. Validatsiya (1–10 savol, takroriy id yo'q, bo'sh savol yo'q) ── 400
  │ 4. batch_id bazada bormi?
  │      bor, mazmuni boshqa ──────────────── 409
  │      bor, tugagan ─────────────────────── saqlangan javob (qayta hisoblanmaydi)
  │      bor, ba'zi savollari retryable xato ─ faqat o'shalar qayta ishlanadi
  │ 5. Savollar 4 ta parallel oqimda ishlanadi
  │ 6. 110 soniyada tugasa ───────────────── 200 + to'liq javob
  │    tugamasa ──────────────────────────── 202 {"status": "processing"}, GET bilan olinadi
```

## Asosiy tushunchalar

**Idempotentlik.** Bir xil so'rovni necha marta yuborsangiz ham, natija bitta bo'ladi. Tarmoq uzilsa, tester bir xil `batch_id` bilan qayta yuboradi. Biz har bir batch'ni bazada (`batches` jadvali) saqlaymiz, shuning uchun qayta yuborilganda LLM qayta chaqirilmaydi. Pul tejaladi va natija o'zgarmaydi. So'rov mazmunining sha256 hashi ham saqlanadi: bir xil `batch_id` boshqa savollar bilan kelsa, bu xato, javob 409 bo'ladi.

**Spetsifikatsiyadagi ikki qoidaning to'qnashuvi.** Bir tomondan, bir xil `batch_id` saqlangan natijani qaytarishi kerak. Ikkinchi tomondan, `retryable: true` xato bo'lsa, tester o'sha `batch_id` bilan qayta yuboradi. Biz ikkalasini shunday bajaramiz: saqlangan natija qaytadi, lekin undagi retryable xato bo'lgan savollar qayta ishlanadi. Faqat o'shalar.

**Xato izolyatsiyasi.** Bitta savolda LLM javob bermasa (timeout), batch baribir 200 qaytaradi. Faqat o'sha savolning `status`i `error` yoki `timeout` bo'ladi va `error` obyekti to'ldiriladi.

**Sync va async.** Asosiy rejim sync: javob shu so'rovning o'zida qaytadi. Batch 110 soniyadan oshsa, server 202 qaytaradi va ishni davom ettiradi. Tester `GET /v1/rag/batch/{id}` ni so'rab turadi (polling). Server o'chib qolsa, chala qolgan batch'lar keyingi ishga tushishda `failed` deb belgilanadi, shuning uchun tester ularni qayta yuboradi.

**Parallel ishlash.** 10 savolni ketma-ket ishlasak, savoliga 5 soniyadan 50 soniya ketadi. 4 ta oqimda bu ~15 soniya. Har bir oqimning o'z baza ulanishi va o'z HTTP klientlari bor. Postgres ulanishini bir vaqtda ikki oqim ishlatishi mumkin emas.

**Rate limit.** Spetsifikatsiya soniyasiga kamida 1 batch'ni talab qiladi, bizda limit 5. Oshib ketsa, 429 va `Retry-After` sarlavhasi qaytadi: tester qancha kutishni shundan biladi.

**`system` obyekti.** Har bir javobda model, embedding, reranker, indeks versiyasi va snapshot sanasi bor. Ustoz ikki run'ni taqqoslaganda farq qayerdan kelganini ko'radi: model o'zgarganmi yoki indeks.

## `doc_id` qarori

lex.uz'da bitta qonunning bir nechta ID'si bor. Masalan, Fuqarolik kodeksi (birinchi qism):

| Variant | ID | URL |
|---|---|---|
| Asl matn (kirill) | `111189` | https://lex.uz/docs/111189 |
| Lotin | `-111189` | https://lex.uz/uz/docs/-111189 |
| Rus | `111181` | |

Spetsifikatsiyada qaysi ID ishlatilishi yozilmagan. Lekin namunada `"doc_id": "lex:XXXXXXX"` va `"url": "https://lex.uz/docs/XXXXXXX"` berilgan: minus belgisi ham, til prefiksi ham yo'q. Bu asl hujjat raqami. Shuning uchun `doc_id` = asl raqam. Matnni lotin variantidan olsak ham shunday.

Tester ID'ni boshqa variantdan olgan bo'lsa ham moslay olishi uchun, har bir iqtibos va kontekstda qo'shimcha `doc_aliases` maydoni bor. Unda hujjatning barcha ID'lari bo'ladi, qaytarilgan ID birinchi turadi. `source_url` paragrafning aniq joyiga olib boradi (`#-153703`). Bunday bog'lanish faqat lotin sahifasida ishlaydi, chunki paragraf ID'lari o'sha sahifaniki.

## Ustoz so'rashi mumkin

**`doc_id` qaysi ID?** lex.uz'dagi asl hujjat raqami, `https://lex.uz/docs/{id}` dagi raqam. Spetsifikatsiyadagi namunaga shu mos keladi. Boshqa variantlarning ID'lari `doc_aliases` da.

**Bir xil batch ikki marta yuborilsa nima bo'ladi?** Ikkinchi marta bazadagi natija qaytadi, LLM chaqirilmaydi. Mazmuni boshqa bo'lsa, javob 409.

**Bitta savol yiqilsa, butun batch yiqiladimi?** Yo'q. Batch 200 qaytaradi, faqat o'sha savolning statusi `error` bo'ladi va `retryable` ko'rsatiladi.

**Batch 120 soniyadan oshsa-chi?** 110 soniyada 202 qaytadi, ish fonda davom etadi va natija GET orqali olinadi. Natijalar bazada saqlanadi, avtomatik o'chirilmaydi, ya'ni 30 kunlik talab bajariladi.

**`o'` va `g'` buzilmaydimi?** Hammasi UTF-8 da ishlaydi va `Content-Type: application/json; charset=utf-8` bilan qaytadi. Javobdagi JSON o'zbekcha harflarni `ʻ` ko'rinishiga o'tkazmaydi. Bu test bilan tekshirilgan.

**Iqtibos kontekstlar ichida bo'lishi qanday kafolatlanadi?** Iqtibos faqat LLM'ga yuborilgan kontekstlardan olinadi, ularning har biri `contexts` ro'yxatida qaytadi. Tizim ko'rmagan manbaga havola berib bo'lmaydi: bunday iqtibos tekshiruvda tashlab yuboriladi (Faza 4).
