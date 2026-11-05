# Faza 2: indeks (chunking, embedding, pgvector)

Faza 2 parse qilingan 385 hujjatni qidirish mumkin bo'lgan bazaga aylantiradi. Natijada **20 789 ta chunk** hosil bo'ladi va har biri ikki yo'l bilan topiladi: ma'no bo'yicha (1 024 o'lchamli vektor) va so'z bo'yicha (full-text indeks). Butun korpusni embedding qilish taxminan 6,6 mln token, ya'ni **~$0,07** turadi.

```
lex.uz HTML ──(Faza 1)──▶ data/parsed/*.json ──(Faza 2)──▶ Postgres: documents + chunks
  bronze                     silver                          gold
```

Kod: `src/lexrag/index/` — `chunk.py` (bo'lish), `embed.py` (vektor), `db.py` (sxema), `indexer.py` (hammasini bog'laydi). Buyruq: `lexrag index`.

## 1. Chunking: nega aynan modda bo'yicha

**Chunk** — qidiruv qaytaradigan matn bo'lagi. LLM butun kodeksni o'qiy olmaydi, shuning uchun unga faqat savolga tegishli bir nechta bo'lak beriladi.

Bizda **1 modda = 1 chunk**. Sababi ustozning spetsifikatsiyasida: har bir iqtibosda `article` ("XXX-modda") bo'lishi shart. Matnni belgilangan uzunlikda (masalan, har 500 token) kessak, bitta bo'lakka ikki moddaning yarmi tushib qolishi mumkin. Unda qaysi moddaga havola berishni aniq aytib bo'lmaydi.

Bu tanlov raqamlarga ham tayanadi. Modda uzunligining medianasi 771 belgi, 90-persentil 2 350. Demak, moddalarning aksariyati bitta chunkka bemalol sig'adi.

**Uzun moddalar.** 4 000 belgidan uzun moddalar (549 ta) paragraf chegarasida ~2 000 belgilik bo'laklarga bo'linadi. Har bir bo'lak oldingi bo'lakning oxirgi paragrafini takrorlaydi. Bu **overlap** deyiladi: "ushbu moddaning birinchi qismida…" kabi gap o'zi tayangan matndan uzilib qolmaydi. Hech qanday chegarasiz bitta ulkan paragraf (amalda bu jadval, eng uzuni 129 223 belgi) gap oxirlarida kesiladi.

## 2. Qism (`part`)

Ustoz iqtibosda `"part": "1-qism"` maydonini so'raydi. Qonunchilik texnikasida moddaning har bir xatboshisi alohida **qism** hisoblanadi. `1)`, `2)` yoki `a)`, `b)` bilan boshlanadigan **bandlar** esa yangi qism ochmaydi, ular o'zidan oldingi qismga tegishli:

```
<xatboshi, ":" bilan tugaydi>   ← 1-qism
1) <band>                       ← 1-qism (band)
2) <band>                       ← 1-qism (band)
<keyingi xatboshi>              ← 2-qism
```

Haqiqiy misol: "Konstitutsiyaviy sud toʻgʻrisida"gi qonunning 4-moddasi "Konstitutsiyaviy sud:" bilan boshlanadi, keyin `1)`–`5)` bandlar keladi. Ularning hammasi 1-qism.

Har bir paragraf qism raqami bilan saqlanadi (`chunks.paragraphs`). Faza 4 da LLM iqtibos keltirganda, iqtibos qaysi paragrafdan olinganini topib, `part` ni aniqlaymiz.

## 3. `chunk_id`

Format: `lex:6445145#art-18.1#p1`. Ustozning namunasidagi `lex:XXXXXXX#art-XXX#p1` ko'rinishiga moslangan.
- `art-18.1` — 18¹-modda. Yuqori indeks URL'larda buzilib ketmasligi uchun nuqta bilan yozilgan.
- `p1` — moddaning nechanchi bo'lagi. Moddaning qismi bilan aralashtirmang.
- Raqamsiz bloklar (preambula va hokazo) `u{tartib}` bo'ladi.

ID matn mazmunidan emas, hujjat va modda raqamidan yasaladi. Shuning uchun indeks qayta qurilganda ham o'zgarmaydi. Spetsifikatsiya buni talab qiladi.

## 4. Embedding

**Embedding** — matnni sonlar vektoriga aylantirish. Ma'nosi yaqin matnlarning vektorlari ham bir-biriga yaqin bo'ladi. Ikki vektorning yaqinligi **cosine similarity** bilan o'lchanadi: ular orasidagi burchak qancha kichik bo'lsa, ma'no shuncha yaqin.

Model: **`baai/bge-m3`** (OpenRouter orqali).

| Xususiyat | Qiymat | Nega muhim |
|---|---|---|
| Tillar | 100+, jumladan o'zbek va rus | Ruscha savol o'zbekcha moddani topa oladi (testda 10 ta shunday savol bor) |
| Kirish uzunligi | 8 192 token | Eng uzun chunkimiz (4 541 belgi) bemalol sig'adi |
| O'lcham | 1 024 | `vector(1024)` ustuni |
| Narx | $0,01 / 1 mln token | Butun korpus ~$0,07 |

Modelga faqat modda matnining o'zi berilmaydi, oldiga kontekst ham qo'shiladi (`embed_text`):

```
Oʻzbekiston Respublikasining Konstitutsiyaviy sudi toʻgʻrisida
16-bob. Ayrim toifadagi ishlar boʻyicha ish yuritishning oʻziga xos xususiyatlari
101-modda. Oʻzbekiston Respublikasining Konstitutsiyasi va qonunlari normalarini sharhlash toʻgʻrisidagi ishlarni yuritish

<modda matni>
```

Sababi: "3-modda. Asosiy tushunchalar" degan matn yuzlab qonunlarda bor. Hujjat nomi va bob qo'shilgandagina vektor qaysi qonun haqida ekanini "biladi".

## 5. pgvector va HNSW

**pgvector** — Postgres kengaytmasi. U `vector` turini va "eng yaqin vektorlarni top" amalini qo'shadi (`ORDER BY embedding <=> $savol_vektori`, bunda `<=>` cosine masofa).

**HNSW** — vektorlar uchun indeks (graf). Indekssiz qidiruv savolni har bir chunk bilan solishtiradi. HNSW esa grafda "yaqin qo'shnilar" bo'ylab sakrab, javobni tezroq topadi. Bunday qidiruv **taxminiy** (approximate): juda kam hollarda eng yaqin chunkni o'tkazib yuborishi mumkin. 20 mingta chunkda to'liq qidiruv ham juda sekin emas, buning aniq vaqtini indekslagandan keyin o'lchaymiz. HNSW korpus kattalashganda (masalan, qarorlar va farmonlar qo'shilganda) kerak bo'ladi.

Takrorlanuvchanlik haqida: indeks o'zgarmagan bo'lsa, HNSW bir xil savolga bir xil natija qaytaradi. Spetsifikatsiya buni talab qiladi.

## 6. Full-text qidiruv: vektor yetmagan joyda

Vektor ma'noni yaxshi ushlaydi, lekin aniq so'z va raqamlarda adashishi mumkin: "18¹-modda", "BHM", "O'RQ-1184". Shuning uchun har bir chunkda `tsv` ustuni ham bor. Bu Postgres'ning so'zlar indeksi (GIN).

Konfiguratsiya `'simple'`: so'zlarni kichik harfga o'tkazadi va ajratadi, lekin o'zakka qisqartirmaydi (stemming yo'q). Postgres'da o'zbek tili lug'ati yo'q, rus yoki ingliz stemmeri esa o'zbek so'zlarini buzadi. `oʻ` ichidagi `ʻ` belgisi so'z ichida qoladi, buni test tekshiradi (`tests/test_index_db.py`).

Faza 4 da ikki natija ro'yxati birlashtiriladi. Bu **gibrid qidiruv** (hybrid retrieval) deyiladi va ikkala usulning kuchli tomonlarini oladi.

## 7. Inkremental indekslash

Har bir chunk uchun uchta hash ishlatiladi:

| Hash | Nimadan | Nima uchun |
|---|---|---|
| `content_hash` | hujjat matni (Faza 1) | lex.uz'da matn o'zgardimi |
| `index_hash` | matn + metadata + chunker versiyasi + model | hujjatni qayta yozish kerakmi |
| `embed_hash` | model + `embed_text` | bu chunkning vektori allaqachon bormi |

Natija: `lexrag index` ni ikkinchi marta ishga tushirsangiz, **hech narsa embedding qilinmaydi** va bu $0 turadi. Agar kodeksda bitta modda o'zgarsa, hujjat qayta yoziladi, lekin faqat o'sha moddaning vektori yangidan hisoblanadi. Bu kelajakdagi kunlik yangilanish (Faza 3) uchun muhim.

## 8. `doc_id`: lex.uz'da bitta hujjatning bir nechta ID'si bor

| Variant | ID | Biz |
|---|---|---|
| Kirillcha asl matn | `6445145` | API `doc_id` sifatida qaytaradi (sozlanadi) |
| Lotincha | `-6445145` | matnni shundan olamiz |
| Ruscha | `6445147` | `aliases` da |

`documents.aliases` ustunida hujjatning barcha ID'lari saqlanadi. Ustoz `expected_doc_id` da qaysi ID'ni ishlatsa ham, uni o'sha hujjatga bog'lay olamiz. API qaysi ID'ni qaytarishi `.env` dagi `LEXRAG_CANONICAL_ID` bilan sozlanadi. Ustozdan javob kelgach, faqat shu sozlama o'zgaradi.

## 9. Spetsifikatsiyaning `system` obyekti uchun metadata

- `doc_status`: indeksda faqat amaldagi hujjatlar bor, shuning uchun hammasi `"amalda"`.
- `edition_date`: lex.uz ko'rsatgan tahrirlar ichidan snapshot sanasigacha bo'lgan eng oxirgisi. 26 ta hujjatda kelajakda kuchga kiradigan tahrir ham bor (masalan, 01.01.2028). U bizdagi matn emas, shuning uchun hisobga olinmaydi.
- `lex_snapshot_date`: HTML yuklangan kun, ya'ni 2026-10-30.
- `index_version`: `lex-idx-2026-10-30`. Bu `index_meta` jadvalida saqlanadi va API `/v1/meta` uni shu yerdan o'qiydi.

## Ustoz so'rashi mumkin

**Chunking modda bo'yichami yoki uzunlik bo'yicha?** Modda bo'yicha. Faqat 4 000 belgidan uzun moddalar (549 ta, 3%) paragraf chegarasida bo'linadi va overlap bilan saqlanadi. Shuning uchun `article` maydoni doim aniq bo'ladi.

**Nega aynan `bge-m3`?** U ko'p tilli (o'zbek va rus), 8 192 tokengacha matnni qabul qiladi va juda arzon. Test to'plamida ruscha va kirillcha savollar bor. Ko'p tilli model ruscha savolni o'zbekcha moddaga bog'lay oladi.

**Nega faqat vektor qidiruv emas?** Vektor qidiruv raqam va aniq atamalarda adashadi ("18¹-modda", "BHM"). Full-text bu bo'shliqni yopadi. Ikkalasi birlashtirilsa, gibrid qidiruv hosil bo'ladi.

**Nega Postgres? Nega Pinecone yoki Chroma emas?** Metadata (hujjat, tahrir, status) va vektorlar bitta bazada, bitta tranzaksiyada turadi. Full-text ham shu yerda. Ikkinchi servisni ushlab turish kerak emas.

**Kuchini yo'qotgan hujjatlar indeksdami?** Yo'q. Discovery'da ular chiqarib tashlanadi. Hujjat keyinroq kuchini yo'qotsa, to'liq indekslashda bazadan o'chiriladi (`prune`). Shu sabab "eskirgan normaga tayanish" xatosi 0 ta bo'lishi kerak.

**Indeks qayta qurilsa ID'lar o'zgaradimi?** Yo'q. `chunk_id` hujjat ID'si va modda raqamidan yasaladi, `doc_id` esa lex.uz'ning raqami.

**Matn o'zgarsa nima bo'ladi?** Hashlar o'zgarishni aniqlaydi. Faqat o'zgargan chunklar qayta embedding qilinadi.

**Embedding qancha turadi?** Butun korpus ~6,6 mln token, ya'ni ~$0,07. Keyingi ishga tushirishlar faqat o'zgarganlarni hisoblaydi.

## Ishga tushirish

```bash
docker compose up -d db          # Postgres + pgvector, port 5434
uv run lexrag index --dry-run    # faqat chunking statistikasi, API chaqirilmaydi
uv run lexrag index              # OPENROUTER_API_KEY .env da bo'lishi kerak
uv run pytest                    # 76 ta test, baza testlari Postgres ishlayotgan bo'lsa ishlaydi
```
