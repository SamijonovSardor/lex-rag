# Faza 4: qidiruv va javob (RAG yadrosi)

Tizim savolga qonundan iqtibos bilan javob beradi. Har bir iqtibos manba matnida aynan borligi tekshiriladi. Dev to'plamda ustozning qidiruv bo'yicha uchala maqsadi bajarildi. To'plamda 47 savol bor: 42 tasining javobi bor (shu jumladan ustozning PDF'dagi ikkala namunasi), 5 tasiniki yo'q. Qidiruv metrikalari javobi bor 42 ta savol bo'yicha hisoblangan:

| Metrika | Bizda | Ustozning maqsadi |
|---|---|---|
| Hit@5 (hujjat) | 95,2% | ≥ 90% |
| Hit@5 (modda) | 90,5% | ≥ 80% |
| MRR | 0,772 | ≥ 0,70 |
| Javobsiz savollarda `no_answer` | 5/5 | ≥ 9/10 |
| Kontekstdan tashqari iqtibos | 0 ta | 0 ta |
| Kechikish p50 / p95 | 4,9–6,5 s / 8,8–11,3 s | ≤ 5 s / ≤ 12 s |

**Bu raqamlarni qanday o'qish kerak.** Dev to'plam sintetik: savollar moddaning o'zidan LLM orqali yozilgan, shuning uchun moddaning so'zlarini takrorlaydi va natija optimistik chiqadi. U sozlamalarni bir-biri bilan solishtirish uchun kerak. Ustozning testidagi natija past bo'lishi mumkin. Haqiqiy baholash Faza 5 da bo'ladi.

Kod: `src/lexrag/rag/`. Buyruq: `lexrag ask "savol"`.

## Oqim

```
savol
 │ 1. til: uz-Latn / uz-Cyrl / ru                       (text/lang.py)
 │ 2. ruscha bo'lsa → LLM o'zbekchaga tarjima qiladi    (rag/rewrite.py)   ~2,8 s
 │    kirillcha bo'lsa → lotinga o'giriladi (cyr_to_lat)
 │ 3. bge-m3 embedding                                   ~0,9 s
 │ 4. vektor qidiruv (top-50) + full-text qidiruv (top-50) → RRF   ~0,3 s
 │ 5. reranker: Cohere rerank-5-pro, 30 nomzod → top-5   ~1,2 s
 │ 6. LLM (gpt-6-luna) JSON javob: matn + [n] + iqtiboslar   ~2,5 s
 │ 7. iqtibos tekshiruvi: quote manbada aynan bormi?
 ▼
javob + citations + contexts
```

## 1. Til aniqlash va topilgan xato

O'zbek kirill va rus tili harflarining ko'pi umumiy. Farqlovchi belgilar: `ў қ ғ ҳ` faqat o'zbekchada, `ы щ` faqat ruschada bor. Ular bo'lmasa, tez-tez uchraydigan so'zlar va qo'shimchalarga qaraladi (`-си`, `-нинг` yoki `-ого`, `-ый`).

**Topilgan xato.** Faza 1 dagi `normalize_query` kirill harflari ko'p bo'lgan har qanday savolni o'zbekcha qoidalar bilan lotinga o'girardi, ruscha savolni ham. Natijada "Какой срок" "Kakoy srok"ga aylanib, qidiruvni buzardi. Endi faqat o'zbek kirilli o'giriladi.

## 2. Gibrid qidiruv va RRF

Ikkita ro'yxat olinadi: vektor bo'yicha top-50 va full-text bo'yicha top-50. Ular **RRF** (Reciprocal Rank Fusion) bilan birlashtiriladi:

```latex
\mathrm{RRF}(d) = \sum_{\text{ro'yxat}} \frac{1}{60 + \mathrm{rank}(d)}
```

**Nega RRF.** Vektor balli (cosine, 0–1) va full-text balli (`ts_rank_cd`) har xil shkalada. Ularni qo'shib bo'lmaydi, o'rinlarni esa bo'ladi. Ikkala ro'yxatda ham yuqorida turgan chunk g'olib chiqadi.

**Kichik, lekin muhim detal.** pgvector'ning HNSW indeksi standart holatda ko'pi bilan 40 ta natija qaytaradi (`hnsw.ef_search = 40`). Biz 50 ta so'raymiz, shuning uchun uni 100 ga oshiramiz. Aks holda jimgina 40 ta qaytardi.

## 3. Reranker

**Bi-encoder va cross-encoder.** Embedding modeli (bi-encoder) savol va moddani **alohida-alohida** vektorga aylantiradi. Bu tez, lekin taxminiy. Reranker (cross-encoder) savol va moddani **birga** o'qib, "bu modda shu savolga javob beradimi?" deb baholaydi. U aniqroq, lekin sekin, shuning uchun faqat birinchi bosqich tanlagan 30 ta nomzodga qo'llanadi.

Dev to'plamdagi o'lchovlar:

| Sozlama | Hit@5 modda | MRR |
|---|---|---|
| Gibrid, rerankersiz | 52,5% | 0,373 |
| + `rerank-5-fast` | 85,0% | 0,779 |
| **+ `rerank-5-pro`** (tanlangan) | **90,5%** | **0,772** |
| + reranker tartibini qidiruv tartibi bilan birlashtirish | 54,8% | 0,390 |

Reranker'siz modda darajasidagi maqsadga yetib bo'lmaydi. Kerakli hujjat topiladi (87,5%), lekin aniq modda kamdan-kam yuqorida turadi. Oxirgi qatordagi g'oya sinab ko'rildi va rad etildi: u natijani reranker'dan oldingi kuchsiz tartib tomon tortib ketdi.

**Nega `pro`.** Ustozning o'z namunasi: "Какой срок исковой давности…?". Javob Fuqarolik kodeksi 150-moddasida: "Umumiy daʼvo muddati — uch yil." Bu atigi 31 belgilik modda. `rerank-5-fast` uni eng past baholadi (0,44), `rerank-5-pro` esa ikkinchi o'ringa qo'ydi (0,617). Narxi: bitta savol uchun ~$0,0006.

**Reranker kechiksa.** Uning javobini 5 soniyadan ortiq kutmaymiz. Kechiksa, gibrid qidiruv tartibi ishlatiladi. Bitta o'lchovda reranker 16,6 soniyaga cho'zilgan edi va p95 kechikishni buzayotgan edi.

## 4. Ruscha savollar

Indeks o'zbekcha. bge-m3 ruscha savolni o'zbekcha moddaga bog'lay oladi, lekin qolgan ikki bosqich buni eplay olmaydi:
- **Full-text qidiruv:** ruscha va o'zbekcha so'zlar umuman mos kelmaydi.
- **Reranker:** ruscha savol bilan u barcha moddalarga 0,002–0,003 baho berdi, ya'ni ularni farqlay olmadi. Uning yoqilishi ruscha savollarda Hit@5 (modda)ni 60% dan **20%** ga tushirdi.

**Yechim:** ruscha savolni avval LLM huquqiy atamalar bilan o'zbekchaga tarjima qiladi ("исковая давность" → "da'vo muddati"). Keyingi uchala bosqich o'zbekcha matn bilan ishlaydi. Natija: ruscha savollarda 20% → **82%**. Tarjima javobdagi `debug.rewritten_query` maydonida ko'rinadi. Javobning o'zi esa savol tilida, ya'ni ruscha beriladi.

## 5. Javob generatsiyasi

Model `openai/gpt-6-luna` (OpenRouter). Narxi: 1M kiruvchi token uchun $0,10, chiquvchi token uchun $0,50. Javob qat'iy **JSON sxema** bilan so'raladi (structured outputs):

```json
{"no_answer": false,
 "answer": "Umumiy daʼvo muddati uch yil [1].",
 "citations": [{"ref": 1, "context": 1, "quote": "Umumiy daʼvo muddati — uch yil"}],
 "confidence": 0.9}
```

Prompt qoidalari: faqat berilgan parchalardan foydalanish; har bir huquqiy da'vodan keyin `[n]` qo'yish; quote'ni harfma-harf ko'chirish va tarjima qilmaslik; raqamlarni aynan ko'chirish; javob bo'lmasa `no_answer: true` va taxmin qilmaslik.

**`temperature` haqida halol javob.** gpt-6-luna "reasoning" model va `temperature` parametrini qo'llab-quvvatlamaydi: OpenRouter ro'yxatida u yo'q. Biz `temperature=0` ni yuboramiz, lekin model uni e'tiborsiz qoldiradi. `seed` esa ishlaydi. Takrorlanuvchanlikning asosiy kafolati boshqa joyda: **manbalar** (qidiruv va reranker) bir xil savol uchun bir xil chiqadi, chunki ular deterministik. Spetsifikatsiyada aynan shu talab qilinadi: "bir xil manbalarni qaytarishi kerak". Javob matnidagi so'zlar biroz farq qilishi mumkin.

`reasoning_effort` = `minimal`: model kamroq "o'ylaydi" va tezroq javob beradi. Dev to'plamda sifat `low` bilan bir xil chiqdi.

## 6. Iqtibos tekshiruvi

LLM'ning iqtiboslariga ishonilmaydi. Har biri uch marta tekshiriladi:
1. **Parcha raqami** biz yuborgan parchalardan biri bo'lishi kerak. Tizim ko'rmagan manbaga havola qolmaydi.
2. **Quote** shu parcha matnida aynan bo'lishi kerak. Faqat apostrof, bo'sh joy va harf katta-kichikligi farqiga yo'l qo'yiladi. Topilsa, iqtibosga modelning nusxasi emas, **manbaning o'z matni** yoziladi.
3. Quote turgan paragrafdan **qism** aniqlanadi (`"part": "1-qism"`).

O'tmagan iqtibos o'chiriladi, javobdagi `[n]` belgilari qayta raqamlanadi. Birorta ham to'g'ri iqtibos qolmasa, javob berilmaydi va savol `no_answer` bo'ladi: iqtibossiz huquqiy javob tekshirib bo'lmaydigan javob. Oxirgi o'lchovda 44 ta iqtibos o'tdi, 1 tasi rad etildi, kontekstdan tashqariga chiqqani 0.

## 7. `no_answer`: taxmin qilmaslik

- "Ertaga havo qanday?" kabi savollarga tizim "berilgan qonunlarda javob yo'q" deydi.
- Dev to'plamda "pasport narxi" savolini javobsiz deb belgilagan edim. Tizim esa "Davlat boji to'g'risida"gi qonundan "1 BHM" degan javobni iqtibos bilan topdi. Xato tizimda emas, mening belgimda edi va tuzatildi.
- Javobi bor 42 savoldan 5 tasida qidiruv kerakli moddani topmadi. Shunda tizim taxmin qilmasdan "topilmadi" dedi. Bu "to'qilgan javob"dan yaxshiroq, lekin baribir yo'qotish: qidiruv yaxshilangani sari bu son kamayadi.

## 8. Kechikish

| Bosqich | p50 |
|---|---|
| Ruscha tarjima (faqat ru) | 2,8 s |
| Embedding | 0,9 s |
| Postgres qidiruv | 0,3 s |
| Reranker | 1,2 s |
| Generatsiya | 2,5 s |

Uchta o'lchovda p50 4,9–6,5 s, p95 8,8–14,6 s bo'ldi. Bu tebranishning katta qismi OpenRouter tarmog'idan keladi. p95 hozir maqsad ichida (≤ 12 s), p50 esa chegara atrofida.

## 9. Narx

Bitta savol ~$0,001: LLM ~$0,0005, reranker ~$0,0006, embedding juda arzon. Ustozning 100 savollik testi taxminan **$0,10** turadi.

## Cheklovlar (halol)

- Dev to'plam sintetik va kichik (47 savol), natija optimistik.
- Javobning **to'g'riligi** va **faithfulness** hali o'lchanmagan, buning uchun LLM-hakam kerak (Faza 5).
- `as_of_date` qo'llab-quvvatlanmaydi: indeksda faqat joriy tahrir bor.
- Bir nechta hujjatga tayanadigan savollar (testda 15 ta) alohida o'lchanmagan.

## Ustoz so'rashi mumkin

**Nega reranker kerak?** Usiz kerakli modda top-5 ga atigi 52,5% hollarda kiradi, maqsad esa ≥ 80%. Reranker savol va moddani birga o'qib baholaydi, natijada 90,5% bo'ladi.

**Ruscha savollar qanday ishlaydi?** Savol LLM bilan o'zbekcha huquqiy so'rovga tarjima qilinadi va o'shani qidiramiz. Javob esa ruscha beriladi. Tarjimasiz reranker ruscha savolda deyarli tasodifiy natija berardi (Hit@5 modda 20%).

**LLM to'qib chiqarsa-chi?** Har bir quote manba matnida aynan qidiriladi. Topilmagan iqtibos o'chiriladi. Bitta ham to'g'ri iqtibos qolmasa, javob berilmaydi.

**temperature=0 ishlaydimi?** gpt-6-luna reasoning model, u temperature'ni qo'llab-quvvatlamaydi, `seed` ni esa qo'llaydi. Manbalar deterministik, javob matni esa biroz farq qilishi mumkin.

**Qism (`part`) qanday aniqlanadi?** Quote turgan xatboshi bo'yicha. Bandlar (`1)`, `a)`) yangi qism ochmaydi (Faza 2).

**RRF nima?** Ikki ro'yxatni ballar bo'yicha emas, o'rinlar bo'yicha birlashtirish usuli: 1/(60 + o'rin) yig'indisi. Shkalalari har xil ballarni solishtirishga hojat qolmaydi.

**Retrieval score nima?** Savol va chunk vektorlari orasidagi cosine o'xshashlik (0–1). `rerank_score` esa reranker bahosi.
