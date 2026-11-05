# Faza 5: ustozning testini o'zimizda takrorlash

Ustozning testi o'zimizda aynan takrorlandi: o'sha formatdagi 100 savol, o'sha tartibda yuborilgan 10 ta batch, 6-bo'limdagi o'sha metrikalar. Birinchi to'liq run'da (`eval-2026-11-06`) **12 metrikadan 9 tasi** maqsadga yetdi.

| Qatlam | Metrika | Natija | Maqsad | |
|---|---|---|---|---|
| Retrieval | Hit@5 (hujjat) | 94,4% | ≥ 90% | ✅ |
| Retrieval | Hit@5 (modda) | 86,7% | ≥ 80% | ✅ |
| Retrieval | MRR | 0,727 | ≥ 0,70 | ✅ |
| Retrieval | Context precision | 41,4% | ≥ 60% | ❌ |
| Generatsiya | Answer correctness | 1,69 | ≥ 1,6 | ✅ |
| Generatsiya | Faithfulness | 97,2% | ≥ 95% | ✅ |
| Manba | Citation accuracy | 81,2% | ≥ 90% | ❌ (prompt tuzatildi, quyida) |
| Manba | To'qilgan manba | 0 | 0 | ✅ |
| Manba | Eskirgan norma | 0 | 0 | ✅ |
| Xavfsizlik | No-answer | 9/10 | ≥ 9/10 | ✅ |
| Operatsion | Latency p50 / p95 | 6,0 s / 10,9 s | ≤ 5 s / ≤ 12 s | ❌ / ✅ |
| Operatsion | Xato ulushi | 0% | ≤ 2% | ✅ |

**Bu raqamlarni qanday o'qish kerak.** Savollarni LLM moddalarning o'zidan yozgan. Unga moddaning so'zlarini takrorlamaslik buyurilgan, baribir bunday savollar odam yozgan savollardan osonroq. Shuning uchun natijani **yuqori chegara** deb o'qing. Javob to'g'riligi, faithfulness, citation accuracy va context precision'ni LLM-hakam baholaydi. Hakam ham `gpt-6-luna`, ya'ni javob bergan model bilan bir xil, shuning uchun o'z javobiga yumshoqroq baho berishi mumkin.

## Fayllar

| Fayl | Nima qiladi |
|---|---|
| `eval/testset.csv` | 100 savol, ustozning CSV formatida (+ `language` ustuni) |
| `scripts/make_testset.py` | To'plamni yaratadi: kategoriyalar va har bir batch'ga aralash taqsimlash |
| `scripts/run_test.py` | Tester tomoni: 10 batch'ni ketma-ket yuboradi, `run_id/bNN.json` saqlaydi, 202 / 429 / 5xx'ni spetsifikatsiya bo'yicha qayta ishlaydi |
| `scripts/score_run.py` | 6-bo'lim metrikalari, kategoriyalar kesimi, xato sabablari |

```bash
uv run uvicorn lexrag.api.app:app --port 8000
uv run python scripts/run_test.py eval/testset.csv --run-id eval-2026-11-06
uv run python scripts/score_run.py eval/testset.csv data/runs/eval-2026-11-06
```

## Kategoriyalar (ustozning taqsimoti)

| Kategoriya | Soni | Hit@5 modda | To'g'rilik (0–2) |
|---|---|---|---|
| Faktik, bitta modda | 35 | 88,6% | 1,86 |
| Raqamlar | 15 | 80,0% | 1,67 |
| Bir nechta modda | 15 | 80,0% (ikkalasi birga: 33%) | **1,33** |
| O'zgargan normalar | 10 | 80,0% | 1,50 |
| Rus va kirill | 10 | 100% | 1,80 |
| Javobsiz | 10 | — | no_answer 9/10 |
| Noaniq / og'zaki | 5 | 100% | 1,80 |

Eng zaif joy ko'p moddali savollar. Ikkala kerakli modda top-5 ga birga faqat uchdan bir holatda kiradi. Faktik va ko'p tilli savollar eng yaxshi natija berdi.

## Metrikalar nimani o'lchaydi

- **Hit@5 (hujjat / modda):** kerakli hujjat yoki modda top-5 kontekst ichida bormi.
- **MRR:** to'g'ri modda nechanchi o'rinda turgani. 1-o'rin → 1, 2-o'rin → 0,5, topilmasa → 0. Barcha savollar bo'yicha o'rtacha olinadi.
- **Context precision:** top-5 kontekstdan nechtasi savolga tegishli.
- **Answer correctness:** kutilgan javob bilan solishtirish. 2 — to'g'ri, 1 — qisman, 0 — noto'g'ri. Javob berilmagan savol 0 deb hisoblanadi.
- **Faithfulness:** javobdagi da'volarning qanchasi kontekstlarda yozilgan. Bu "to'qimadimi?" degan savolga javob.
- **Citation accuracy:** keltirilgan iqtibos aynan o'sha gapni tasdiqlaydimi.

## Bajarilmagan uchta metrika

**Context precision 41%.** Savolga odatda bitta modda javob beradi, biz esa 5 ta kontekst qaytaramiz. Ideal tizimda ham bunda precision past chiqadi. Saqlangan natijalar ustida kesishni oflayn sinab ko'rdim, ya'ni reranker bahosi past kontekstlarni tashlab yuborishni. 60% ga yetish uchun shunchalik qattiq kesish kerak bo'ldiki, Hit@5 81% ga tushdi va LLM hozir iqtibos qilayotgan 23 ta kontekst yo'qoldi. Bu metrikani sun'iy ko'tarish bo'lardi, javoblar esa yomonlashardi. Shuning uchun kesmadim. Ustozning o'zi chegaralarni "birinchi run uchun taklif" deb yozgan.

**Citation accuracy 81%.** Ikki sabab topildi:
1. Bitta gapda bir nechta da'vo bor, iqtibos esa faqat bittasini tasdiqlaydi.
2. `[1]` belgisi gap boshida turibdi.

Prompt tuzatildi: bitta gapda bitta qoida bo'lishi, iqtibosda raqam, organ yoki shartning o'zi bo'lishi va belgining gap oxirida turishi talab qilinadi. Muammoli 20 ta savolda natija 45% dan **73%** ga ko'tarildi. To'liq qayta sinov hali qilinmagan.

**Latency p50 6,0 s.** Bunda 4 ta savol parallel ishlanadi va OpenRouter tarmog'ining tebranishi ±1 s qo'shadi. p95 esa maqsad ichida.

## Xato sabablari (6-bo'lim talabi)

Har bir noto'g'ri javob bitta sababga yozildi:

| Sabab | Soni | Ma'nosi | Nimani tuzatish kerak |
|---|---|---|---|
| retrieval | 12 | To'g'ri modda topilmagan | Qidiruv, chunking |
| generatsiya | 10 | Modda topilgan, javob noto'g'ri | Prompt, model |
| manba | 11 | Javob to'g'ri, iqtibos noto'g'ri | Iqtibos qoidalari (tuzatildi) |

## Ustoz so'rashi mumkin

**Testni qanday o'tkazdingiz?** Spetsifikatsiyadagidek: CSV'dagi 100 savol 10 ta batch'ga bo'lindi va ketma-ket yuborildi. Kutilgan javoblar API'ga yuborilmadi. Har bir javob `run_id/bNN.json` ga saqlandi, keyin 6-bo'lim metrikalari hisoblandi.

**Javob to'g'riligini kim baholadi?** LLM-hakam. U kutilgan javob bilan solishtirib 0, 1 yoki 2 baho qo'yadi. Cheklov: hakam javob bergan model bilan bir xil. Kuchliroq hakam modeli bizning budjetga sig'madi.

**Eng katta zaiflik nima?** Ko'p moddali savollar: to'g'rilik 1,33. Ikkala modda top-5 ga birga faqat 33% hollarda kiradi.

**Context precision nega past?** Savolga bitta modda javob berganda 5 ta kontekst ichida precision tabiiy ravishda past bo'ladi. Uni kontekstlarni kesib ko'tarish mumkin, lekin bu Hit@5 ni pasaytiradi va javob sifatini yomonlashtiradi. Raqamlar yuqorida.

**Test qancha turdi?** 100 savol ~$0,10, LLM-hakam esa yana ~$0,15.
