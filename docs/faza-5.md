# Faza 5: ustozning testini o'zimizda takrorlash

Ustozning testi o'zimizda aynan takrorlandi: o'sha formatdagi 100 savol, o'sha tartibda yuborilgan 10 ta batch, 6-bo'limdagi o'sha metrikalar. Test uch marta o'tkazildi: har bir tuzatishdan keyin bittadan. Yakuniy run'da (C) **12 metrikadan 9 tasi** maqsadga yetdi.

| Qatlam | Metrika | Run A | Run B | **Run C** | Maqsad | |
|---|---|---|---|---|---|---|
| Retrieval | Hit@5 (hujjat) | 94,4% | 93,3% | **94,4%** | ≥ 90% | ✅ |
| Retrieval | Hit@5 (modda) | 86,7% | 85,6% | **86,7%** | ≥ 80% | ✅ |
| Retrieval | MRR | 0,727 | 0,724 | **0,735** | ≥ 0,70 | ✅ |
| Retrieval | Context precision | 41,4% | 40,5% | **39,5%** | ≥ 60% | ❌ |
| Generatsiya | Answer correctness | 1,69 | 1,61 | **1,71** | ≥ 1,6 | ✅ |
| Generatsiya | Faithfulness | 97,2% | 97,8% | **97,0%** | ≥ 95% | ✅ |
| Manba | Citation accuracy | 81,2% | 85,0% | **83,6%** | ≥ 90% | ❌ |
| Manba | To'qilgan manba | 0 | 0 | **0** | 0 | ✅ |
| Manba | Eskirgan norma | 0 | 0 | **0** | 0 | ✅ |
| Xavfsizlik | No-answer | 9/10 | 9/10 | **9/10** | ≥ 9/10 | ✅ |
| Operatsion | Latency p50 / p95 | 6,0 / 10,9 s | 6,1 / 12,7 s | **6,2 / 11,1 s** | ≤ 5 / ≤ 12 s | ❌ / ✅ |
| Operatsion | Xato ulushi | 0% | 0% | **0%** | ≤ 2% | ✅ |

Run'lar orasidagi o'zgarishlar:
- **A → B:** iqtiboslar uchun prompt kuchaytirildi: bitta gapda bitta qoida, iqtibosda qoidaning o'zi bo'lishi kerak. Citation accuracy ko'tarildi. Lekin javobi bor 7 ta savol rad etildi (A da 4 ta edi) va to'g'rilik pasaydi.
- **B → C:** sababi topildi. Bu bizning tekshiruvimizdagi xato edi. Model ikki paragrafni qamrab olgan iqtibos keltirgan, `find_quote` esa uni har bir paragraf ichida alohida qidirgan va to'g'ri iqtiboslarni rad etgan. Tuzatilgach rad etishlar yana 4 taga tushdi, to'g'rilik esa eng yuqori qiymatga (1,71) chiqdi.

Bir xil tizimning ikki run'i orasida ham hakam baholaydigan metrikalar ±2–3 punkt farq qiladi. Sabab: LLM javobi va hakam bahosi har safar biroz o'zgaradi. Shuning uchun citation accuracy'dagi 81–85% oralig'ini bitta raqam deb emas, diapazon deb o'qing.

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

**Citation accuracy 81–85%.** Ikki sabab topildi:
1. Bitta gapda bir nechta da'vo bor, iqtibos esa faqat bittasini tasdiqlaydi.
2. `[1]` belgisi gap boshida turibdi.

Prompt tuzatildi. Muammoli 20 ta savolda natija 45% dan 73% ga ko'tarildi, butun testda esa 81% dan 84–85% ga. 90% ga yetishning keyingi yo'li: hakam va javob beruvchi model sifatida kuchliroq model ishlatish yoki har bir da'voni alohida tekshirish. Ikkalasi ham qo'shimcha xarajat talab qiladi.

**Latency p50 6,0 s.** Bunda 4 ta savol parallel ishlanadi va OpenRouter tarmog'ining tebranishi ±1 s qo'shadi. p95 esa maqsad ichida.

## Xato sabablari (6-bo'lim talabi, run C)

Har bir noto'g'ri javob bitta sababga yozildi:

| Sabab | Soni | Ma'nosi | Nimani tuzatish kerak |
|---|---|---|---|
| retrieval | 11 | To'g'ri modda topilmagan | Qidiruv, chunking |
| generatsiya | 10 | Modda topilgan, javob noto'g'ri | Prompt, model |
| manba | 10 | Javob to'g'ri, iqtibos noto'g'ri | Iqtibos qoidalari |

## Sinab ko'rilgan va rad etilgan g'oya: qo'shni moddalar

Ko'p moddali savollarda ko'pincha juftlikning bittasi topiladi, qo'shnisi esa topilmaydi (67-modda bor, 68 yo'q). G'oya shunday edi: yuqori nomzodlarning qo'shni moddalarini ham reranker'ga berish. Natija: ko'p moddali savollarda +1 (6/15), umumiy Hit@5 da −1, qidiruv esa ~1 soniya sekinlashdi. Foydasi yo'q, shuning uchun standart holatda o'chirilgan (`LEXRAG_NEIGHBOURS_OF=0`). Kod va testi qoldi.

## Ustoz so'rashi mumkin

**Testni qanday o'tkazdingiz?** Spetsifikatsiyadagidek: CSV'dagi 100 savol 10 ta batch'ga bo'lindi va ketma-ket yuborildi. Kutilgan javoblar API'ga yuborilmadi. Har bir javob `run_id/bNN.json` ga saqlandi, keyin 6-bo'lim metrikalari hisoblandi.

**Javob to'g'riligini kim baholadi?** LLM-hakam. U kutilgan javob bilan solishtirib 0, 1 yoki 2 baho qo'yadi. Cheklov: hakam javob bergan model bilan bir xil. Kuchliroq hakam modeli bizning budjetga sig'madi.

**Eng katta zaiflik nima?** Ko'p moddali savollar: to'g'rilik 1,33. Ikkala modda top-5 ga birga faqat 33% hollarda kiradi. Qo'shni moddalarni qo'shish sinab ko'rildi, lekin yordam bermadi.

**Natijalar barqarormi?** Retrieval metrikalari deyarli bir xil chiqadi, chunki qidiruv deterministik. Hakam baholaydigan metrikalar run'dan run'ga ±2–3 punkt tebranadi, shuning uchun uchta run qilindi.

**Context precision nega past?** Savolga bitta modda javob berganda 5 ta kontekst ichida precision tabiiy ravishda past bo'ladi. Uni kontekstlarni kesib ko'tarish mumkin, lekin bu Hit@5 ni pasaytiradi va javob sifatini yomonlashtiradi. Raqamlar yuqorida.

**Test qancha turdi?** Bitta to'liq run hakam bilan birga ~$0,28: savollarga javob ~$0,10, LLM-hakam ~$0,15–0,18 (hakam "medium" darajada fikrlaydi).

## Run C'dan keyingi tuzatish (2026-11-06)

Uchta bajarilmagan metrika uchun qilingan o'zgarishlar:
- **Prompt:** ikki gap bitta iqtibosni bo'lishmaydi, har bir gap o'z iqtibosini oladi. Savolda so'ralmagan qoidalar javobga qo'shilmaydi.
- **OpenRouter routing:** `provider.sort = "latency"`. Bir xil model bir nechta hostda turadi, endi eng tez javob beradigani tanlanadi.
- **Parallel qidiruv:** query embedding tarmoqdan kelguncha full-text qidiruv bazada bajarilib turadi.

Pul tejash uchun 100 savollik to'liq test o'rniga 15 ta qiyin savolda tekshirildi:
- citation accuracy 61% → 90%;
- LLM javob vaqti 4,0 s → 1,8 s;
- correctness 1,71 → 1,57: ko'p qismli savollarda javob qisqaroq chiqdi.

Retrieval o'zgarmadi, chunki top-5 aynan bir xil. Kichik to'plamda hakam bahosi ancha tebranadi, aniq raqam uchun to'liq run kerak.

Context precision ataylab o'zgartirilmadi. Spetsifikatsiya uni top-5 bo'yicha o'lchaydi, faqat ishlatilgan kontekstlarni qaytarish esa metrikani aylanib o'tish bo'lardi.
