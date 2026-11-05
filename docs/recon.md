# Lex.uz razvedkasi (2026-10-30)

Scraper yozishdan oldin saytning tuzilishi o'rganildi. Barcha qarorlar quyidagi faktlarga tayanadi.

## Sayt

- ASP.NET WebForms. Sahifalar **serverda render qilinadi**, ya'ni JS kerak emas va brauzer avtomatlashtirish (Playwright) ishlatilmaydi.
- `robots.txt` yo'q (`/Pages/404.aspx` ga yo'naltiradi). Shunga qaramay **1 so'rov/soniya** cheklovini o'zimiz qo'yganmiz, chunki bu davlat xizmati.
- Mavjud bo'lmagan hujjat `302 → /Pages/404.aspx` qaytaradi va bu sahifa `200` bilan keladi. Shuning uchun status kodini emas, yakuniy URL'ni tekshiramiz.

## Hujjat ID'lari va tillar

| ID | Ma'nosi |
|---|---|
| `6445145` | Konstitutsiya, kirill yozuvidagi asl matn |
| `-6445145` | **o'sha hujjatning lotin yozuvidagi varianti** (manfiy ID) |
| `6445147`, `6451070` | rus va ingliz tarjimalari (alohida ID) |

Variantlarning ID'lari hujjat sahifasidagi til tugmalarida bor (`openUrl('/uz/docs/…')`). Ular `Document.twins` maydoniga yoziladi.

ID belgisi (musbat yoki manfiy) yozuvni har doim to'g'ri ko'rsatmaydi: 2026-yilgi yangi hujjatlarda musbat ID ham lotincha bo'lishi mumkin. Shuning uchun yozuv **matn bo'yicha aniqlanadi**: kirill harflari 50% dan ko'p bo'lsa, hujjat kirillda deb hisoblanadi.

## Hujjat sahifasining tuzilishi

Matn tekis bloklar ketma-ketligidan iborat. Har bir blokning o'z **doimiy element ID'si** bor:

```html
<div class="CLAUSE_DEFAULT lx_elem"> …UI tugmalari… <div name="-174405" id="-174405">1-modda. …</div></div>
```

| Blok sinfi | Nima | Biz nima qilamiz |
|---|---|---|
| `TEXT_HEADER_DEFAULT` | Qism / Bo'lim / Bob / § | Sarlavhalar stekiga qo'shamiz → `Unit.path` |
| `CLAUSE_DEFAULT` | `18<sup>1</sup>-modda. Nomi` | Yangi modda; raqam `18¹` ko'rinishida saqlanadi (181-moddadan farqli) |
| `ACT_TEXT`, `TABLE_STD` | Modda matni, jadval | `Paragraph` (o'z elem_id'si bilan) |
| `ACT_TITLE`, `ACT_FORM`, `ACT_ESSENTIAL_ELEMENTS*`, `SIGNATURE` | Rekvizitlar | Metadata, matnga kirmaydi |
| `CHANGES_ORIGINS` (lx_no_select) | "(… Qonuni tahririda — …)" | `Unit.amendments`, embedding'ga kirmaydi |
| `COMMENT` + `COMMENTLEXUZ` | LexUZ sharhi, "Qarang: …" | `Unit.notes` |
| `COMMENT` (`edi-…` id) | "Oldingi tahrirga qarang" | Tashlab yuboriladi |
| `INDEXES_ON_REF` | OKOZ tematik klassifikatori | `Unit.topics`, keyinchalik filtr uchun |

Element ID'si tufayli iqtibos **aniq paragrafga** havola beradi: `https://lex.uz/uz/docs/-111453#-174405`.

Boshqa metadata:
- `docHeader`: rekvizitlar qatori va "Kuchga kirish sanasi".
- `?ONDATE2=dd.mm.yyyy`: redaksiyalar ro'yxati. Kelajakdagi redaksiyalar ham bor (masalan, `01.01.2028`).

## Ro'yxat (discovery)

`/uz/search/nat?lang=4&sort_id=3975&form_id=…`

| form_id | Turi |
|---|---|
| 4131 | Konstitutsiya |
| 3964 | Kodeks |
| 3968 | Qonun |

- Har sahifada 20 qator. Keyingi sahifa `__doPostBack('ucFoundActsControl$LinkButton1')` orqali ochiladi: `__VIEWSTATE` maydonlari bilan POST yuboriladi. Oxirgi sahifada bu havola yo'q.
- Holat ikonkasi: `status_code_y` (✓) amalda; `status_code_r`, `status_code_n` (✕) amalda emas.
- `lang=4` lotin yozuvidagi variantlarni qaytaradi.

## Inkremental yangilanish

- `/uz/rss`: oxirgi hujjatlar. Har birida qabul qilingan va kuchga kirish sanalari bor. Kunlik DAG uchun manba.
- Muammo: kodeksga o'zgartirish kiritilganda RSS'da o'zgartiruvchi qonun chiqadi, kodeksning o'zi chiqmaydi. Yechim (Faza 3): RSS'dagi o'zgartiruvchi qonundan maqsadli hujjatlarni aniqlash, xavfsizlik tarmog'i sifatida haftalik to'liq qayta tekshirish. Har bir hujjatning `content_hash`'i o'zgarmagan bo'lsa, qayta embedding qilinmaydi.

## Qamrov

Discovery natijasi: 1 618 ta hujjat (Konstitutsiya, kodekslar, qonunlar).

Qamrovga faqat **amaldagi, mazmunli** hujjatlar kiradi. Quyidagilar chiqarib tashlanadi:
- "…ga o'zgartish va qo'shimchalar kiritish to'g'risida";
- "…kuchini yo'qotgan deb topish";
- ratifikatsiya, xalqaro shartnomaga qo'shilish va kodeksni tasdiqlash to'g'risidagi qonunlar.

Sabab: ularning mazmuni asosiy hujjatning konsolidatsiyalangan (amaldagi) matnida allaqachon bor. Ularni indekslash qidiruvga deyarli takroriy shovqin qo'shadi.

## Transliteratsiya sifati

Konstitutsiyaning kirillcha asl matnini (`6445145`) bizning `cyr_to_lat` funksiyamiz bilan lotinga o'girib, rasmiy lotin variant (`-6445145`) bilan so'zma-so'z solishtirdik:

| Versiya | Moslik |
|---|---|
| Harf-ma-harf qoidalar | 99,82% |
| + `ъ` + yotlashgan unli qoidasi (`субъект → subyekt`), o'zlashma so'zlar (`бюджет → budjet`, `октябрь → oktabr`) | **99,99%** (10 729 so'zdan 1 ta farq) |

Bu `tests/test_parse.py::test_transliteration_matches_official_latin_text` testida doimiy tekshiriladi.

## Raqobatchi

`ai.lex.uz` — rasmiy AI yordamchi. Bizning farqimiz: modda va paragraf darajasidagi tekshiriladigan iqtiboslar, ochiq va avtomatik yangilanadigan pipeline, raqamlar bilan o'lchangan sifat (Faza 5).
