# Faza 7: qamrovni kengaytirish — farmonlar (2026-11-07)

Indeksda Konstitutsiya, kodekslar va qonunlar bor edi: 385 ta hujjat. Endi unga **Prezident farmonlari va farmoyishlari** qo'shildi. Ish hozircha alohida lokal bazada (`lexrag_full`) qilingan. Production (Neon) o'zgarmadi.

## lex.uz'da nechta hujjat bor (o'lchangan)

| Turi | Jami | Amaldagi va mazmunli |
|---|---|---|
| Konstitutsiya, kodeks, qonun | 1 620 | 385 |
| Farmon + farmoyish | 2 046 | 980 |
| Qaror (Prezident, Vazirlar Mahkamasi va boshqalar) | 19 901 | 8 020 |
| Buyruq (vazirliklar) | 3 739 | 988 |
| **Jami milliy qonunchilik** | **28 508** | **~10 400** |

"Mazmunli" degani: o'zgartiruvchi, kuchini yo'qotgan deb topuvchi va ratifikatsiya hujjatlari chiqarib tashlangan (Faza 1 dagi `in_scope` qoidasi).

## Nima o'zgardi

1. **Discovery.** `lexrag discover --forms decree directive resolution order --out …` buyrug'i yangi turlarni yig'adi. Qonunlar ro'yxati `sort_id` ("Qonun hujjatlari" guruhi) bilan olinadi. Farmonlar uchun bu filtr qo'yilmaydi, aks holda ro'yxat bo'sh qaytadi.
2. **Band bo'yicha chunking.** Farmonda modda yo'q, qoidaning birligi **band**. Masalan: "3. Quyidagilar belgilansin: (i) …, (ii) …". Moddasi yo'q hujjatda har bir band bitta chunk bo'ladi va iqtibos "3-band" deb ko'rsatiladi. Kichik bandlar ((a), 1)) o'z bandi ichida qoladi. Ilovada raqamlash qaytadan 1 dan boshlansa, ID'ga unit raqami qo'shiladi (`u4-band-1`). Moddasi bor 350 ta qonunning chunk ID'lari o'zgarmadi.
3. **Eng yangi farmon qoidasi.** BHM'ni belgilagan har yilgi "Ish haqi, pensiyalar… oshirish to'g'risida" farmonlarining hammasi lex.uz'da hamon "amalda" deb turibdi. LLM'ga farmonning qabul qilingan sanasi beriladi. Promptdagi qoida: turli sanadagi farmonlar bir narsaga har xil qiymat bersa, eng yangisi olinadi.
4. **Iqtibos tekshiruvidagi xato tuzatildi.** Uzun iqtibosga LLM bizning `(4-qism)` belgilarimizni ham ko'chirardi va tekshiruv iqtibosni rad etardi. Endi bu belgilar solishtirishdan oldin olib tashlanadi. Run C'da ham 2 ta iqtibos shu sabab yo'qolgan edi.

## Natija

- 976 ta farmon yuklandi. 932 tasi parse bo'ldi, 46 tasining sahifa tuzilishi boshqacha.
- Ulardan **31 776 ta chunk** chiqdi. Embedding ~$0.10 turdi. `lexrag_full` bazasida jami 52 565 ta chunk, 881 MB.

| Sinov (reranker bilan) | Faqat qonunlar | Qonun + farmon |
|---|---|---|
| Qonunlar bo'yicha 90 savol: Hit@5 hujjat | 94,4% | 92,2% |
| Hit@5 modda | 86,7% | 84,4% |
| MRR | 0,735 | 0,703 |
| Farmonlar bo'yicha 21 savol: Hit@5 modda | — | 71,4% |

Qonun savollaridan 2 tasi tushib qoldi: farmonlardagi o'xshash bandlar "shovqin" qo'shdi, lekin uchala raqam ham talab chegarasidan yuqori qoldi. Farmon savollaridagi xatolarning ko'pi umumiy ijro bandlarida: "vazirlik ikki oy ichida taklif kiritsin" degan band yuzlab farmonda bir xil yoziladi. "Bazaviy hisoblash miqdori hozir qancha?" degan savolga endi to'g'ri javob chiqadi: 440 000 so'm (2026-yil 1-sentabrdan), eng yangi farmondan iqtibos bilan.

## Keyingisi va cheklovlar

- **Production.** 881 MB bepul Neon'ga sig'maydi, shuning uchun boshqa baza kerak.
- **Qarorlar.** ~9 000 ta hujjat. Taxminan 290 ming chunk va ~5 GB bo'ladi, embedding ~$1, yuklash 3–5 soat oladi.
- **Parse bo'lmaganlar.** 46 ta farmonning sahifa tuzilishini alohida ko'rib chiqish kerak.

## Ustoz so'rashi mumkin

**Farmonda modda yo'q, qanday bo'ldingiz?** Har bir band alohida chunk bo'ldi, iqtibos "3-band" deb ko'rsatiladi. Bandning kichik bandlari o'zi bilan birga qoladi.

**Eski farmon eski BHM'ni ko'rsatsa-chi?** lex.uz eski farmonni rasman bekor qilmagan, shuning uchun u "amalda" turibdi. LLM'ga farmonlarning sanasini beramiz, promptda esa "bir xil narsaga turli qiymat bo'lsa, eng yangisini ol" degan qoida bor.

**Hujjat ko'paysa, sifat tushmaydimi?** Biroz tushadi: qonun savollarida Hit@5 2 punktga pasaydi. Buni bir xil savollar to'plamini ikkala indeksda o'lchab bildik.
