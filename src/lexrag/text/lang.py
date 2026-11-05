"""Which of the three question languages the teacher's spec accepts: uz-Latn, uz-Cyrl, ru.

Uzbek Cyrillic and Russian share most letters, so the alphabet alone is not enough:
  ў қ ғ ҳ  exist only in Uzbek Cyrillic;
  ы щ      exist only in Russian;
and when neither shows up (short questions), frequent words and endings decide.
The language matters twice: Uzbek Cyrillic is transliterated to Latin before search (the
index is Latin), Russian is not (transliterating it produces nonsense like "Kakoy srok").
"""

import re
from typing import Literal

from lexrag.text.normalize import cyrillic_ratio

Language = Literal["uz-Latn", "uz-Cyrl", "ru"]

_WORD_RE = re.compile(r"[а-яёўқғҳ]+")
_UZ_WORDS = {
    "ва", "бу", "учун", "билан", "нима", "ким", "қандай", "қанча", "мумкин", "эмас", "бор",
    "йўқ", "агар", "керак", "қонун", "қонуни", "кодекси", "модда", "моддаси", "ҳуқуқ",
}  # fmt: skip
_RU_WORDS = {
    "и", "в", "во", "на", "не", "что", "как", "какой", "какая", "какие", "каков", "для", "по",
    "с", "со", "о", "об", "ли", "или", "это", "есть", "при", "срок", "статья", "кодекс",
    "закон", "право", "может", "можно", "если", "кто", "сколько",
}  # fmt: skip
_UZ_ENDINGS = ("нинг", "ларни", "лари", "дан", "лар", "си", "да", "га", "ни")
_RU_ENDINGS = ("ого", "его", "ому", "ыми", "ими", "ая", "яя", "ое", "ые", "ий", "ый", "ов", "ть")


def detect_language(text: str) -> Language:
    if cyrillic_ratio(text) <= 0.3:
        return "uz-Latn"
    low = text.lower()
    if any(ch in low for ch in "ўқғҳ"):
        return "uz-Cyrl"
    if any(ch in low for ch in "ыщ"):
        return "ru"
    words = _WORD_RE.findall(low)
    uz = sum(w in _UZ_WORDS for w in words) + sum(
        len(w) > 3 and w.endswith(_UZ_ENDINGS) for w in words
    )
    ru = sum(w in _RU_WORDS for w in words) + sum(
        len(w) > 3 and w.endswith(_RU_ENDINGS) for w in words
    )
    return "uz-Cyrl" if uz > ru else "ru"
