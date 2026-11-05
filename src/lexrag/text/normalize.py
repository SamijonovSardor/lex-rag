"""Uzbek text normalization.

Two problems make naive search fail on Uzbek legal text:

1. Two scripts. Older acts exist only in Cyrillic, users type queries in either.
   We store everything in Latin and transliterate Cyrillic (documents *and* queries).
2. The "apostrophe zoo". The letters oʻ / gʻ and the tutuq belgisi (maʼno) are typed as
   ' ` ‘ ’ ʻ ʼ … — six spellings of one word never match each other.
   We fold them to the official characters: U+02BB (ʻ) after o/g, U+02BC (ʼ) elsewhere.
"""

import re
import unicodedata

OKINA = "ʻ"  # ʻ  — oʻ, gʻ
TUTUQ = "ʼ"  # ʼ  — maʼno, taʼlim
_APOSTROPHES = "'`‘’ʻʼʹ′´"

_OG_APOS_RE = re.compile(rf"([OoGg])[{_APOSTROPHES}]")
_INNER_APOS_RE = re.compile(rf"(?<=[^\WOoGg])[{_APOSTROPHES}](?=\w)")


def normalize_apostrophes(text: str) -> str:
    text = _OG_APOS_RE.sub(rf"\1{OKINA}", text)  # boʻlim, togʻ (also at the end of a word)
    # Whatever is left between two letters is a tutuq. Quotes around words stay untouched.
    return _INNER_APOS_RE.sub(TUTUQ, text)


# --- Cyrillic → Latin (official 1995 alphabet) ------------------------------------------

_CYR2LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ё": "yo", "ж": "j", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p",
    "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "x", "ч": "ch", "ш": "sh",
    "щ": "sh", "ъ": TUTUQ, "ы": "i", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    "ў": "o" + OKINA, "қ": "q", "ғ": "g" + OKINA, "ҳ": "h",
}  # fmt: skip
_CYR_VOWELS = set("аеёиоуэюяў")
_CYR_LETTER_RE = re.compile(r"[Ѐ-ӿ]")
# Loanwords whose official Latin spelling does not follow letter-by-letter rules.
_LOANWORD_STEMS = {"бюджет": "budjet", "сентябр": "sentabr", "октябр": "oktabr"}
_LOANWORD_RE = re.compile("|".join(_LOANWORD_STEMS), re.IGNORECASE)


def _loanword(m: re.Match[str]) -> str:
    word, lat = m[0], _LOANWORD_STEMS[m[0].lower()]
    if word.isupper():
        return lat.upper()
    return lat.capitalize() if word[0].isupper() else lat


def _is_cyr_letter(ch: str) -> bool:
    return bool(ch) and "Ѐ" <= ch <= "ӿ"


def _translit_char(prev: str, ch: str, nxt: str) -> str:
    low = ch.lower()
    if low == "е":
        # ye at the start of a word and after a vowel / hard or soft sign: Ер → Yer, поэт stays e
        out = "ye" if (not prev.isalpha() or prev.lower() in _CYR_VOWELS or prev in "ъьЪЬ") else "e"
    elif low == "ц":
        # Конституция → Konstitutsiya, but цех → sex
        out = "ts" if prev.lower() in _CYR_VOWELS else "s"
    elif low == "ъ" and nxt.lower() in "еёюя":
        out = ""  # субъект → subyekt: the iotated vowel already brings the y
    else:
        out = _CYR2LAT.get(low)
        if out is None:
            return ch
    if not ch.isupper() or not out:
        return out
    # ШАҲАР → SHAHAR, but Шаҳар → Shahar
    whole_word_upper = nxt.isupper() or (prev.isupper() and not nxt.isalpha())
    return out.upper() if whole_word_upper else out[0].upper() + out[1:]


def cyr_to_lat(text: str) -> str:
    if not _CYR_LETTER_RE.search(text):
        return text
    text = _LOANWORD_RE.sub(_loanword, text)
    out = []
    for i, ch in enumerate(text):
        if not _is_cyr_letter(ch):
            out.append(ch)
            continue
        prev = text[i - 1] if i else ""
        nxt = text[i + 1] if i + 1 < len(text) else ""
        out.append(_translit_char(prev, ch, nxt))
    return "".join(out)


def cyrillic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if _is_cyr_letter(c)) / len(letters)


# --- whitespace & full pipeline ---------------------------------------------------------

_SPACES_RE = re.compile(r"[ \t  -​  　]+")


def clean_whitespace(text: str) -> str:
    lines = (_SPACES_RE.sub(" ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def normalize_text(text: str, *, transliterate: bool = False) -> str:
    """Canonical form stored in the index. Also applied to user queries."""
    text = unicodedata.normalize("NFC", text)
    if transliterate:
        text = cyr_to_lat(text)
    return clean_whitespace(normalize_apostrophes(text))


def normalize_query(text: str, language: str | None = None) -> str:
    """Uzbek Cyrillic → Latin like the index; Russian stays Cyrillic (bge-m3 reads it)."""
    from lexrag.text.lang import detect_language  # lang imports this module

    language = language or detect_language(text)
    return normalize_text(text, transliterate=language == "uz-Cyrl")
