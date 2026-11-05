import pytest

from lexrag.text.normalize import (
    OKINA,
    TUTUQ,
    clean_whitespace,
    cyr_to_lat,
    cyrillic_ratio,
    normalize_apostrophes,
    normalize_query,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("o'zbek", f"o{OKINA}zbek"),
        ("O`zbekiston", f"O{OKINA}zbekiston"),
        ("g‘alla", f"g{OKINA}alla"),
        ("boʼlim", f"bo{OKINA}lim"),  # wrong mark after o is still the oʻ letter
        ("tog' va bog'", f"tog{OKINA} va bog{OKINA}"),  # end of word
        ("ma'lumot", f"ma{TUTUQ}lumot"),
        ("ta’lim", f"ta{TUTUQ}lim"),
        ("maʻno", f"ma{TUTUQ}no"),  # wrong mark after a consonant/vowel is a tutuq
        ("'Qonun'", "'Qonun'"),  # quotes around a word are not letters
        ("O‘RQ-1184", f"O{OKINA}RQ-1184"),
    ],
)
def test_normalize_apostrophes(raw: str, expected: str) -> None:
    assert normalize_apostrophes(raw) == expected


@pytest.mark.parametrize(
    ("cyr", "lat"),
    [
        ("Ўзбекистон Республикаси Конституцияси", "Oʻzbekiston Respublikasi Konstitutsiyasi"),
        ("БИРИНЧИ БЎЛИМ", "BIRINCHI BOʻLIM"),
        ("Ер кодекси", "Yer kodeksi"),
        ("ЕР КОДЕКСИ", "YER KODEKSI"),
        ("ҳуқуқ ва эркинликлар", "huquq va erkinliklar"),
        ("маъмурий жавобгарлик", "maʼmuriy javobgarlik"),
        ("ғалла, шахс, чора", "gʻalla, shaxs, chora"),
        ("Ёшлар, ЁШЛАР", "Yoshlar, YOSHLAR"),
        ("цех", "sex"),
        ("субъект, объект", "subyekt, obyekt"),
        ("Давлат бюджети, ОКТЯБРЬ", "Davlat budjeti, OKTABR"),
        ("1-модда.", "1-modda."),
        ("Latin text stays", "Latin text stays"),
    ],
)
def test_cyr_to_lat(cyr: str, lat: str) -> None:
    assert cyr_to_lat(cyr) == lat


def test_cyrillic_ratio() -> None:
    assert cyrillic_ratio("Ўзбекистон") == 1.0
    assert cyrillic_ratio("Oʻzbekiston") == 0.0
    assert cyrillic_ratio("123 —") == 0.0


def test_clean_whitespace_keeps_lines_drops_blank() -> None:
    assert clean_whitespace("  a  b \n\n\t c  ") == "a b\nc"


def test_query_normalization_makes_scripts_meet() -> None:
    # A Cyrillic query and a sloppy-apostrophe Latin query land on the same string.
    assert (
        normalize_query("меҳнат таътили") == normalize_query("mehnat ta'tili") == "mehnat taʼtili"
    )
