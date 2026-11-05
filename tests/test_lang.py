import pytest

from lexrag.text.lang import detect_language


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("Yillik asosiy mehnat ta'tilining eng kam muddati qancha?", "uz-Latn"),
        ("Oʻzbekiston Respublikasi Konstitutsiyasi", "uz-Latn"),
        ("Йиллик асосий меҳнат таътилининг энг кам муддати қанча?", "uz-Cyrl"),
        ("Ер кодекси", "uz-Cyrl"),  # no ў қ ғ ҳ: the Uzbek ending -си decides
        ("Какой срок исковой давности установлен Гражданским кодексом?", "ru"),
        ("Сколько дней длится ежегодный отпуск?", "ru"),
        ("Трудовой кодекс", "ru"),
        ("Что такое БРВ?", "ru"),
    ],
)
def test_detect_language(text: str, lang: str) -> None:
    assert detect_language(text) == lang


def test_russian_questions_are_not_transliterated() -> None:
    from lexrag.text.normalize import normalize_query

    assert normalize_query("Какой срок исковой давности?") == "Какой срок исковой давности?"
    assert normalize_query("Меҳнат таътили қанча?") == "Mehnat taʼtili qancha?"
