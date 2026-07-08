import unicodedata

from metrics import exact_match, normalize_answer, token_f1


def test_normalize_strips_case_punctuation_whitespace():
    assert normalize_answer("  Ghi đường Hồ Chí Minh.  ") == "ghi đường hồ chí minh"


def test_normalize_handles_unicode_composition():
    # NFD (combining diacritic) vs NFC (precomposed) encodings of the same
    # Vietnamese text must normalize equal.
    nfc = "hồ"
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfc != nfd  # sanity: the two source encodings really do differ
    assert normalize_answer(nfc) == normalize_answer(nfd)


def test_exact_match_over_multiple_refs():
    assert exact_match("Hà Nội", ["hà nội", "thủ đô"]) == 1.0
    assert exact_match("Sài Gòn", ["hà nội"]) == 0.0


def test_token_f1_partial_overlap():
    # pred shares 3 tokens with a 5-token ref -> p=3/3, r=3/5, f1=2*1*0.6/1.6=0.75
    score = token_f1("có hai người", ["có hai người đàn ông"])
    assert abs(score - 0.75) < 1e-6


def test_token_f1_empty_pred_is_zero():
    assert token_f1("", ["gì đó"]) == 0.0
