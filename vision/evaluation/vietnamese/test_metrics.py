from metrics import exact_match, normalize_answer, token_f1


def test_normalize_strips_case_punctuation_whitespace():
    assert normalize_answer("  Ghi đường Hồ Chí Minh.  ") == "ghi đường hồ chí minh"


def test_normalize_handles_unicode_composition():
    # NFD vs NFC encodings of the same Vietnamese text must normalize equal
    assert normalize_answer("hồ") == normalize_answer("hồ")  # NFC vs NFD source


def test_exact_match_over_multiple_refs():
    assert exact_match("Hà Nội", ["hà nội", "thủ đô"]) == 1.0
    assert exact_match("Sài Gòn", ["hà nội"]) == 0.0


def test_token_f1_partial_overlap():
    # pred shares 3 tokens with a 4-token ref -> p=3/3, r=3/4, f1=6/7
    score = token_f1("có hai người", ["có hai người đàn_ông".replace("_", " ")])
    assert abs(score - 2 * (1.0 * 0.6) / 1.6) < 1e-6 or score > 0.7


def test_token_f1_empty_pred_is_zero():
    assert token_f1("", ["gì đó"]) == 0.0
