"""
Tests para normalize_text + metricas con referencia (F1, EM, Accuracy).

Normalizacion (case, acentos, dashes, puntuacion) se testea una vez en
normalize_text. Las metricas testean logica propia: formula F1, overlap,
EM estricto, accuracy con labels.
"""
from shared.metrics import (
    normalize_text, _remove_accents, _dashes_to_spaces, tokenize_text,
    f1_score, exact_match, accuracy,
)


# =================================================================
# normalize_text
# =================================================================

def test_normalize_basic():
    assert normalize_text("  Hello   World  ") == "hello world"
    assert normalize_text("") == ""
    assert normalize_text(None) == ""


def test_remove_accents():
    assert _remove_accents("café") == "cafe"
    assert _remove_accents("über") == "uber"
    assert _remove_accents("niño") == "nino"
    assert _remove_accents("résumé") == "resume"
    assert _remove_accents("hello") == "hello"


def test_dashes_to_spaces():
    """Hyphen-minus, en-dash, em-dash, minus sign → espacio."""
    assert _dashes_to_spaces("1969-1974").strip() == "1969 1974"
    assert _dashes_to_spaces("1969\u20131974").strip() == "1969 1974"
    assert _dashes_to_spaces("1969\u20141974").strip() == "1969 1974"
    assert _dashes_to_spaces("1969\u22121974").strip() == "1969 1974"


def test_tokenize():
    assert tokenize_text("1969-1974") == ["1969", "1974"]
    assert tokenize_text("!!??...") == []


def test_normalize_remove_articles():
    assert normalize_text(
        "The cat and a dog", remove_articles=True, language="en"
    ) == "cat and dog"
    assert normalize_text(
        "El gato y un perro", remove_articles=True, language="es"
    ) == "gato y perro"


# =================================================================
# F1 Score
# =================================================================

def test_f1_boundaries():
    """Identico → 1.0, sin overlap → 0.0, empty → 0.0."""
    r1 = f1_score("the cat sat", "the cat sat")
    assert r1.value == 1.0
    assert r1.details["precision"] == 1.0

    assert f1_score("alpha beta", "gamma delta").value == 0.0
    assert f1_score("", "hello").value == 0.0
    assert f1_score("hello", "").value == 0.0


def test_f1_partial_overlap():
    """Verifica formula con overlap parcial en ambas direcciones."""
    # generated superset: precision = 2/6, recall = 2/2 → F1 = 0.5
    r = f1_score("the cat sat on the mat", "the cat")
    assert abs(r.value - 0.5) < 0.01
    assert r.details["recall"] == 1.0

    # generated subset: precision = 2/2, recall = 2/6 → F1 = 0.5
    r2 = f1_score("the cat", "the cat sat on the mat")
    assert abs(r2.value - 0.5) < 0.01
    assert r2.details["precision"] == 1.0


def test_f1_duplicate_tokens():
    """Counter intersection respeta frecuencias."""
    r = f1_score("the the the", "the the")
    assert abs(r.value - 0.8) < 0.01
    assert r.details["num_common_tokens"] == 2


def test_f1_normalization():
    """Case, acentos, dashes no afectan match."""
    assert f1_score("YES", "yes").value == 1.0
    assert f1_score("café latte", "cafe latte").value == 1.0
    # HotpotQA real: dashes se separan en tokens
    r = f1_score("1969-1974", "1969 until 1974")
    assert r.value > 0.7


# =================================================================
# Exact Match
# =================================================================

def test_em_boundaries():
    """Identico → 1.0, diferente → 0.0, empty → 0.0."""
    r = exact_match("hello world", "hello world")
    assert r.value == 1.0
    assert r.details["is_match"] is True

    assert exact_match("hello", "world").value == 0.0
    assert exact_match("", "hello").value == 0.0


def test_em_normalization():
    """Case, puntuacion, acentos, dashes, espacios: todos normalizados."""
    assert exact_match("YES", "yes").value == 1.0
    assert exact_match("hello!", "hello").value == 1.0
    assert exact_match("café", "cafe").value == 1.0
    assert exact_match("1969-1974", "1969 1974").value == 1.0
    assert exact_match("  hello   world  ", "hello world").value == 1.0


def test_em_without_normalize():
    """Sin normalizacion, case y puntuacion importan."""
    assert exact_match("Hello!", "hello", normalize=False).value == 0.0


# =================================================================
# Accuracy
# =================================================================

def test_accuracy_basic():
    """Match, mismatch, case insensitive, puntuacion."""
    assert accuracy("yes", "yes").value == 1.0
    assert accuracy("yes", "no").value == 0.0
    assert accuracy("YES", "yes").value == 1.0
    assert accuracy("Yes.", "yes").value == 1.0
    assert accuracy("", "yes").value == 0.0


def test_accuracy_extra_text_no_match():
    """Accuracy es EM, no 'contains'. Texto extra no matchea."""
    assert accuracy("Yes, the answer is correct", "yes").value == 0.0


def test_accuracy_valid_labels():
    r1 = accuracy("yes", "yes", valid_labels=["yes", "no"])
    assert r1.value == 1.0
    assert r1.details["is_valid_label"] is True

    r2 = accuracy("maybe", "maybe", valid_labels=["yes", "no"])
    assert r2.value == 1.0
    assert r2.details["is_valid_label"] is False
