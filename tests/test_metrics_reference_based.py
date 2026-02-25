"""
Tests para TextNormalizer + ReferenceBasedMetrics (F1, EM, Accuracy).

Normalizacion (case, acentos, dashes, puntuacion) se testea una vez en
TextNormalizer. Las metricas testean logica propia: formula F1, overlap,
EM estricto, accuracy con labels.
"""
from shared.metrics import TextNormalizer, ReferenceBasedMetrics


# =================================================================
# TextNormalizer
# =================================================================

def test_normalize_basic():
    assert TextNormalizer.normalize("  Hello   World  ") == "hello world"
    assert TextNormalizer.normalize("") == ""
    assert TextNormalizer.normalize(None) == ""


def test_remove_accents():
    assert TextNormalizer._remove_accents("café") == "cafe"
    assert TextNormalizer._remove_accents("über") == "uber"
    assert TextNormalizer._remove_accents("niño") == "nino"
    assert TextNormalizer._remove_accents("résumé") == "resume"
    assert TextNormalizer._remove_accents("hello") == "hello"


def test_dashes_to_spaces():
    """Hyphen-minus, en-dash, em-dash, minus sign → espacio."""
    assert TextNormalizer._dashes_to_spaces("1969-1974").strip() == "1969 1974"
    assert TextNormalizer._dashes_to_spaces("1969\u20131974").strip() == "1969 1974"
    assert TextNormalizer._dashes_to_spaces("1969\u20141974").strip() == "1969 1974"
    assert TextNormalizer._dashes_to_spaces("1969\u22121974").strip() == "1969 1974"


def test_tokenize():
    assert TextNormalizer.tokenize("1969-1974") == ["1969", "1974"]
    assert TextNormalizer.tokenize("!!??...") == []


def test_normalize_remove_articles():
    assert TextNormalizer.normalize(
        "The cat and a dog", remove_articles=True, language="en"
    ) == "cat and dog"
    assert TextNormalizer.normalize(
        "El gato y un perro", remove_articles=True, language="es"
    ) == "gato y perro"


# =================================================================
# F1 Score
# =================================================================

def test_f1_boundaries():
    """Identico → 1.0, sin overlap → 0.0, empty → 0.0."""
    r1 = ReferenceBasedMetrics.f1_score("the cat sat", "the cat sat")
    assert r1.value == 1.0
    assert r1.details["precision"] == 1.0

    assert ReferenceBasedMetrics.f1_score("alpha beta", "gamma delta").value == 0.0
    assert ReferenceBasedMetrics.f1_score("", "hello").value == 0.0
    assert ReferenceBasedMetrics.f1_score("hello", "").value == 0.0


def test_f1_partial_overlap():
    """Verifica formula con overlap parcial en ambas direcciones."""
    # generated superset: precision = 2/6, recall = 2/2 → F1 = 0.5
    r = ReferenceBasedMetrics.f1_score("the cat sat on the mat", "the cat")
    assert abs(r.value - 0.5) < 0.01
    assert r.details["recall"] == 1.0

    # generated subset: precision = 2/2, recall = 2/6 → F1 = 0.5
    r2 = ReferenceBasedMetrics.f1_score("the cat", "the cat sat on the mat")
    assert abs(r2.value - 0.5) < 0.01
    assert r2.details["precision"] == 1.0


def test_f1_duplicate_tokens():
    """Counter intersection respeta frecuencias."""
    r = ReferenceBasedMetrics.f1_score("the the the", "the the")
    assert abs(r.value - 0.8) < 0.01
    assert r.details["num_common_tokens"] == 2


def test_f1_normalization():
    """Case, acentos, dashes no afectan match."""
    assert ReferenceBasedMetrics.f1_score("YES", "yes").value == 1.0
    assert ReferenceBasedMetrics.f1_score("café latte", "cafe latte").value == 1.0
    # HotpotQA real: dashes se separan en tokens
    r = ReferenceBasedMetrics.f1_score("1969-1974", "1969 until 1974")
    assert r.value > 0.7


# =================================================================
# Exact Match
# =================================================================

def test_em_boundaries():
    """Identico → 1.0, diferente → 0.0, empty → 0.0."""
    r = ReferenceBasedMetrics.exact_match("hello world", "hello world")
    assert r.value == 1.0
    assert r.details["is_match"] is True

    assert ReferenceBasedMetrics.exact_match("hello", "world").value == 0.0
    assert ReferenceBasedMetrics.exact_match("", "hello").value == 0.0


def test_em_normalization():
    """Case, puntuacion, acentos, dashes, espacios: todos normalizados."""
    assert ReferenceBasedMetrics.exact_match("YES", "yes").value == 1.0
    assert ReferenceBasedMetrics.exact_match("hello!", "hello").value == 1.0
    assert ReferenceBasedMetrics.exact_match("café", "cafe").value == 1.0
    assert ReferenceBasedMetrics.exact_match("1969-1974", "1969 1974").value == 1.0
    assert ReferenceBasedMetrics.exact_match("  hello   world  ", "hello world").value == 1.0


def test_em_without_normalize():
    """Sin normalizacion, case y puntuacion importan."""
    assert ReferenceBasedMetrics.exact_match("Hello!", "hello", normalize=False).value == 0.0


# =================================================================
# Accuracy
# =================================================================

def test_accuracy_basic():
    """Match, mismatch, case insensitive, puntuacion."""
    assert ReferenceBasedMetrics.accuracy("yes", "yes").value == 1.0
    assert ReferenceBasedMetrics.accuracy("yes", "no").value == 0.0
    assert ReferenceBasedMetrics.accuracy("YES", "yes").value == 1.0
    assert ReferenceBasedMetrics.accuracy("Yes.", "yes").value == 1.0
    assert ReferenceBasedMetrics.accuracy("", "yes").value == 0.0


def test_accuracy_extra_text_no_match():
    """Accuracy es EM, no 'contains'. Texto extra no matchea."""
    assert ReferenceBasedMetrics.accuracy("Yes, the answer is correct", "yes").value == 0.0


def test_accuracy_valid_labels():
    r1 = ReferenceBasedMetrics.accuracy("yes", "yes", valid_labels=["yes", "no"])
    assert r1.value == 1.0
    assert r1.details["is_valid_label"] is True

    r2 = ReferenceBasedMetrics.accuracy("maybe", "maybe", valid_labels=["yes", "no"])
    assert r2.value == 1.0
    assert r2.details["is_valid_label"] is False
