"""
Tests DT-9: _extract_score_fallback() regex robusta.

Cubre: decimales 0-1, falsos positivos rechazados, escala 1-10 con
prefijo "score:", fracciones N/M, defaults, respuestas LLM reales.
"""
import pytest

from shared.metrics import LLMJudgeMetrics

_extract = LLMJudgeMetrics._extract_score_fallback


@pytest.mark.parametrize("text,expected", [
    ("score: 0", 0.0),
    ("score: 1", 1.0),
    ("the score is 0.85", 0.85),
    ("score: 1.0", 1.0),
    ("0.0", 0.0),
    ("I would rate this 0.7 overall", 0.7),
    ("Based on my analysis, the faithfulness score is 0.75", 0.75),
    ("The response is well-grounded. Score: 0.9", 0.9),
])
def test_decimal_extraction(text, expected):
    """Decimales 0-1 se extraen correctamente, incluyendo respuestas LLM reales."""
    assert abs(_extract(text) - expected) < 0.001


@pytest.mark.parametrize("text", [
    "10.5 points out of 10",
    "there are 100 reasons",
    "about 20 tokens were used",
])
def test_false_positives_rejected(text):
    """Numeros >1 no producen score 1.0 espurio."""
    assert _extract(text) != 1.0


@pytest.mark.parametrize("text,expected", [
    ("score: 8", 0.8),
    ("score: 3", 0.3),
    ("score: 10", 1.0),
])
def test_score_prefix_normalizes_1_to_10(text, expected):
    """Enteros 1-10 con prefijo 'score:' se normalizan a 0-1."""
    assert abs(_extract(text) - expected) < 0.001


@pytest.mark.parametrize("text,expected", [
    ("I would rate this 8/10", 0.8),
    ("1/2", 0.5),
    ("10/10", 1.0),
    ("0/0", 0.0),
])
def test_fractions(text, expected):
    """Fracciones N/M se normalizan. 0/0 retorna 0.0 (fraccion descartada)."""
    assert abs(_extract(text) - expected) < 0.001


def test_no_score_prefix_integer_ignored():
    """Entero sin prefijo 'score:' no se normaliza — evita falsos positivos."""
    assert _extract("I give it 8") == 0.5


@pytest.mark.parametrize("text", [
    "I cannot provide a score",
    "",
])
def test_default_when_no_extractable_score(text):
    """Sin numero extraible, retorna default 0.5."""
    assert _extract(text) == 0.5
