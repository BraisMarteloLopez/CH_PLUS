"""
Tests para _format_context().

Cobertura:
  - Lista vacia retorna placeholder
  - Headers [Doc N] presentes y numerados
  - Truncacion por max_chars (docs que no caben se omiten)
  - Un solo doc largo: cabe entero o no cabe (se omite, resultado vacio con placeholder? No: el break ocurre antes de append, asi que si el primer doc no cabe, parts queda vacio)
  - Multiples docs, todos caben
  - Separador \n\n entre docs

Sin dependencias externas.
"""
from unittest.mock import MagicMock

from sandbox_mteb.evaluator import MTEBEvaluator
from sandbox_mteb.config import MTEBConfig, MinIOStorageConfig
from shared.config_base import InfraConfig, RerankerConfig
from shared.retrieval.core import RetrievalConfig


def _make_evaluator(max_context_chars: int = 4000) -> MTEBEvaluator:
    config = MTEBConfig(
        infra=InfraConfig(),
        storage=MinIOStorageConfig(),
        retrieval=RetrievalConfig(),
        reranker=RerankerConfig(enabled=False),
        generation_enabled=True,
    )
    evaluator = MTEBEvaluator(config)
    evaluator._max_context_chars = max_context_chars
    return evaluator


# =================================================================
# Tests
# =================================================================

def test_empty_list_returns_placeholder():
    """Lista vacia retorna placeholder fijo."""
    ev = _make_evaluator()
    result = ev._format_context([])
    assert result == "[No se encontraron documentos]", f"Obtenido: '{result}'"
    print("PASS: lista vacia retorna placeholder")


def test_single_doc_has_header():
    """Un solo documento incluye header [Doc 1]."""
    ev = _make_evaluator()
    result = ev._format_context(["contenido del doc"])
    assert result.startswith("[Doc 1]\n"), f"No empieza con header: '{result[:30]}'"
    assert "contenido del doc" in result
    print("PASS: single doc tiene header [Doc 1]")


def test_multiple_docs_numbered():
    """Multiples docs tienen headers numerados secuencialmente."""
    ev = _make_evaluator()
    result = ev._format_context(["doc uno", "doc dos", "doc tres"])

    assert "[Doc 1]" in result
    assert "[Doc 2]" in result
    assert "[Doc 3]" in result
    assert "doc uno" in result
    assert "doc dos" in result
    assert "doc tres" in result
    print("PASS: multiples docs numerados [Doc 1], [Doc 2], [Doc 3]")


def test_docs_separated_by_double_newline():
    """Documentos separados por \\n\\n."""
    ev = _make_evaluator()
    result = ev._format_context(["aaa", "bbb"])

    # Estructura esperada: "[Doc 1]\naaa\n\n[Doc 2]\nbbb"
    parts = result.split("\n\n")
    assert len(parts) == 2, f"Esperado 2 partes separadas por \\n\\n, obtenido {len(parts)}"
    assert parts[0].endswith("aaa")
    assert parts[1].endswith("bbb")
    print("PASS: docs separados por \\n\\n")


def test_truncation_omits_docs_that_exceed_limit():
    """Docs que no caben en max_chars se omiten (break antes de append)."""
    # header "[Doc N]\n" = 8 chars
    # Primer doc: sep_len=0 + part_len(8+4)=12. length=12.
    # Segundo doc: sep_len=2 + part_len(8+4)=12. Guard: 12+2+12=26 > 20. Break.
    # Con max_context_chars=20, segundo doc no cabe
    ev = _make_evaluator(max_context_chars=20)
    result = ev._format_context(["aaaa", "bbbb"])

    assert "aaaa" in result
    assert "bbbb" not in result, f"Segundo doc no deberia caber: '{result}'"
    assert "[Doc 2]" not in result
    print("PASS: truncacion omite docs que exceden limite")


def test_truncation_first_doc_too_large():
    """Si el primer doc ya excede max_chars, result es string vacio (join de lista vacia)."""
    ev = _make_evaluator(max_context_chars=5)
    # "[Doc 1]\n" = 8 chars > 5, asi que el primer doc no cabe
    result = ev._format_context(["este doc es largo"])
    assert result == "", f"Esperado string vacio, obtenido: '{result}'"
    print("PASS: primer doc excede limite -> string vacio")


def test_all_docs_fit_within_limit():
    """Todos los docs caben si max_chars es suficiente."""
    ev = _make_evaluator(max_context_chars=10000)
    docs = [f"documento numero {i}" for i in range(5)]
    result = ev._format_context(docs)

    for i in range(1, 6):
        assert f"[Doc {i}]" in result, f"Falta [Doc {i}]"
    print("PASS: todos los docs caben con limite amplio")


def test_exact_boundary():
    """Doc que cabe exactamente en el limite se incluye."""
    # Calcular tamano exacto para un doc
    content = "x"  # 1 char
    header = "[Doc 1]\n"  # 8 chars
    # length check: 0 + 8 + 1 = 9 > max_length?
    # Si max_length=9, 9 > 9 es False, asi que cabe
    ev = _make_evaluator(max_context_chars=9)
    result = ev._format_context(["x"])
    assert "x" in result, f"Doc deberia caber exactamente: '{result}'"
    print("PASS: doc que cabe exactamente en el limite se incluye")


def test_empty_string_docs():
    """Documentos con string vacio: headers se generan pero sin contenido."""
    ev = _make_evaluator()
    result = ev._format_context(["", "contenido"])
    assert "[Doc 1]" in result
    assert "[Doc 2]" in result
    assert "contenido" in result
    print("PASS: docs con string vacio generan headers")


if __name__ == "__main__":
    test_empty_list_returns_placeholder()
    test_single_doc_has_header()
    test_multiple_docs_numbered()
    test_docs_separated_by_double_newline()
    test_truncation_omits_docs_that_exceed_limit()
    test_truncation_first_doc_too_large()
    test_all_docs_fit_within_limit()
    test_exact_boundary()
    test_empty_string_docs()
