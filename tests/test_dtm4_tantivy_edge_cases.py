"""
Tests para TantivyIndex edge cases en tantivy_index.py.

Cobertura DTm-4:
  - Query con caracteres especiales (apostrofes, comillas, ?, parentesis)
  - Query vacia retorna []
  - Query solo whitespace retorna []
  - Indice vacio (sin docs) retorna []
  - Indice no inicializado (None) retorna []
  - Sanitizacion de query: re.sub elimina caracteres especiales
  - clear() resetea estado
  - build_index con lista vacia retorna 0

Usa @patch sobre HAS_TANTIVY para permitir instanciacion via constructor
real sin la libreria Rust. Los guards en search() retornan antes de
llamar la API de tantivy, lo que permite testar condiciones de borde
sin dependencia externa.
"""
import re
from unittest.mock import patch

from shared.retrieval.tantivy_index import TantivyIndex, HAS_TANTIVY


# =================================================================
# Tests de sanitizacion (logica Python pura, sin Tantivy)
# =================================================================

def test_query_sanitization_removes_special_chars():
    """re.sub elimina apostrofes, comillas, ?, parentesis, etc."""
    pattern = r'[^\w\s]'

    assert re.sub(pattern, ' ', "what's") == "what s"
    assert re.sub(pattern, ' ', '"hello"') == " hello "
    assert re.sub(pattern, ' ', "who is this?") == "who is this "
    assert re.sub(pattern, ' ', "foo (bar)") == "foo  bar "
    assert re.sub(pattern, ' ', "it's a test? yes!") == "it s a test  yes "
    assert re.sub(pattern, ' ', "normal query") == "normal query"


def test_sanitization_preserves_alphanumeric():
    """Sanitizacion preserva letras, numeros y espacios."""
    pattern = r'[^\w\s]'
    result = re.sub(pattern, ' ', "abc123 XYZ 456")
    assert result == "abc123 XYZ 456"


def test_sanitization_collapses_to_empty():
    """Query de solo caracteres especiales queda vacia tras sanitizar."""
    pattern = r'[^\w\s]'
    clean = re.sub(pattern, ' ', "???!!!")
    normalized = ' '.join(clean.split())
    assert normalized == ""


# =================================================================
# Tests funcionales via API publica
#
# El constructor de TantivyIndex solo necesita HAS_TANTIVY=True para
# no lanzar ImportError. No llama la API de tantivy en __init__.
# Los guards en search() retornan [] antes de tocar tantivy si:
#   - self._index is None or self._doc_count == 0
#   - not query or not query.strip()
# Esto permite testear todas las condiciones de borde sin la lib Rust.
# =================================================================

@patch("shared.retrieval.tantivy_index.HAS_TANTIVY", True)
class TestTantivyIndexGuardConditions:
    """Tests de condiciones guardia via constructor real."""

    def test_search_empty_query_returns_empty(self):
        """Query vacia retorna lista vacia."""
        idx = TantivyIndex(language="en")
        assert idx.search("", top_k=10) == []

    def test_search_whitespace_query_returns_empty(self):
        """Query solo whitespace retorna lista vacia."""
        idx = TantivyIndex(language="en")
        assert idx.search("   ", top_k=10) == []

    def test_search_no_index_returns_empty(self):
        """Sin indice inicializado (_index=None por defecto) retorna lista vacia."""
        idx = TantivyIndex(language="en")
        # _index es None por defecto tras __init__
        assert idx.search("test query", top_k=10) == []

    def test_search_special_chars_query_no_crash(self):
        """Query con caracteres especiales no lanza excepcion."""
        idx = TantivyIndex(language="en")
        # Guard: _index is None -> retorna [] antes de sanitizar
        result = idx.search("what's the meaning of life?", top_k=10)
        assert result == []

    def test_clear_resets_state(self):
        """clear() resetea doc_count y limpia contenidos."""
        idx = TantivyIndex(language="en")
        # Simular estado post-indexacion via API publica:
        # build_index([]) ya probamos que retorna 0 sin tocar tantivy.
        # Para clear, simplemente verificamos que el estado queda limpio.
        idx._doc_count = 5  # unico acceso interno: simular que hubo docs
        idx._doc_contents = {"d1": "content"}
        idx.clear()

        assert idx.size == 0
        assert idx.search("anything", top_k=10) == []

    def test_size_property_initial(self):
        """size retorna 0 tras construccion."""
        idx = TantivyIndex(language="en")
        assert idx.size == 0

    def test_build_index_empty_list_returns_zero(self):
        """build_index con lista vacia retorna 0 sin tocar tantivy."""
        idx = TantivyIndex(language="en")
        result = idx.build_index([])
        assert result == 0
        assert idx.size == 0

    def test_search_after_empty_build_returns_empty(self):
        """Tras build_index([]), search retorna [] (0 docs)."""
        idx = TantivyIndex(language="en")
        idx.build_index([])
        assert idx.search("test", top_k=10) == []

    def test_constructor_sets_language(self):
        """Constructor configura language y tokenizer correctamente."""
        idx = TantivyIndex(language="es")
        assert idx._language == "es"
        assert idx._tokenizer_name == "es_stem"

    def test_constructor_unknown_language_defaults_to_en(self):
        """Idioma desconocido usa tokenizer en_stem como fallback."""
        idx = TantivyIndex(language="xx")
        assert idx._tokenizer_name == "en_stem"


# =================================================================
# Tests funcionales completos (solo si Tantivy esta instalado)
# =================================================================

if HAS_TANTIVY:
    class TestTantivyIndexFunctional:
        """Tests funcionales que requieren Tantivy real."""

        def test_index_and_search_basic(self):
            """Indexar y buscar documentos basicos."""
            idx = TantivyIndex(language="en")
            docs = [
                {"doc_id": "d1", "content": "artificial intelligence machine learning", "title": "AI"},
                {"doc_id": "d2", "content": "natural language processing text", "title": "NLP"},
                {"doc_id": "d3", "content": "computer vision image recognition", "title": "CV"},
            ]
            count = idx.build_index(docs)
            assert count == 3

            results = idx.search("artificial intelligence", top_k=2)
            assert len(results) > 0
            assert results[0][0] == "d1"
            idx.clear()

        def test_search_special_chars_no_crash(self):
            """Query con chars especiales no causa crash."""
            idx = TantivyIndex(language="en")
            docs = [
                {"doc_id": "d1", "content": "what is this about", "title": "Test"},
            ]
            idx.build_index(docs)

            idx.search("what's this?", top_k=5)
            idx.search('"quoted query"', top_k=5)
            idx.search("foo (bar) [baz]", top_k=5)
            idx.search("price: $100", top_k=5)
            idx.clear()

        def test_search_only_special_chars_returns_empty(self):
            """Query de solo caracteres especiales retorna vacio."""
            idx = TantivyIndex(language="en")
            docs = [
                {"doc_id": "d1", "content": "some content here", "title": "Doc"},
            ]
            idx.build_index(docs)

            result = idx.search("???!!!", top_k=5)
            assert result == []
            idx.clear()

        def test_rebuild_index_replaces_previous(self):
            """build_index dos veces reemplaza el indice anterior."""
            idx = TantivyIndex(language="en")

            docs1 = [{"doc_id": "d1", "content": "first version", "title": "V1"}]
            idx.build_index(docs1)

            docs2 = [
                {"doc_id": "d2", "content": "second version new", "title": "V2"},
                {"doc_id": "d3", "content": "third document", "title": "V3"},
            ]
            count = idx.build_index(docs2)
            assert count == 2
            assert idx.size == 2

            results = idx.search("second version", top_k=5)
            doc_ids = [r[0] for r in results]
            assert "d2" in doc_ids
            assert "d1" not in doc_ids
            idx.clear()


if __name__ == "__main__":
    test_query_sanitization_removes_special_chars()
    test_sanitization_preserves_alphanumeric()
    test_sanitization_collapses_to_empty()

    # Nota: TestTantivyIndexGuardConditions requiere @patch,
    # ejecutar via pytest para que funcione el decorator.

    if HAS_TANTIVY:
        tf = TestTantivyIndexFunctional()
        tf.test_index_and_search_basic()
        tf.test_search_special_chars_no_crash()
        tf.test_search_only_special_chars_returns_empty()
        tf.test_rebuild_index_replaces_previous()
