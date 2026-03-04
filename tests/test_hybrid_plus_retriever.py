"""
Tests unitarios para HybridPlusRetriever.

Cubre:
  - index_documents con cross-refs (mock spaCy + inner retriever)
  - index_documents sin spaCy (graceful degradation, BM25+Vector+RRF puro)
  - _swap_to_original_contents (metadata, strategy_used)
  - clear_index (limpieza completa)
  - factory get_retriever() devuelve instancia correcta (sin llm_service)

spaCy/NIM/ChromaDB NO requeridos: todo mockeado.
"""

from unittest.mock import MagicMock, patch

import pytest

from shared.retrieval.core import RetrievalConfig, RetrievalResult, RetrievalStrategy
from shared.retrieval.hybrid_plus_retriever import HybridPlusRetriever


# =========================================================================
# Helpers: mocks reutilizables
# =========================================================================

def _make_mock_inner_retriever():
    """Mock del inner retriever (HybridRetriever o SimpleVector)."""
    inner = MagicMock()
    inner.index_documents.return_value = True
    inner.retrieve.return_value = RetrievalResult(
        doc_ids=["d1", "d2"],
        contents=["indexed content 1", "indexed content 2"],
        scores=[0.9, 0.8],
        strategy_used=RetrievalStrategy.HYBRID_PLUS,
        metadata={},
    )
    inner.retrieve_by_vector.return_value = RetrievalResult(
        doc_ids=["d1"],
        contents=["indexed content 1"],
        scores=[0.95],
        strategy_used=RetrievalStrategy.HYBRID_PLUS,
        metadata={},
    )
    inner.clear_index = MagicMock()
    return inner


def _make_retriever(inner=None):
    """Crea HybridPlusRetriever con mock inner inyectado."""
    config = RetrievalConfig(strategy=RetrievalStrategy.HYBRID_PLUS)
    embedding_model = MagicMock()
    mock_inner = inner or _make_mock_inner_retriever()

    with patch(
        "shared.retrieval.hybrid_retriever.HybridRetriever",
        return_value=mock_inner,
    ), patch(
        "shared.retrieval.hybrid_retriever.HAS_BM25", True,
    ), patch(
        "shared.retrieval.hybrid_retriever.HAS_TANTIVY", True,
    ):
        retriever = HybridPlusRetriever(
            config=config,
            embedding_model=embedding_model,
        )

    retriever._inner_retriever = mock_inner
    return retriever


SAMPLE_DOCS = [
    {"doc_id": "d1", "content": "Scott Derrickson directed Sinister.", "title": "Sinister (film)"},
    {"doc_id": "d2", "content": "Scott Derrickson was born in Sacramento.", "title": "Scott Derrickson"},
    {"doc_id": "d3", "content": "Python is a programming language.", "title": "Python"},
]


# =========================================================================
# Test: index_documents con cross-refs
# =========================================================================

class TestIndexDocumentsWithCrossRefs:

    def test_inner_receives_cross_refs_in_content(self):
        """Cuando HAS_SPACY=True, inner retriever recibe docs con cross-refs."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)

        mock_linker = MagicMock()
        mock_linker.compute_cross_refs.return_value = {
            "d1": "Related: Scott Derrickson (shared: scott derrickson)",
            "d2": "Related: Sinister (film) (shared: scott derrickson)",
        }
        mock_linker.get_stats.return_value = {"total_docs": 3, "total_entities": 4}

        with patch("shared.retrieval.entity_linker.HAS_SPACY", True), \
             patch("shared.retrieval.entity_linker.EntityLinker", return_value=mock_linker):
            result = retriever.index_documents(SAMPLE_DOCS)

        assert result is True

        enriched_docs = inner.index_documents.call_args[0][0]

        d1_doc = next(d for d in enriched_docs if d["doc_id"] == "d1")
        assert "Related: Scott Derrickson" in d1_doc["content"]
        assert "Scott Derrickson directed Sinister." in d1_doc["content"]

        d2_doc = next(d for d in enriched_docs if d["doc_id"] == "d2")
        assert "Related: Sinister (film)" in d2_doc["content"]

        d3_doc = next(d for d in enriched_docs if d["doc_id"] == "d3")
        assert "Related:" not in d3_doc["content"]

    def test_no_llm_enrichment_in_content(self):
        """Contenido indexado NO contiene contexto LLM — solo original + cross-refs."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)

        mock_linker = MagicMock()
        mock_linker.compute_cross_refs.return_value = {
            "d1": "Related: Scott Derrickson (shared: scott derrickson)",
        }
        mock_linker.get_stats.return_value = {"total_docs": 1}

        with patch("shared.retrieval.entity_linker.HAS_SPACY", True), \
             patch("shared.retrieval.entity_linker.EntityLinker", return_value=mock_linker):
            retriever.index_documents([SAMPLE_DOCS[0]])

        enriched_docs = inner.index_documents.call_args[0][0]
        d1_content = enriched_docs[0]["content"]

        # Original text present
        assert "Scott Derrickson directed Sinister." in d1_content
        # Cross-ref present
        assert "Related:" in d1_content
        # No "Context" or LLM-generated preamble
        assert "Context" not in d1_content

    def test_linker_stats_stored(self):
        """Despues de indexar con spaCy, _linker_stats se almacena."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)

        expected_stats = {"total_docs": 1, "total_entities": 2}
        mock_linker = MagicMock()
        mock_linker.compute_cross_refs.return_value = {}
        mock_linker.get_stats.return_value = expected_stats

        with patch("shared.retrieval.entity_linker.HAS_SPACY", True), \
             patch("shared.retrieval.entity_linker.EntityLinker", return_value=mock_linker):
            retriever.index_documents([SAMPLE_DOCS[0]])

        assert retriever._linker_stats == expected_stats

    def test_empty_documents_returns_false(self):
        """Lista de documentos vacia retorna False."""
        retriever = _make_retriever()
        assert retriever.index_documents([]) is False

    def test_original_contents_stored(self):
        """Contenido original almacenado para swap posterior."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)

        with patch("shared.retrieval.entity_linker.HAS_SPACY", False):
            retriever.index_documents(SAMPLE_DOCS[:2])

        assert retriever._original_contents["d1"] == "Scott Derrickson directed Sinister."
        assert retriever._original_contents["d2"] == "Scott Derrickson was born in Sacramento."


# =========================================================================
# Test: index_documents sin spaCy (graceful degradation)
# =========================================================================

class TestIndexDocumentsWithoutSpacy:

    def test_no_spacy_still_indexes(self):
        """Con HAS_SPACY=False, indexa sin cross-refs (BM25+Vector+RRF puro)."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)

        with patch("shared.retrieval.entity_linker.HAS_SPACY", False):
            result = retriever.index_documents(SAMPLE_DOCS[:2])

        assert result is True
        inner.index_documents.assert_called_once()

        enriched_docs = inner.index_documents.call_args[0][0]
        for doc in enriched_docs:
            assert "Related:" not in doc["content"]

    def test_no_spacy_logs_warning(self, caplog):
        """Con HAS_SPACY=False, se emite warning."""
        import logging
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)

        with patch("shared.retrieval.entity_linker.HAS_SPACY", False), \
             caplog.at_level(logging.WARNING):
            retriever.index_documents([SAMPLE_DOCS[0]])

        assert any("spaCy no disponible" in msg for msg in caplog.messages)

    def test_no_spacy_content_is_original(self):
        """Sin spaCy, contenido indexado es identico al original."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)

        with patch("shared.retrieval.entity_linker.HAS_SPACY", False):
            retriever.index_documents([SAMPLE_DOCS[0]])

        enriched_docs = inner.index_documents.call_args[0][0]
        assert enriched_docs[0]["content"] == SAMPLE_DOCS[0]["content"]


# =========================================================================
# Test: _swap_to_original_contents
# =========================================================================

class TestSwapToOriginalContents:

    def test_retrieve_returns_original_content(self):
        """retrieve() devuelve contenido original, no indexado con cross-refs."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)
        retriever._original_contents = {
            "d1": "original content 1",
            "d2": "original content 2",
        }

        result = retriever.retrieve("some query")
        assert result.contents == ["original content 1", "original content 2"]

    def test_retrieve_sets_strategy(self):
        """retrieve() marca strategy_used como HYBRID_PLUS."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)
        retriever._original_contents = {"d1": "orig1", "d2": "orig2"}

        result = retriever.retrieve("query")
        assert result.strategy_used == RetrievalStrategy.HYBRID_PLUS

    def test_retrieve_sets_metadata(self):
        """retrieve() establece metadata de cross-linking."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)
        retriever._original_contents = {"d1": "orig1", "d2": "orig2"}
        retriever._linker_stats = {"total_docs": 2}

        result = retriever.retrieve("query")

        assert result.metadata["entity_cross_linking"] is True
        assert result.metadata["linker_stats"] == {"total_docs": 2}

    def test_retrieve_by_vector_returns_original(self):
        """retrieve_by_vector() tambien devuelve contenido original."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)
        retriever._original_contents = {"d1": "orig1"}

        result = retriever.retrieve_by_vector("query", [0.1, 0.2])

        assert result.contents == ["orig1"]
        assert result.strategy_used == RetrievalStrategy.HYBRID_PLUS

    def test_unknown_doc_id_preserves_indexed(self):
        """Si doc_id no esta en _original_contents, preserva content del inner."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)
        retriever._original_contents = {}

        result = retriever.retrieve("query")
        assert result.contents == ["indexed content 1", "indexed content 2"]


# =========================================================================
# Test: clear_index
# =========================================================================

class TestClearIndex:

    def test_clear_resets_all_state(self):
        """clear_index() limpia inner, originals, stats y flag."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)

        retriever._original_contents = {"d1": "x", "d2": "y"}
        retriever._linker_stats = {"total_docs": 2}
        retriever._is_indexed = True

        retriever.clear_index()

        inner.clear_index.assert_called_once()
        assert retriever._original_contents == {}
        assert retriever._linker_stats == {}
        assert retriever._is_indexed is False


# =========================================================================
# Test: Factory get_retriever
# =========================================================================

class TestFactoryHybridPlus:

    def test_factory_returns_hybrid_plus(self):
        """get_retriever con HYBRID_PLUS devuelve HybridPlusRetriever."""
        config = RetrievalConfig(strategy=RetrievalStrategy.HYBRID_PLUS)
        mock_embedding = MagicMock()

        with patch(
            "shared.retrieval.hybrid_retriever.HybridRetriever",
        ), patch(
            "shared.retrieval.hybrid_retriever.HAS_BM25", True,
        ), patch(
            "shared.retrieval.hybrid_retriever.HAS_TANTIVY", True,
        ):
            from shared.retrieval import get_retriever
            retriever = get_retriever(config, mock_embedding)

        assert isinstance(retriever, HybridPlusRetriever)

    def test_factory_does_not_require_llm(self):
        """HYBRID_PLUS no requiere llm_service (a diferencia del antiguo CONTEXTUAL_HYBRID)."""
        config = RetrievalConfig(strategy=RetrievalStrategy.HYBRID_PLUS)
        mock_embedding = MagicMock()

        with patch(
            "shared.retrieval.hybrid_retriever.HybridRetriever",
        ), patch(
            "shared.retrieval.hybrid_retriever.HAS_BM25", True,
        ), patch(
            "shared.retrieval.hybrid_retriever.HAS_TANTIVY", True,
        ):
            from shared.retrieval import get_retriever
            # No llm_service -> no error
            retriever = get_retriever(config, mock_embedding)

        assert isinstance(retriever, HybridPlusRetriever)

    def test_factory_passes_entity_config(self):
        """Factory pasa entity_max_cross_refs, entity_min_shared, entity_max_doc_fraction."""
        config = RetrievalConfig(
            strategy=RetrievalStrategy.HYBRID_PLUS,
            entity_max_cross_refs=5,
            entity_min_shared=2,
            entity_max_doc_fraction=0.10,
        )
        mock_embedding = MagicMock()

        with patch(
            "shared.retrieval.hybrid_retriever.HybridRetriever",
        ), patch(
            "shared.retrieval.hybrid_retriever.HAS_BM25", True,
        ), patch(
            "shared.retrieval.hybrid_retriever.HAS_TANTIVY", True,
        ):
            from shared.retrieval import get_retriever
            retriever = get_retriever(config, mock_embedding)

        assert retriever._max_cross_refs == 5
        assert retriever._min_shared_entities == 2
        assert retriever._max_entity_doc_fraction == 0.10
