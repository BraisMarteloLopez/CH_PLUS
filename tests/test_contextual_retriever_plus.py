"""
Tests unitarios para ContextualRetrieverPlus.

Cubre:
  - index_documents con cross-refs (mock spaCy + inner retriever)
  - index_documents sin spaCy (HAS_SPACY=False, graceful degradation)
  - _swap_to_original_contents (metadata, strategy_used)
  - clear_index (limpieza completa)
  - factory get_retriever() devuelve instancia correcta

spaCy/NIM/ChromaDB NO requeridos: todo mockeado.
"""

from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from shared.retrieval.core import RetrievalConfig, RetrievalResult, RetrievalStrategy
from shared.retrieval.contextual_retriever import (
    ContextualRetrieverPlus,
    EnrichedChunk,
    LLMContextGenerator,
)


# =========================================================================
# Helpers: mocks reutilizables
# =========================================================================

def _make_mock_inner_retriever():
    """Mock del inner retriever (HybridRetriever o SimpleVector)."""
    inner = MagicMock()
    inner.index_documents.return_value = True
    inner.retrieve.return_value = RetrievalResult(
        doc_ids=["d1", "d2"],
        contents=["enriched content 1", "enriched content 2"],
        scores=[0.9, 0.8],
        strategy_used=RetrievalStrategy.CONTEXTUAL_HYBRID,
        metadata={},
    )
    inner.retrieve_by_vector.return_value = RetrievalResult(
        doc_ids=["d1"],
        contents=["enriched content 1"],
        scores=[0.95],
        strategy_used=RetrievalStrategy.CONTEXTUAL_HYBRID,
        metadata={},
    )
    inner.clear_index = MagicMock()
    return inner


def _make_mock_context_generator():
    """Mock de LLMContextGenerator."""
    gen = MagicMock(spec=LLMContextGenerator)
    gen.get_stats.return_value = {
        "total_generated": 3, "cache_hits": 0, "errors": 0,
        "cache_size": 3, "with_parent": 0, "fallback": 3,
    }
    gen.clear_cache = MagicMock()
    return gen


def _make_retriever_plus(inner=None, context_gen=None):
    """Crea ContextualRetrieverPlus con mocks inyectados.

    HybridRetriever se importa localmente en __init__, asi que parcheamos
    el modulo fuente (shared.retrieval.hybrid_retriever).
    """
    config = RetrievalConfig(strategy=RetrievalStrategy.CONTEXTUAL_HYBRID_PLUS)
    embedding_model = MagicMock()
    context_gen = context_gen or _make_mock_context_generator()
    mock_inner = inner or _make_mock_inner_retriever()

    # Patch at source module (local import in __init__)
    with patch(
        "shared.retrieval.hybrid_retriever.HybridRetriever",
        return_value=mock_inner,
    ), patch(
        "shared.retrieval.hybrid_retriever.HAS_BM25", True,
    ), patch(
        "shared.retrieval.hybrid_retriever.HAS_TANTIVY", True,
    ):
        retriever = ContextualRetrieverPlus(
            config=config,
            embedding_model=embedding_model,
            context_generator=context_gen,
        )

    # Ensure the mock inner is set (in case patch order differs)
    retriever._inner_retriever = mock_inner
    return retriever


SAMPLE_DOCS = [
    {"doc_id": "d1", "content": "Scott Derrickson directed Sinister.", "title": "Sinister (film)"},
    {"doc_id": "d2", "content": "Scott Derrickson was born in Sacramento.", "title": "Scott Derrickson"},
    {"doc_id": "d3", "content": "Python is a programming language.", "title": "Python"},
]


# =========================================================================
# Test 4.1: index_documents con cross-refs
# =========================================================================

class TestIndexDocumentsWithCrossRefs:

    def test_inner_receives_cross_refs_in_content(self):
        """Cuando HAS_SPACY=True, inner retriever recibe docs con cross-refs."""
        inner = _make_mock_inner_retriever()
        context_gen = _make_mock_context_generator()
        retriever = _make_retriever_plus(inner=inner, context_gen=context_gen)

        # Mock _run_batch_generation to return enriched chunks
        enriched = [
            EnrichedChunk("d1", "Scott Derrickson directed Sinister.", "Context for d1", "Sinister (film)"),
            EnrichedChunk("d2", "Scott Derrickson was born in Sacramento.", "Context for d2", "Scott Derrickson"),
            EnrichedChunk("d3", "Python is a programming language.", "Context for d3", "Python"),
        ]
        retriever._run_batch_generation = MagicMock(return_value=enriched)

        # Mock EntityLinker to produce cross-refs
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

        # Verificar que inner retriever recibio docs enriquecidos
        call_args = inner.index_documents.call_args
        enriched_docs = call_args[0][0]

        # d1 y d2 deben tener cross-refs en el content
        d1_doc = next(d for d in enriched_docs if d["doc_id"] == "d1")
        assert "Related: Scott Derrickson" in d1_doc["content"]
        assert "Context for d1" in d1_doc["content"]  # contexto LLM preservado

        d2_doc = next(d for d in enriched_docs if d["doc_id"] == "d2")
        assert "Related: Sinister (film)" in d2_doc["content"]

        # d3 no tiene cross-refs -> content solo tiene contexto + original
        d3_doc = next(d for d in enriched_docs if d["doc_id"] == "d3")
        assert "Related:" not in d3_doc["content"]
        assert "Context for d3" in d3_doc["content"]

    def test_linker_stats_stored(self):
        """Despues de indexar con spaCy, _linker_stats se almacena."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever_plus(inner=inner)

        enriched = [
            EnrichedChunk("d1", "content1", "ctx1", "T1"),
        ]
        retriever._run_batch_generation = MagicMock(return_value=enriched)

        expected_stats = {"total_docs": 1, "total_entities": 2}
        mock_linker = MagicMock()
        mock_linker.compute_cross_refs.return_value = {}
        mock_linker.get_stats.return_value = expected_stats

        with patch("shared.retrieval.entity_linker.HAS_SPACY", True), \
             patch("shared.retrieval.entity_linker.EntityLinker", return_value=mock_linker):
            retriever.index_documents([{"doc_id": "d1", "content": "x", "title": "T1"}])

        assert retriever._linker_stats == expected_stats

    def test_empty_documents_returns_false(self):
        """Lista de documentos vacia retorna False."""
        retriever = _make_retriever_plus()
        assert retriever.index_documents([]) is False

    def test_original_contents_stored(self):
        """Contenido original almacenado para swap posterior."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever_plus(inner=inner)

        enriched = [
            EnrichedChunk("d1", "original text 1", "ctx1", "T1"),
            EnrichedChunk("d2", "original text 2", "ctx2", "T2"),
        ]
        retriever._run_batch_generation = MagicMock(return_value=enriched)

        with patch("shared.retrieval.entity_linker.HAS_SPACY", False):
            retriever.index_documents([
                {"doc_id": "d1", "content": "original text 1", "title": "T1"},
                {"doc_id": "d2", "content": "original text 2", "title": "T2"},
            ])

        assert retriever._original_contents["d1"] == "original text 1"
        assert retriever._original_contents["d2"] == "original text 2"


# =========================================================================
# Test 4.2: index_documents sin spaCy (graceful degradation)
# =========================================================================

class TestIndexDocumentsWithoutSpacy:

    def test_no_spacy_still_indexes(self):
        """Con HAS_SPACY=False, indexa sin cross-refs (como CONTEXTUAL_HYBRID)."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever_plus(inner=inner)

        enriched = [
            EnrichedChunk("d1", "original 1", "ctx1", "T1"),
            EnrichedChunk("d2", "original 2", "ctx2", "T2"),
        ]
        retriever._run_batch_generation = MagicMock(return_value=enriched)

        with patch("shared.retrieval.entity_linker.HAS_SPACY", False):
            result = retriever.index_documents([
                {"doc_id": "d1", "content": "original 1", "title": "T1"},
                {"doc_id": "d2", "content": "original 2", "title": "T2"},
            ])

        assert result is True
        inner.index_documents.assert_called_once()

        # Sin cross-refs: content = contexto + original, sin "Related:"
        enriched_docs = inner.index_documents.call_args[0][0]
        for doc in enriched_docs:
            assert "Related:" not in doc["content"]

    def test_no_spacy_logs_warning(self, caplog):
        """Con HAS_SPACY=False, se emite warning de degradacion."""
        import logging
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever_plus(inner=inner)

        enriched = [EnrichedChunk("d1", "x", "ctx", "T")]
        retriever._run_batch_generation = MagicMock(return_value=enriched)

        with patch("shared.retrieval.entity_linker.HAS_SPACY", False), \
             caplog.at_level(logging.WARNING):
            retriever.index_documents([{"doc_id": "d1", "content": "x", "title": "T"}])

        assert any("spaCy no disponible" in msg for msg in caplog.messages)


# =========================================================================
# Test 4.3: _swap_to_original_contents
# =========================================================================

class TestSwapToOriginalContents:

    def test_retrieve_returns_original_content(self):
        """retrieve() devuelve contenido original, no enriquecido."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever_plus(inner=inner)

        # Simular que ya se indexo
        retriever._original_contents = {
            "d1": "original content 1",
            "d2": "original content 2",
        }

        result = retriever.retrieve("some query")

        assert result.contents == ["original content 1", "original content 2"]

    def test_retrieve_sets_strategy(self):
        """retrieve() marca strategy_used como CONTEXTUAL_HYBRID_PLUS."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever_plus(inner=inner)
        retriever._original_contents = {"d1": "orig1", "d2": "orig2"}

        result = retriever.retrieve("query")
        assert result.strategy_used == RetrievalStrategy.CONTEXTUAL_HYBRID_PLUS

    def test_retrieve_sets_metadata(self):
        """retrieve() establece metadata de enrichment y cross-linking."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever_plus(inner=inner)
        retriever._original_contents = {"d1": "orig1", "d2": "orig2"}
        retriever._linker_stats = {"total_docs": 2}

        result = retriever.retrieve("query")

        assert result.metadata["contextual_enrichment"] is True
        assert result.metadata["entity_cross_linking"] is True
        assert "context_generator_stats" in result.metadata
        assert result.metadata["linker_stats"] == {"total_docs": 2}

    def test_retrieve_by_vector_returns_original(self):
        """retrieve_by_vector() tambien devuelve contenido original."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever_plus(inner=inner)
        retriever._original_contents = {"d1": "orig1"}

        result = retriever.retrieve_by_vector("query", [0.1, 0.2])

        assert result.contents == ["orig1"]
        assert result.strategy_used == RetrievalStrategy.CONTEXTUAL_HYBRID_PLUS

    def test_unknown_doc_id_preserves_enriched(self):
        """Si doc_id no esta en _original_contents, preserva content del inner."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever_plus(inner=inner)
        # No hay mapping para d1/d2 -> se mantiene lo del inner
        retriever._original_contents = {}

        result = retriever.retrieve("query")
        assert result.contents == ["enriched content 1", "enriched content 2"]


# =========================================================================
# Test 4.4: clear_index
# =========================================================================

class TestClearIndex:

    def test_clear_resets_all_state(self):
        """clear_index() limpia inner, cache, originals, stats y flag."""
        inner = _make_mock_inner_retriever()
        context_gen = _make_mock_context_generator()
        retriever = _make_retriever_plus(inner=inner, context_gen=context_gen)

        # Simular estado post-indexacion
        retriever._original_contents = {"d1": "x", "d2": "y"}
        retriever._linker_stats = {"total_docs": 2}
        retriever._is_indexed = True

        retriever.clear_index()

        inner.clear_index.assert_called_once()
        context_gen.clear_cache.assert_called_once()
        assert retriever._original_contents == {}
        assert retriever._linker_stats == {}
        assert retriever._is_indexed is False


# =========================================================================
# Test 4.5: Factory get_retriever
# =========================================================================

class TestFactoryContextualHybridPlus:

    def test_factory_returns_plus(self):
        """get_retriever con CONTEXTUAL_HYBRID_PLUS devuelve ContextualRetrieverPlus."""
        config = RetrievalConfig(strategy=RetrievalStrategy.CONTEXTUAL_HYBRID_PLUS)
        mock_embedding = MagicMock()
        mock_llm = MagicMock()

        with patch(
            "shared.retrieval.hybrid_retriever.HybridRetriever",
        ), patch(
            "shared.retrieval.hybrid_retriever.HAS_BM25", True,
        ), patch(
            "shared.retrieval.hybrid_retriever.HAS_TANTIVY", True,
        ):
            from shared.retrieval import get_retriever
            retriever = get_retriever(
                config, mock_embedding, llm_service=mock_llm,
            )

        assert isinstance(retriever, ContextualRetrieverPlus)

    def test_factory_plus_requires_llm(self):
        """get_retriever sin llm_service para PLUS lanza ValueError."""
        config = RetrievalConfig(strategy=RetrievalStrategy.CONTEXTUAL_HYBRID_PLUS)
        mock_embedding = MagicMock()

        from shared.retrieval import get_retriever
        with pytest.raises(ValueError, match="CONTEXTUAL_HYBRID_PLUS requiere llm_service"):
            get_retriever(config, mock_embedding, llm_service=None)

    def test_factory_passes_entity_config(self):
        """Factory pasa entity_max_cross_refs, entity_min_shared, entity_max_doc_fraction."""
        config = RetrievalConfig(
            strategy=RetrievalStrategy.CONTEXTUAL_HYBRID_PLUS,
            entity_max_cross_refs=5,
            entity_min_shared=2,
            entity_max_doc_fraction=0.10,
        )
        mock_embedding = MagicMock()
        mock_llm = MagicMock()

        with patch(
            "shared.retrieval.hybrid_retriever.HybridRetriever",
        ), patch(
            "shared.retrieval.hybrid_retriever.HAS_BM25", True,
        ), patch(
            "shared.retrieval.hybrid_retriever.HAS_TANTIVY", True,
        ):
            from shared.retrieval import get_retriever
            retriever = get_retriever(
                config, mock_embedding, llm_service=mock_llm,
            )

        assert retriever._max_cross_refs == 5
        assert retriever._min_shared_entities == 2
        assert retriever._max_entity_doc_fraction == 0.10
