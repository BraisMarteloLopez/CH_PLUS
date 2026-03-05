"""
Tests unitarios para HybridPlusRetriever.

Cubre:
  - index_documents: inline enrichment + graph construction
  - index_documents sin spaCy (graceful degradation)
  - _expand_with_graph: graph expansion during retrieval
  - _swap_to_original_contents: contenido original para generacion
  - clear_index (limpieza completa)
  - factory get_retriever() devuelve instancia correcta (sin llm_service)

Arquitectura hibrida: LiteGraph inline + graph expansion + swap.
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
        contents=["enriched content 1", "enriched content 2"],
        scores=[0.9, 0.8],
        strategy_used=RetrievalStrategy.HYBRID_PLUS,
        metadata={},
    )
    inner.retrieve_by_vector.return_value = RetrievalResult(
        doc_ids=["d1"],
        contents=["enriched content 1"],
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
# Test: index_documents with inline enrichment + graph
# =========================================================================

class TestIndexDocumentsWithCrossRefs:

    def test_dual_indexing_vector_gets_original_bm25_gets_enriched(self):
        """Inner retriever recibe docs originales para vector, enriquecidos para BM25."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)

        mock_linker = MagicMock()
        mock_linker.compute_cross_refs.return_value = {
            "d1": "See also Scott Derrickson regarding scott derrickson.",
            "d2": "See also Sinister (film) regarding scott derrickson.",
        }
        mock_linker.get_cross_ref_graph.side_effect = lambda doc_id: {
            "d1": ["d2"], "d2": ["d1"],
        }.get(doc_id, [])
        mock_linker.get_stats.return_value = {"total_docs": 3, "total_entities": 4}

        with patch("shared.retrieval.entity_linker.HAS_SPACY", True), \
             patch("shared.retrieval.entity_linker.EntityLinker", return_value=mock_linker):
            result = retriever.index_documents(SAMPLE_DOCS)

        assert result is True

        # Vector docs (first positional arg) should be ORIGINAL content
        vector_docs = inner.index_documents.call_args[0][0]

        d1_vec = next(d for d in vector_docs if d["doc_id"] == "d1")
        assert d1_vec["content"] == "Scott Derrickson directed Sinister."
        assert "See also" not in d1_vec["content"]

        d2_vec = next(d for d in vector_docs if d["doc_id"] == "d2")
        assert d2_vec["content"] == "Scott Derrickson was born in Sacramento."
        assert "See also" not in d2_vec["content"]

        # BM25 docs (keyword arg) should be ENRICHED content
        bm25_docs = inner.index_documents.call_args[1].get("bm25_documents")
        assert bm25_docs is not None

        d1_bm25 = next(d for d in bm25_docs if d["doc_id"] == "d1")
        assert "See also Scott Derrickson" in d1_bm25["content"]
        assert "Scott Derrickson directed Sinister." in d1_bm25["content"]

        d2_bm25 = next(d for d in bm25_docs if d["doc_id"] == "d2")
        assert "See also Sinister (film)" in d2_bm25["content"]

        d3_bm25 = next(d for d in bm25_docs if d["doc_id"] == "d3")
        assert "See also" not in d3_bm25["content"]

    def test_graph_stored_from_linker(self):
        """Grafo de cross-refs se construye via get_cross_ref_graph."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)

        mock_linker = MagicMock()
        mock_linker.compute_cross_refs.return_value = {}
        mock_linker.get_cross_ref_graph.side_effect = lambda doc_id: {
            "d1": ["d2"], "d2": ["d1"],
        }.get(doc_id, [])
        mock_linker.get_stats.return_value = {"total_docs": 3}

        with patch("shared.retrieval.entity_linker.HAS_SPACY", True), \
             patch("shared.retrieval.entity_linker.EntityLinker", return_value=mock_linker):
            retriever.index_documents(SAMPLE_DOCS)

        assert retriever._cross_ref_graph == {"d1": ["d2"], "d2": ["d1"]}

    def test_no_llm_enrichment_in_content(self):
        """Vector docs NO contienen cross-refs ni contexto LLM. BM25 si tiene cross-refs."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)

        mock_linker = MagicMock()
        mock_linker.compute_cross_refs.return_value = {
            "d1": "See also Scott Derrickson regarding scott derrickson.",
        }
        mock_linker.get_cross_ref_graph.return_value = []
        mock_linker.get_stats.return_value = {"total_docs": 1}

        with patch("shared.retrieval.entity_linker.HAS_SPACY", True), \
             patch("shared.retrieval.entity_linker.EntityLinker", return_value=mock_linker):
            retriever.index_documents([SAMPLE_DOCS[0]])

        # Vector docs: original, sin cross-refs
        vector_docs = inner.index_documents.call_args[0][0]
        d1_vec = vector_docs[0]["content"]
        assert "Scott Derrickson directed Sinister." in d1_vec
        assert "See also" not in d1_vec
        assert "Context" not in d1_vec

        # BM25 docs: con cross-refs inline
        bm25_docs = inner.index_documents.call_args[1].get("bm25_documents")
        assert bm25_docs is not None
        d1_bm25 = bm25_docs[0]["content"]
        assert "See also" in d1_bm25
        assert "Context" not in d1_bm25

    def test_linker_stats_stored(self):
        """Despues de indexar con spaCy, _linker_stats se almacena."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)

        expected_stats = {"total_docs": 1, "total_entities": 2}
        mock_linker = MagicMock()
        mock_linker.compute_cross_refs.return_value = {}
        mock_linker.get_cross_ref_graph.return_value = []
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
            assert "See also" not in doc["content"]

    def test_no_spacy_logs_warning(self, caplog):
        """Con HAS_SPACY=False, se emite warning."""
        import logging
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)

        with patch("shared.retrieval.entity_linker.HAS_SPACY", False), \
             caplog.at_level(logging.WARNING):
            retriever.index_documents([SAMPLE_DOCS[0]])

        assert any("spaCy no disponible" in msg for msg in caplog.messages)

    def test_no_spacy_content_is_original_and_no_bm25_docs(self):
        """Sin spaCy, contenido indexado es identico al original y bm25_documents=None."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)

        with patch("shared.retrieval.entity_linker.HAS_SPACY", False):
            retriever.index_documents([SAMPLE_DOCS[0]])

        vector_docs = inner.index_documents.call_args[0][0]
        assert vector_docs[0]["content"] == SAMPLE_DOCS[0]["content"]

        # No cross-refs -> bm25_documents should be None (shared indexing)
        bm25_docs = inner.index_documents.call_args[1].get("bm25_documents")
        assert bm25_docs is None

    def test_no_spacy_empty_graph(self):
        """Sin spaCy, el grafo queda vacio."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)

        with patch("shared.retrieval.entity_linker.HAS_SPACY", False):
            retriever.index_documents(SAMPLE_DOCS)

        assert retriever._cross_ref_graph == {}


# =========================================================================
# Test: _swap_to_original_contents (for generation)
# =========================================================================

class TestSwapToOriginalContents:

    def test_retrieve_returns_original_content(self):
        """retrieve() devuelve contenido original, no enriquecido."""
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
        assert result.contents == ["enriched content 1", "enriched content 2"]


# =========================================================================
# Test: _expand_with_graph (graph expansion during retrieval)
# =========================================================================

class TestGraphExpansion:

    def test_retrieve_expands_with_neighbors(self):
        """retrieve() anade vecinos del grafo al resultado."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)
        retriever._cross_ref_graph = {"d1": ["d3"], "d2": ["d3"]}
        retriever._original_contents = {
            "d1": "original 1", "d2": "original 2", "d3": "original 3",
        }

        result = retriever.retrieve("some query")

        assert "d3" in result.doc_ids
        # Content should be original (post-swap)
        d3_idx = result.doc_ids.index("d3")
        assert result.contents[d3_idx] == "original 3"
        assert result.metadata["graph_expanded"] >= 1

    def test_no_duplicate_expansion(self):
        """No se duplican docs que ya estan en el resultado."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)
        retriever._cross_ref_graph = {"d2": ["d1"]}
        retriever._original_contents = {"d1": "orig1", "d2": "orig2"}

        result = retriever.retrieve("query")

        assert result.doc_ids.count("d1") == 1
        assert result.metadata["graph_expanded"] == 0

    def test_no_graph_no_expansion(self):
        """Sin grafo, no hay expansion."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)
        retriever._cross_ref_graph = {}
        retriever._original_contents = {"d1": "orig1", "d2": "orig2"}

        result = retriever.retrieve("query")

        assert len(result.doc_ids) == 2
        assert result.metadata["graph_expanded"] == 0

    def test_neighbor_scores_lower_than_original(self):
        """Vecinos expandidos tienen scores menores que los originales."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)
        retriever._cross_ref_graph = {"d1": ["d3"]}
        retriever._original_contents = {"d1": "o1", "d2": "o2", "d3": "o3"}

        result = retriever.retrieve("query")

        d3_idx = result.doc_ids.index("d3")
        d3_score = result.scores[d3_idx]
        original_min = min(result.scores[:2])
        assert d3_score < original_min

    def test_graph_expansion_respects_cap(self):
        """Graph expansion stops after max_graph_expansion docs."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)
        # Set a very low cap
        retriever.config.max_graph_expansion = 1
        # d1 has 3 neighbors, but only 1 should be added
        retriever._cross_ref_graph = {"d1": ["d3", "d4", "d5"]}
        retriever._original_contents = {
            "d1": "o1", "d2": "o2",
            "d3": "o3", "d4": "o4", "d5": "o5",
        }

        result = retriever.retrieve("query")

        assert result.metadata["graph_expanded"] == 1

    def test_graph_expansion_no_cap_when_zero(self):
        """max_graph_expansion=0 means unlimited expansion."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)
        retriever.config.max_graph_expansion = 0
        retriever._cross_ref_graph = {"d1": ["d3", "d4", "d5"]}
        retriever._original_contents = {
            "d1": "o1", "d2": "o2",
            "d3": "o3", "d4": "o4", "d5": "o5",
        }

        result = retriever.retrieve("query")

        assert result.metadata["graph_expanded"] == 3

    def test_retrieve_by_vector_also_expands(self):
        """retrieve_by_vector() tambien aplica graph expansion."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)
        retriever._cross_ref_graph = {"d1": ["d3"]}
        retriever._original_contents = {"d1": "o1", "d3": "o3"}

        result = retriever.retrieve_by_vector("query", [0.1, 0.2])

        assert "d3" in result.doc_ids
        assert result.strategy_used == RetrievalStrategy.HYBRID_PLUS


# =========================================================================
# Test: clear_index
# =========================================================================

class TestClearIndex:

    def test_clear_resets_all_state(self):
        """clear_index() limpia inner, originals, graph, stats y flag."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)

        retriever._original_contents = {"d1": "x", "d2": "y"}
        retriever._cross_ref_graph = {"d1": ["d2"]}
        retriever._linker_stats = {"total_docs": 2}
        retriever._is_indexed = True

        retriever.clear_index()

        inner.clear_index.assert_called_once()
        assert retriever._original_contents == {}
        assert retriever._cross_ref_graph == {}
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
