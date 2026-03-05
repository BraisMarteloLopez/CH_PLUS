"""
Tests unitarios para HybridPlusRetriever.

Cubre:
  - index_documents con graph expansion (mock spaCy + inner retriever)
  - index_documents sin spaCy (graceful degradation, BM25+Vector+RRF puro)
  - _expand_with_graph (graph expansion, metadata, strategy_used)
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
        contents=["content 1", "content 2"],
        scores=[0.9, 0.8],
        strategy_used=RetrievalStrategy.HYBRID_PLUS,
        metadata={},
    )
    inner.retrieve_by_vector.return_value = RetrievalResult(
        doc_ids=["d1"],
        contents=["content 1"],
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
# Test: index_documents con graph (contenido limpio)
# =========================================================================

class TestIndexDocumentsWithGraph:

    def test_inner_receives_clean_content(self):
        """Con HAS_SPACY=True, inner retriever recibe docs con contenido LIMPIO."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)

        mock_linker = MagicMock()
        mock_linker.compute_cross_ref_graph.return_value = {
            "d1": ["d2"],
            "d2": ["d1"],
        }
        mock_linker.get_stats.return_value = {"total_docs": 3, "total_entities": 4}

        with patch("shared.retrieval.entity_linker.HAS_SPACY", True), \
             patch("shared.retrieval.entity_linker.EntityLinker", return_value=mock_linker):
            result = retriever.index_documents(SAMPLE_DOCS)

        assert result is True

        indexed_docs = inner.index_documents.call_args[0][0]

        # Contenido debe ser ORIGINAL, sin "Related:" contaminando
        for doc, sample in zip(indexed_docs, SAMPLE_DOCS):
            assert doc["content"] == sample["content"]
            assert "Related:" not in doc["content"]

    def test_graph_stored(self):
        """Grafo de cross-refs se almacena correctamente."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)

        expected_graph = {"d1": ["d2"], "d2": ["d1"]}
        mock_linker = MagicMock()
        mock_linker.compute_cross_ref_graph.return_value = expected_graph
        mock_linker.get_stats.return_value = {"total_docs": 3}

        with patch("shared.retrieval.entity_linker.HAS_SPACY", True), \
             patch("shared.retrieval.entity_linker.EntityLinker", return_value=mock_linker):
            retriever.index_documents(SAMPLE_DOCS)

        assert retriever._cross_ref_graph == expected_graph

    def test_linker_stats_stored(self):
        """Despues de indexar con spaCy, _linker_stats se almacena."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)

        expected_stats = {"total_docs": 3, "total_entities": 4}
        mock_linker = MagicMock()
        mock_linker.compute_cross_ref_graph.return_value = {}
        mock_linker.get_stats.return_value = expected_stats

        with patch("shared.retrieval.entity_linker.HAS_SPACY", True), \
             patch("shared.retrieval.entity_linker.EntityLinker", return_value=mock_linker):
            retriever.index_documents(SAMPLE_DOCS)

        assert retriever._linker_stats == expected_stats

    def test_empty_documents_returns_false(self):
        """Lista de documentos vacia retorna False."""
        retriever = _make_retriever()
        assert retriever.index_documents([]) is False

    def test_doc_content_map_stored(self):
        """Mapa doc_id -> contenido se almacena para graph expansion."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)

        with patch("shared.retrieval.entity_linker.HAS_SPACY", False):
            retriever.index_documents(SAMPLE_DOCS[:2])

        assert retriever._doc_content_map["d1"] == "Scott Derrickson directed Sinister."
        assert retriever._doc_content_map["d2"] == "Scott Derrickson was born in Sacramento."


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

        indexed_docs = inner.index_documents.call_args[0][0]
        for doc in indexed_docs:
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

        indexed_docs = inner.index_documents.call_args[0][0]
        assert indexed_docs[0]["content"] == SAMPLE_DOCS[0]["content"]

    def test_no_spacy_empty_graph(self):
        """Sin spaCy, el grafo queda vacio."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)

        with patch("shared.retrieval.entity_linker.HAS_SPACY", False):
            retriever.index_documents(SAMPLE_DOCS)

        assert retriever._cross_ref_graph == {}


# =========================================================================
# Test: _expand_with_graph (graph expansion during retrieval)
# =========================================================================

class TestGraphExpansion:

    def test_retrieve_expands_with_neighbors(self):
        """retrieve() anade vecinos del grafo al resultado."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)
        retriever._cross_ref_graph = {"d1": ["d3"], "d2": ["d3"]}
        retriever._doc_content_map = {
            "d1": "content 1",
            "d2": "content 2",
            "d3": "content 3",
        }

        result = retriever.retrieve("some query")

        # d3 deberia aparecer como vecino expandido
        assert "d3" in result.doc_ids
        assert "content 3" in result.contents
        assert result.metadata["graph_expanded"] >= 1

    def test_no_duplicate_expansion(self):
        """No se duplican docs que ya estan en el resultado."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)
        # d1 ya esta en resultado, no debe duplicarse
        retriever._cross_ref_graph = {"d2": ["d1"]}
        retriever._doc_content_map = {"d1": "content 1", "d2": "content 2"}

        result = retriever.retrieve("query")

        # d1 solo debe aparecer una vez
        assert result.doc_ids.count("d1") == 1
        assert result.metadata["graph_expanded"] == 0

    def test_no_graph_no_expansion(self):
        """Sin grafo, no hay expansion."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)
        retriever._cross_ref_graph = {}

        result = retriever.retrieve("query")

        assert result.doc_ids == ["d1", "d2"]
        assert result.metadata["graph_expanded"] == 0
        assert result.metadata["entity_cross_linking"] is False

    def test_retrieve_sets_strategy(self):
        """retrieve() marca strategy_used como HYBRID_PLUS."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)
        retriever._cross_ref_graph = {}

        result = retriever.retrieve("query")
        assert result.strategy_used == RetrievalStrategy.HYBRID_PLUS

    def test_retrieve_sets_metadata_with_graph(self):
        """retrieve() establece metadata de cross-linking cuando hay grafo."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)
        retriever._cross_ref_graph = {"d1": ["d3"]}
        retriever._doc_content_map = {"d3": "content 3"}
        retriever._linker_stats = {"total_docs": 3}

        result = retriever.retrieve("query")

        assert result.metadata["entity_cross_linking"] is True
        assert result.metadata["linker_stats"] == {"total_docs": 3}

    def test_retrieve_by_vector_also_expands(self):
        """retrieve_by_vector() tambien aplica graph expansion."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)
        retriever._cross_ref_graph = {"d1": ["d3"]}
        retriever._doc_content_map = {"d3": "content 3"}

        result = retriever.retrieve_by_vector("query", [0.1, 0.2])

        assert "d3" in result.doc_ids
        assert result.strategy_used == RetrievalStrategy.HYBRID_PLUS

    def test_neighbor_scores_lower_than_original(self):
        """Vecinos expandidos tienen scores menores que los originales."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)
        retriever._cross_ref_graph = {"d1": ["d3"]}
        retriever._doc_content_map = {"d3": "content 3"}

        result = retriever.retrieve("query")

        # d3 es el ultimo (expandido)
        d3_idx = result.doc_ids.index("d3")
        d3_score = result.scores[d3_idx]

        # Score de vecino debe ser menor que cualquier score original
        original_min = min(result.scores[:2])
        assert d3_score < original_min


# =========================================================================
# Test: clear_index
# =========================================================================

class TestClearIndex:

    def test_clear_resets_all_state(self):
        """clear_index() limpia inner, graph, content_map, stats y flag."""
        inner = _make_mock_inner_retriever()
        retriever = _make_retriever(inner=inner)

        retriever._cross_ref_graph = {"d1": ["d2"]}
        retriever._doc_content_map = {"d1": "x", "d2": "y"}
        retriever._linker_stats = {"total_docs": 2}
        retriever._is_indexed = True

        retriever.clear_index()

        inner.clear_index.assert_called_once()
        assert retriever._cross_ref_graph == {}
        assert retriever._doc_content_map == {}
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
