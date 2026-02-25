"""
Test DT-8 #9:  Docs en orden ascendente -> sort los reordena descendente.
Test DT-8 #10: Docs con scores identicos -> no falla.
Test DT-8 #11: Doc sin relevance_score -> default 0.0, queda al final.
"""
from unittest.mock import MagicMock

from shared.retrieval.core import RetrievalResult, RetrievalStrategy
from shared.retrieval.reranker import CrossEncoderReranker


class FakeDocument:
    """Minimo para simular LangChain Document."""
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def _make_reranker_with_mock_compress(fake_docs):
    """Construye CrossEncoderReranker con compress_documents mockeado."""
    reranker = object.__new__(CrossEncoderReranker)
    reranker.base_url = "mock"
    reranker.model_name = "mock"
    reranker._reranker = MagicMock()
    reranker._reranker.compress_documents = MagicMock(return_value=fake_docs)
    return reranker


def _make_retrieval_result(n=10):
    return RetrievalResult(
        doc_ids=[f"doc_{i}" for i in range(n)],
        contents=[f"content_{i}" for i in range(n)],
        scores=[1.0 - i * 0.1 for i in range(n)],
        strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
    )


def test_ascending_order_gets_sorted():
    """Docs llegan en orden ascendente (0.1, 0.5, 0.9) -> sort produce (0.9, 0.5, 0.1)."""
    fake_docs = [
        FakeDocument("c_low", {"doc_id": "low", "relevance_score": 0.1}),
        FakeDocument("c_mid", {"doc_id": "mid", "relevance_score": 0.5}),
        FakeDocument("c_high", {"doc_id": "high", "relevance_score": 0.9}),
    ]

    reranker = _make_reranker_with_mock_compress(fake_docs)
    result = reranker.rerank("query", _make_retrieval_result(), top_n=3)

    assert result.doc_ids[0] == "high", (
        f"Primer doc deberia ser 'high' (0.9), obtenido '{result.doc_ids[0]}'"
    )
    assert result.doc_ids[1] == "mid"
    assert result.doc_ids[2] == "low"
    assert result.scores == [0.9, 0.5, 0.1], (
        f"Scores desordenados: {result.scores}"
    )

    print("PASS: docs en orden ascendente se reordenan descendente")


def test_identical_scores():
    """Docs con scores identicos no producen error."""
    fake_docs = [
        FakeDocument("c_a", {"doc_id": "a", "relevance_score": 0.7}),
        FakeDocument("c_b", {"doc_id": "b", "relevance_score": 0.7}),
        FakeDocument("c_c", {"doc_id": "c", "relevance_score": 0.7}),
    ]

    reranker = _make_reranker_with_mock_compress(fake_docs)
    result = reranker.rerank("query", _make_retrieval_result(), top_n=3)

    assert len(result.doc_ids) == 3, f"Esperado 3 docs, obtenido {len(result.doc_ids)}"
    assert all(s == 0.7 for s in result.scores), f"Scores inesperados: {result.scores}"

    print("PASS: scores identicos no fallan")


def test_missing_relevance_score():
    """Doc sin relevance_score en metadata -> default 0.0, queda al final."""
    fake_docs = [
        FakeDocument("c_no_score", {"doc_id": "no_score"}),  # sin relevance_score
        FakeDocument("c_high", {"doc_id": "high", "relevance_score": 0.9}),
        FakeDocument("c_mid", {"doc_id": "mid", "relevance_score": 0.5}),
    ]

    reranker = _make_reranker_with_mock_compress(fake_docs)
    result = reranker.rerank("query", _make_retrieval_result(), top_n=3)

    assert result.doc_ids[-1] == "no_score", (
        f"Doc sin score deberia ser ultimo, obtenido orden: {result.doc_ids}"
    )
    assert result.scores[-1] == 0.0, (
        f"Score default deberia ser 0.0, obtenido {result.scores[-1]}"
    )

    print("PASS: doc sin relevance_score queda al final con score 0.0")


if __name__ == "__main__":
    test_ascending_order_gets_sorted()
    test_identical_scores()
    test_missing_relevance_score()
