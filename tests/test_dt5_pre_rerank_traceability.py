"""
Tests DT-5: Trazabilidad de candidatos pre-rerank via _execute_retrieval.

Verifica que:
  - Con reranker: pre_rerank_candidate_ids contiene el pool completo
  - generation_doc_ids es subconjunto de pre_rerank_candidate_ids
  - Sin reranker: pre_rerank_candidate_ids queda vacio
  - Doc promovido por reranker es trazable a posicion original

Nota: la serializacion condicional de estos campos en to_dict() se
testea en test_dtm4_build_run_aggregation.py (test_query_result_to_dict_conditional_fields).
"""
from unittest.mock import MagicMock

from shared.retrieval.core import RetrievalResult, RetrievalConfig
from shared.config_base import InfraConfig, RerankerConfig


def _make_evaluator(reranker_enabled=True):
    from sandbox_mteb.evaluator import MTEBEvaluator
    from sandbox_mteb.config import MTEBConfig, MinIOStorageConfig

    config = MTEBConfig(
        infra=InfraConfig(),
        storage=MinIOStorageConfig(),
        retrieval=RetrievalConfig(retrieval_k=5, pre_fusion_k=20),
        reranker=RerankerConfig(enabled=reranker_enabled, top_n=3),
        generation_enabled=True,
    )
    return MTEBEvaluator(config)


def _mock_retriever(doc_ids):
    mock = MagicMock()
    result = RetrievalResult(
        doc_ids=doc_ids,
        contents=[f"content_{did}" for did in doc_ids],
        scores=[1.0 - i * 0.01 for i in range(len(doc_ids))],
        retrieval_time_ms=10.0,
    )
    mock.retrieve.return_value = result
    mock.retrieve_by_vector.return_value = result
    return mock


def _mock_reranker(top_ids):
    mock = MagicMock()
    mock.rerank.return_value = RetrievalResult(
        doc_ids=top_ids,
        contents=[f"content_{did}" for did in top_ids],
        scores=[0.99 - i * 0.1 for i in range(len(top_ids))],
        retrieval_time_ms=15.0,
        metadata={"reranked": True},
    )
    return mock


def test_pre_rerank_ids_populated_with_reranker():
    """Con reranker: pre_rerank_candidate_ids = pool completo, generation_doc_ids = post-rerank."""
    evaluator = _make_evaluator(reranker_enabled=True)
    all_ids = [f"doc_{i:03d}" for i in range(20)]
    reranked_top = ["doc_015", "doc_003", "doc_007"]

    evaluator._retriever = _mock_retriever(all_ids)
    evaluator._reranker = _mock_reranker(reranked_top)

    detail, reranked_ok = evaluator._execute_retrieval("test query", ["doc_003"])

    assert detail.pre_rerank_candidate_ids == all_ids
    assert detail.generation_doc_ids == reranked_top
    assert set(detail.generation_doc_ids).issubset(set(detail.pre_rerank_candidate_ids))


def test_pre_rerank_ids_empty_without_reranker():
    """Sin reranker: pre_rerank_candidate_ids vacio, reranked_status=None."""
    evaluator = _make_evaluator(reranker_enabled=False)
    evaluator._retriever = _mock_retriever([f"doc_{i:03d}" for i in range(5)])
    evaluator._reranker = None

    detail, reranked_ok = evaluator._execute_retrieval("test query", ["doc_001"])

    assert detail.pre_rerank_candidate_ids == []
    assert reranked_ok is None


def test_reranker_promotes_low_ranked_doc():
    """Doc en posicion 16 promovido al top 3: trazable via pre_rerank_candidate_ids."""
    evaluator = _make_evaluator(reranker_enabled=True)
    all_ids = [f"doc_{i:03d}" for i in range(20)]
    evaluator._retriever = _mock_retriever(all_ids)
    evaluator._reranker = _mock_reranker(["doc_015", "doc_000", "doc_001"])

    detail, _ = evaluator._execute_retrieval("test", ["doc_015"])

    # doc_015 NO en top 5 pre-rerank, SI en generation y pre_rerank
    assert "doc_015" not in detail.retrieved_doc_ids
    assert "doc_015" in detail.generation_doc_ids
    assert detail.pre_rerank_candidate_ids.index("doc_015") == 15
