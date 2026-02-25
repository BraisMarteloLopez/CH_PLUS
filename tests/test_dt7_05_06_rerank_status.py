"""
Test DT-7 #5: Rerank exitoso -> reranked_ok=True, _rerank_failures=0.
Test DT-7 #6: Rerank fallido -> reranked_ok=False, _rerank_failures incrementa.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from shared.retrieval.core import RetrievalResult, RetrievalStrategy, RetrievalConfig
from shared.config_base import InfraConfig, RerankerConfig


# ---------------------------------------------------------------
# Mocks minimos
# ---------------------------------------------------------------

class MockRetriever:
    """Retorna documentos fijos sin llamar a ningun servicio."""

    def retrieve_by_vector(self, query_text, query_vector, top_k=None):
        k = top_k or 20
        return RetrievalResult(
            doc_ids=[f"doc_{i}" for i in range(k)],
            contents=[f"content_{i}" for i in range(k)],
            scores=[1.0 - i * 0.01 for i in range(k)],
            strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
        )

    def retrieve(self, query_text, top_k=None):
        return self.retrieve_by_vector(query_text, None, top_k)


class MockRerankerSuccess:
    """Simula rerank exitoso."""

    def __init__(self):
        self.top_n = 5

    def rerank(self, query, retrieval_result, top_n=5):
        return RetrievalResult(
            doc_ids=retrieval_result.doc_ids[:top_n],
            contents=retrieval_result.contents[:top_n],
            scores=[0.9, 0.8, 0.7, 0.6, 0.5][:top_n],
            retrieval_time_ms=retrieval_result.retrieval_time_ms + 10.0,
            strategy_used=retrieval_result.strategy_used,
            metadata={"reranked": True, "reranker_model": "mock"},
        )


class MockRerankerFail:
    """Simula rerank que falla (exception en compress_documents).
    Replica el fallback de CrossEncoderReranker.rerank()."""

    def __init__(self):
        self.top_n = 5

    def rerank(self, query, retrieval_result, top_n=5):
        # Simula el fallback exacto de reranker.py lineas 128-143
        return RetrievalResult(
            doc_ids=retrieval_result.doc_ids[:top_n],
            contents=retrieval_result.contents[:top_n],
            scores=retrieval_result.scores[:top_n],
            retrieval_time_ms=retrieval_result.retrieval_time_ms,
            strategy_used=retrieval_result.strategy_used,
            metadata={
                **retrieval_result.metadata,
                "reranked": False,
                "rerank_error": "simulated timeout",
            },
        )


def _make_evaluator(reranker):
    """Construye MTEBEvaluator con mocks inyectados, sin infra real."""
    from sandbox_mteb.evaluator import MTEBEvaluator
    from sandbox_mteb.config import MTEBConfig, MinIOStorageConfig

    config = MTEBConfig(
        infra=InfraConfig(),
        storage=MinIOStorageConfig(),
        retrieval=RetrievalConfig(retrieval_k=20, pre_fusion_k=150),
        reranker=RerankerConfig(enabled=True, top_n=5),
    )
    evaluator = MTEBEvaluator(config)
    evaluator._retriever = MockRetriever()
    evaluator._reranker = reranker
    return evaluator


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------

def test_rerank_success():
    evaluator = _make_evaluator(MockRerankerSuccess())

    detail, reranked_ok = evaluator._execute_retrieval(
        "test query", ["doc_0", "doc_1"]
    )

    assert reranked_ok is True, f"Esperado True, obtenido {reranked_ok}"
    assert evaluator._rerank_failures == 0, (
        f"Esperado 0 failures, obtenido {evaluator._rerank_failures}"
    )
    assert len(detail.generation_doc_ids) == 5, (
        f"Esperado 5 docs post-rerank, obtenido {len(detail.generation_doc_ids)}"
    )
    print("PASS: rerank exitoso -> reranked_ok=True, _rerank_failures=0")


def test_rerank_failure():
    evaluator = _make_evaluator(MockRerankerFail())

    detail, reranked_ok = evaluator._execute_retrieval(
        "test query", ["doc_0", "doc_1"]
    )

    assert reranked_ok is False, f"Esperado False, obtenido {reranked_ok}"
    assert evaluator._rerank_failures == 1, (
        f"Esperado 1 failure, obtenido {evaluator._rerank_failures}"
    )

    # Segundo fallo: contador debe incrementar
    _, reranked_ok2 = evaluator._execute_retrieval(
        "another query", ["doc_0"]
    )
    assert reranked_ok2 is False
    assert evaluator._rerank_failures == 2, (
        f"Esperado 2 failures, obtenido {evaluator._rerank_failures}"
    )

    print("PASS: rerank fallido -> reranked_ok=False, _rerank_failures incrementa")


if __name__ == "__main__":
    test_rerank_success()
    test_rerank_failure()
