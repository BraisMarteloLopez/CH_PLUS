"""
Test DT-7: Sin reranker → reranked_status=None, rerank_failures=None.
"""
from shared.retrieval.core import RetrievalResult, RetrievalStrategy, RetrievalConfig
from shared.config_base import InfraConfig, RerankerConfig
from shared.types import LoadedDataset, NormalizedDocument


class MockRetriever:
    def retrieve(self, query_text, top_k=None):
        k = top_k or 20
        return RetrievalResult(
            doc_ids=[f"doc_{i}" for i in range(k)],
            contents=[f"content_{i}" for i in range(k)],
            scores=[1.0 - i * 0.01 for i in range(k)],
            strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
        )

    def retrieve_by_vector(self, query_text, query_vector, top_k=None):
        return self.retrieve(query_text, top_k)


def test_no_reranker():
    """Sin reranker: reranked_status=None, generation_doc_ids vacio, config_snapshot rerank_failures=None."""
    from sandbox_mteb.evaluator import MTEBEvaluator
    from sandbox_mteb.config import MTEBConfig, MinIOStorageConfig

    config = MTEBConfig(
        infra=InfraConfig(),
        storage=MinIOStorageConfig(),
        retrieval=RetrievalConfig(retrieval_k=20),
        reranker=RerankerConfig(enabled=False),
    )
    evaluator = MTEBEvaluator(config)
    evaluator._retriever = MockRetriever()

    detail, reranked_status = evaluator._execute_retrieval("test query", ["doc_0"])

    assert reranked_status is None
    assert len(detail.generation_doc_ids) == 0

    # config_snapshot tambien refleja None
    dataset = LoadedDataset(name="test", corpus={"doc_0": NormalizedDocument("doc_0", "c")})
    run = evaluator._build_run("test_run", dataset, [], 1.0, 1)
    assert run.config_snapshot["rerank_failures"] is None
