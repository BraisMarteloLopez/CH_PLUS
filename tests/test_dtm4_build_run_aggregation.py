"""
Test de integracion para _build_run() en evaluator.py.

Cobertura DTm-4:
  Valida la agregacion matematica de metricas individuales (query-level)
  a metricas agregadas (run-level) en EvaluationRun. Este es el paso
  final del pipeline: errores aqui corrompen todo el JSON de salida.

  - avg_hit_rate_at_5 correcto
  - avg_mrr correcto
  - avg_recall_at_k y avg_ndcg_at_k correctos para cada K
  - retrieval_complement_recall_at_k = 1 - avg_recall_at_k
  - avg_generation_score incluye zeros (no los filtra)
  - num_queries_evaluated / num_queries_failed correcto
  - config_snapshot contiene campos esperados
  - Sin queries completadas: metricas a 0
  - Queries mixtas (completed + failed): solo completed cuentan

Sin dependencias externas.
"""
from shared.types import (
    EvaluationRun,
    EvaluationStatus,
    LoadedDataset,
    NormalizedQuery,
    NormalizedDocument,
    QueryRetrievalDetail,
    GenerationResult,
    QueryEvaluationResult,
    DatasetType,
    MetricType,
)
from sandbox_mteb.evaluator import MTEBEvaluator
from sandbox_mteb.config import MTEBConfig, MinIOStorageConfig
from shared.config_base import InfraConfig, RerankerConfig
from shared.retrieval.core import RetrievalConfig


# =================================================================
# Helpers
# =================================================================

def _make_evaluator(**kwargs) -> MTEBEvaluator:
    config = MTEBConfig(
        infra=InfraConfig(),
        storage=MinIOStorageConfig(),
        retrieval=RetrievalConfig(),
        reranker=RerankerConfig(enabled=kwargs.get("reranker_enabled", False)),
        generation_enabled=kwargs.get("generation_enabled", True),
    )
    return MTEBEvaluator(config)


def _make_dataset() -> LoadedDataset:
    return LoadedDataset(
        name="test_dataset",
        dataset_type=DatasetType.HYBRID,
        queries=[
            NormalizedQuery(query_id="q1", query_text="Q1"),
            NormalizedQuery(query_id="q2", query_text="Q2"),
        ],
        corpus={"d1": NormalizedDocument(doc_id="d1", content="C1")},
    )


def _make_query_result(
    query_id: str,
    hit_at_5: float,
    mrr: float,
    recall_at_k: dict,
    ndcg_at_k: dict,
    primary_value: float = 0.5,
    status: EvaluationStatus = EvaluationStatus.COMPLETED,
    n_retrieved: int = 5,
    n_expected: int = 2,
) -> QueryEvaluationResult:
    """Crea un QueryEvaluationResult sintetico con metricas conocidas."""
    retrieval = QueryRetrievalDetail(
        retrieved_doc_ids=[f"d{i}" for i in range(n_retrieved)],
        retrieved_contents=[f"content_{i}" for i in range(n_retrieved)],
        retrieval_scores=[1.0 - i * 0.1 for i in range(n_retrieved)],
        expected_doc_ids=[f"d{i}" for i in range(n_expected)],
    )
    # Override metricas calculadas automaticamente con valores conocidos
    retrieval.hit_at_k = {5: hit_at_5, 1: 0.0, 3: 0.0, 10: hit_at_5, 20: hit_at_5}
    retrieval.mrr = mrr
    retrieval.recall_at_k = recall_at_k
    retrieval.ndcg_at_k = ndcg_at_k

    generation = GenerationResult(
        generated_response="answer", generation_time_ms=100.0
    ) if status == EvaluationStatus.COMPLETED else None

    return QueryEvaluationResult(
        query_id=query_id,
        query_text=f"Question {query_id}",
        dataset_name="test",
        dataset_type=DatasetType.HYBRID,
        retrieval=retrieval,
        generation=generation,
        primary_metric_type=MetricType.F1_SCORE,
        primary_metric_value=primary_value,
        status=status,
    )


# =================================================================
# Tests
# =================================================================

def test_avg_hit_rate_at_5():
    """avg_hit_rate_at_5 = mean de hit_at_k[5] de queries completadas."""
    ev = _make_evaluator()
    dataset = _make_dataset()

    qr1 = _make_query_result("q1", hit_at_5=1.0, mrr=1.0,
                              recall_at_k={5: 1.0}, ndcg_at_k={5: 1.0})
    qr2 = _make_query_result("q2", hit_at_5=0.0, mrr=0.0,
                              recall_at_k={5: 0.0}, ndcg_at_k={5: 0.0})

    run = ev._build_run("test_run", dataset, [qr1, qr2], 10.0, 100)

    expected = (1.0 + 0.0) / 2.0
    assert abs(run.avg_hit_rate_at_5 - expected) < 1e-10, (
        f"avg_hit_rate_at_5: {run.avg_hit_rate_at_5} != {expected}"
    )


def test_avg_mrr():
    """avg_mrr = mean de mrr de queries completadas."""
    ev = _make_evaluator()
    dataset = _make_dataset()

    qr1 = _make_query_result("q1", hit_at_5=1.0, mrr=1.0,
                              recall_at_k={5: 1.0}, ndcg_at_k={5: 1.0})
    qr2 = _make_query_result("q2", hit_at_5=1.0, mrr=0.5,
                              recall_at_k={5: 0.5}, ndcg_at_k={5: 0.5})

    run = ev._build_run("test_run", dataset, [qr1, qr2], 10.0, 100)

    expected = (1.0 + 0.5) / 2.0
    assert abs(run.avg_mrr - expected) < 1e-10


def test_avg_recall_at_k():
    """avg_recall_at_k agrega correctamente por cada valor de K."""
    ev = _make_evaluator()
    dataset = _make_dataset()

    qr1 = _make_query_result("q1", hit_at_5=1.0, mrr=1.0,
                              recall_at_k={1: 0.5, 5: 1.0, 10: 1.0},
                              ndcg_at_k={1: 0.5, 5: 1.0, 10: 1.0})
    qr2 = _make_query_result("q2", hit_at_5=0.0, mrr=0.0,
                              recall_at_k={1: 0.0, 5: 0.5, 10: 0.5},
                              ndcg_at_k={1: 0.0, 5: 0.5, 10: 0.5})

    run = ev._build_run("test_run", dataset, [qr1, qr2], 10.0, 100)

    assert abs(run.avg_recall_at_k[1] - 0.25) < 1e-10
    assert abs(run.avg_recall_at_k[5] - 0.75) < 1e-10
    assert abs(run.avg_recall_at_k[10] - 0.75) < 1e-10


def test_complement_recall_is_one_minus_recall():
    """retrieval_complement_recall_at_k = 1 - avg_recall_at_k para cada K."""
    ev = _make_evaluator()
    dataset = _make_dataset()

    qr1 = _make_query_result("q1", hit_at_5=1.0, mrr=1.0,
                              recall_at_k={5: 0.8, 10: 0.9},
                              ndcg_at_k={5: 0.8, 10: 0.9})
    qr2 = _make_query_result("q2", hit_at_5=1.0, mrr=0.5,
                              recall_at_k={5: 0.6, 10: 0.7},
                              ndcg_at_k={5: 0.6, 10: 0.7})

    run = ev._build_run("test_run", dataset, [qr1, qr2], 10.0, 100)

    for k in [5, 10]:
        expected_complement = 1.0 - run.avg_recall_at_k[k]
        actual_complement = run.retrieval_complement_recall_at_k[k]
        assert abs(actual_complement - expected_complement) < 1e-10, (
            f"complement_recall@{k}: {actual_complement} != {expected_complement}"
        )


def test_generation_score_includes_zeros():
    """avg_generation_score incluye queries con primary_value=0.0."""
    ev = _make_evaluator()
    dataset = _make_dataset()

    qr1 = _make_query_result("q1", hit_at_5=1.0, mrr=1.0,
                              recall_at_k={5: 1.0}, ndcg_at_k={5: 1.0},
                              primary_value=0.8)
    qr2 = _make_query_result("q2", hit_at_5=0.0, mrr=0.0,
                              recall_at_k={5: 0.0}, ndcg_at_k={5: 0.0},
                              primary_value=0.0)

    run = ev._build_run("test_run", dataset, [qr1, qr2], 10.0, 100)

    # Si incluyera solo non-zero, seria 0.8. Con zeros: (0.8 + 0.0) / 2 = 0.4
    expected = (0.8 + 0.0) / 2.0
    assert run.avg_generation_score is not None
    assert abs(run.avg_generation_score - expected) < 1e-10, (
        f"avg_generation_score: {run.avg_generation_score} != {expected} "
        "(zeros no incluidos?)"
    )


def test_failed_queries_not_counted_in_metrics():
    """Solo queries COMPLETED contribuyen a metricas. FAILED se excluyen."""
    ev = _make_evaluator()
    dataset = _make_dataset()

    qr1 = _make_query_result("q1", hit_at_5=1.0, mrr=1.0,
                              recall_at_k={5: 1.0}, ndcg_at_k={5: 1.0},
                              primary_value=0.9)
    qr2 = _make_query_result("q2", hit_at_5=0.0, mrr=0.0,
                              recall_at_k={5: 0.0}, ndcg_at_k={5: 0.0},
                              status=EvaluationStatus.FAILED)

    run = ev._build_run("test_run", dataset, [qr1, qr2], 10.0, 100)

    # Solo q1 cuenta (q2 esta FAILED)
    assert run.num_queries_evaluated == 1
    assert run.num_queries_failed == 1
    assert abs(run.avg_hit_rate_at_5 - 1.0) < 1e-10
    assert abs(run.avg_mrr - 1.0) < 1e-10


def test_no_completed_queries_zeros():
    """Sin queries completadas, todas las metricas son 0."""
    ev = _make_evaluator()
    dataset = _make_dataset()

    qr1 = _make_query_result("q1", hit_at_5=1.0, mrr=1.0,
                              recall_at_k={5: 1.0}, ndcg_at_k={5: 1.0},
                              status=EvaluationStatus.FAILED)

    run = ev._build_run("test_run", dataset, [qr1], 10.0, 100)

    assert run.num_queries_evaluated == 0
    assert run.num_queries_failed == 1
    assert run.avg_hit_rate_at_5 == 0.0
    assert run.avg_mrr == 0.0
    assert run.avg_recall_at_k == {}
    assert run.avg_ndcg_at_k == {}


def test_config_snapshot_fields():
    """config_snapshot contiene campos de configuracion esperados."""
    ev = _make_evaluator()
    dataset = _make_dataset()

    qr1 = _make_query_result("q1", hit_at_5=1.0, mrr=1.0,
                              recall_at_k={5: 1.0}, ndcg_at_k={5: 1.0})

    run = ev._build_run("test_run", dataset, [qr1], 10.0, 100)

    snapshot = run.config_snapshot
    required_keys = [
        "retrieval_strategy", "retrieval_k", "pre_fusion_k",
        "bm25_weight", "vector_weight", "rrf_k",
        "corpus_shuffle_seed", "max_queries", "max_corpus",
        "generation_enabled", "max_context_chars",
        "reranker_enabled", "corpus_total_available", "corpus_indexed",
    ]
    for key in required_keys:
        assert key in snapshot, f"Falta '{key}' en config_snapshot"


def test_corpus_indexed_size():
    """total_documents refleja corpus_indexed_size pasado."""
    ev = _make_evaluator()
    dataset = _make_dataset()

    qr1 = _make_query_result("q1", hit_at_5=1.0, mrr=1.0,
                              recall_at_k={5: 1.0}, ndcg_at_k={5: 1.0})

    run = ev._build_run("test_run", dataset, [qr1], 10.0, indexed_corpus_size=500)

    assert run.total_documents == 500
    assert run.config_snapshot["corpus_indexed"] == 500


def test_run_metadata():
    """EvaluationRun tiene run_id, dataset_name, status correctos."""
    ev = _make_evaluator()
    dataset = _make_dataset()

    qr1 = _make_query_result("q1", hit_at_5=1.0, mrr=1.0,
                              recall_at_k={5: 1.0}, ndcg_at_k={5: 1.0})

    run = ev._build_run("my_run_id", dataset, [qr1], 42.5, 100)

    assert run.run_id == "my_run_id"
    assert run.dataset_name == "hotpotqa"  # default de MTEBConfig
    assert run.status == EvaluationStatus.COMPLETED
    assert abs(run.execution_time_seconds - 42.5) < 1e-10


def test_single_query_no_averaging_error():
    """Con una sola query, metricas deben ser iguales a las de la query."""
    ev = _make_evaluator()
    dataset = _make_dataset()

    qr = _make_query_result("q1", hit_at_5=1.0, mrr=0.333,
                             recall_at_k={5: 0.5, 10: 0.75},
                             ndcg_at_k={5: 0.6, 10: 0.8},
                             primary_value=0.7)

    run = ev._build_run("test_run", dataset, [qr], 10.0, 100)

    assert abs(run.avg_hit_rate_at_5 - 1.0) < 1e-10
    assert abs(run.avg_mrr - 0.333) < 1e-10
    assert abs(run.avg_recall_at_k[5] - 0.5) < 1e-10
    assert abs(run.avg_recall_at_k[10] - 0.75) < 1e-10
    assert abs(run.avg_generation_score - 0.7) < 1e-10


def test_gen_zero_count_in_snapshot():
    """config_snapshot reporta gen_zero_count y gen_nonzero_count."""
    ev = _make_evaluator()
    dataset = _make_dataset()

    qr1 = _make_query_result("q1", hit_at_5=1.0, mrr=1.0,
                              recall_at_k={5: 1.0}, ndcg_at_k={5: 1.0},
                              primary_value=0.0)
    qr2 = _make_query_result("q2", hit_at_5=1.0, mrr=1.0,
                              recall_at_k={5: 1.0}, ndcg_at_k={5: 1.0},
                              primary_value=0.8)

    run = ev._build_run("test_run", dataset, [qr1, qr2], 10.0, 100)

    assert run.config_snapshot["gen_zero_count"] == 1
    assert run.config_snapshot["gen_nonzero_count"] == 1


def test_generation_disabled_no_gen_score():
    """Con generation_enabled=False, avg_generation_score es None."""
    ev = _make_evaluator(generation_enabled=False)
    dataset = _make_dataset()

    qr = _make_query_result("q1", hit_at_5=1.0, mrr=1.0,
                             recall_at_k={5: 1.0}, ndcg_at_k={5: 1.0})

    run = ev._build_run("test_run", dataset, [qr], 10.0, 100)

    assert run.avg_generation_score is None


# =================================================================
# Tests de contrato JSON: to_dict() / to_dict_full()
#
# Validan el schema de serializacion que report.py y consumidores
# downstream esperan. Distinto de los tests de agregacion arriba
# (que validan la matematica interna).
# =================================================================

def test_to_dict_schema_keys():
    """to_dict() contiene todas las keys esperadas del JSON de salida."""
    ev = _make_evaluator()
    dataset = _make_dataset()

    qr = _make_query_result("q1", hit_at_5=1.0, mrr=0.5,
                             recall_at_k={5: 0.8}, ndcg_at_k={5: 0.7})

    run = ev._build_run("test_run", dataset, [qr], 10.0, 100)
    d = run.to_dict()

    expected_keys = {
        "run_id", "dataset_name", "embedding_model", "retrieval_strategy",
        "config_snapshot", "num_queries_evaluated", "num_queries_failed",
        "total_documents", "avg_hit_rate_at_5", "avg_mrr",
        "avg_recall_at_k", "avg_ndcg_at_k",
        "retrieval_complement_recall_at_k",
        "avg_retrieved_count", "avg_expected_count",
        "avg_generation_recall", "avg_generation_hit",
        "reranker_rescue_count",
        "avg_generation_score", "execution_time_seconds",
        "timestamp", "status",
    }
    missing = expected_keys - set(d.keys())
    extra = set(d.keys()) - expected_keys
    assert not missing, f"Keys faltantes en to_dict(): {missing}"
    assert not extra, f"Keys inesperadas en to_dict(): {extra}"


def test_to_dict_rounding():
    """Valores float se redondean a 4 decimales en to_dict()."""
    ev = _make_evaluator()
    dataset = _make_dataset()

    qr = _make_query_result("q1", hit_at_5=1.0, mrr=0.333333333,
                             recall_at_k={5: 0.666666666},
                             ndcg_at_k={5: 0.777777777},
                             primary_value=0.123456789)

    run = ev._build_run("test_run", dataset, [qr], 10.123456, 100)
    d = run.to_dict()

    assert d["avg_mrr"] == 0.3333, f"MRR no redondeado: {d['avg_mrr']}"
    assert d["avg_recall_at_k"][5] == 0.6667, f"Recall no redondeado: {d['avg_recall_at_k'][5]}"
    assert d["avg_ndcg_at_k"][5] == 0.7778, f"NDCG no redondeado: {d['avg_ndcg_at_k'][5]}"
    assert d["execution_time_seconds"] == 10.12, f"Tiempo no redondeado: {d['execution_time_seconds']}"
    assert d["avg_generation_score"] == 0.1235, f"Gen score no redondeado: {d['avg_generation_score']}"


def test_to_dict_full_includes_query_results():
    """to_dict_full() agrega array query_results al dict base."""
    ev = _make_evaluator()
    dataset = _make_dataset()

    qr = _make_query_result("q1", hit_at_5=1.0, mrr=1.0,
                             recall_at_k={5: 1.0}, ndcg_at_k={5: 1.0})

    run = ev._build_run("test_run", dataset, [qr], 10.0, 100)
    d_full = run.to_dict_full()

    assert "query_results" in d_full, "to_dict_full() debe incluir query_results"
    assert len(d_full["query_results"]) == 1
    assert d_full["query_results"][0]["query_id"] == "q1"

    # to_dict() base NO incluye query_results
    d_base = run.to_dict()
    assert "query_results" not in d_base


def test_query_result_to_dict_schema():
    """QueryEvaluationResult.to_dict() contiene keys esperadas."""
    qr = _make_query_result("q1", hit_at_5=1.0, mrr=1.0,
                             recall_at_k={5: 1.0}, ndcg_at_k={5: 1.0})
    d = qr.to_dict()

    required_keys = {
        "query_id", "query_text", "dataset_name", "dataset_type",
        "retrieved_doc_ids", "hit_at_k", "recall_at_k", "ndcg_at_k", "mrr",
        "generated_response", "expected_response",
        "primary_metric_type", "primary_metric_value",
        "secondary_metrics", "status",
    }
    missing = required_keys - set(d.keys())
    assert not missing, f"Keys faltantes en QueryResult.to_dict(): {missing}"


def test_query_result_to_dict_conditional_fields():
    """Campos condicionales solo aparecen cuando tienen contenido."""
    qr = _make_query_result("q1", hit_at_5=1.0, mrr=1.0,
                             recall_at_k={5: 1.0}, ndcg_at_k={5: 1.0})
    d = qr.to_dict()

    # Sin reranker: no generation_doc_ids, no pre_rerank, no metadata
    assert "generation_doc_ids" not in d
    assert "pre_rerank_candidate_ids" not in d
    assert "metadata" not in d

    # Con reranker: generation_doc_ids y pre_rerank presentes
    qr.retrieval.generation_doc_ids = ["d0", "d1"]
    qr.retrieval.pre_rerank_candidate_ids = ["d0", "d1", "d2", "d3"]
    qr.metadata = {"reranked": True}
    d_reranked = qr.to_dict()

    assert d_reranked["generation_doc_ids"] == ["d0", "d1"]
    assert d_reranked["pre_rerank_candidate_ids"] == ["d0", "d1", "d2", "d3"]
    assert d_reranked["metadata"]["reranked"] is True


if __name__ == "__main__":
    test_avg_hit_rate_at_5()
    test_avg_mrr()
    test_avg_recall_at_k()
    test_complement_recall_is_one_minus_recall()
    test_generation_score_includes_zeros()
    test_failed_queries_not_counted_in_metrics()
    test_no_completed_queries_zeros()
    test_config_snapshot_fields()
    test_corpus_indexed_size()
    test_run_metadata()
    test_single_query_no_averaging_error()
    test_gen_zero_count_in_snapshot()
    test_generation_disabled_no_gen_score()
    test_to_dict_schema_keys()
    test_to_dict_rounding()
    test_to_dict_full_includes_query_results()
    test_query_result_to_dict_schema()
    test_query_result_to_dict_conditional_fields()
