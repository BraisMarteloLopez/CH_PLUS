"""
Tests para metricas de retrieval efectivo (post-rerank).

Cubren:
  - QueryRetrievalDetail: generation_recall, generation_hit
  - EvaluationRun: avg_generation_recall, avg_generation_hit, reranker_rescue_count
  - QueryEvaluationResult.to_dict(): serializacion condicional
  - RunExporter: columnas en summary y detail CSV
"""

import csv
import tempfile
from pathlib import Path

import pytest

from shared.types import (
    DatasetType,
    EvaluationRun,
    EvaluationStatus,
    GenerationResult,
    MetricType,
    QueryEvaluationResult,
    QueryRetrievalDetail,
)
from shared.report import RunExporter


# =========================================================================
# QueryRetrievalDetail: calculo de generation_recall y generation_hit
# =========================================================================


class TestGenerationMetricsCalculation:
    """Calculo de metricas post-rerank en QueryRetrievalDetail."""

    def test_both_gold_in_generation(self):
        """Reranker promueve ambos gold docs -> recall=1.0, hit=1.0."""
        qrd = QueryRetrievalDetail(
            retrieved_doc_ids=["a", "b", "c"],
            retrieved_contents=["ca", "cb", "cc"],
            retrieval_scores=[0.9, 0.8, 0.7],
            expected_doc_ids=["x", "y"],
            generation_doc_ids=["x", "a", "y"],
            generation_contents=["cx", "ca", "cy"],
        )
        assert qrd.generation_recall == 1.0
        assert qrd.generation_hit == 1.0

    def test_one_gold_in_generation(self):
        """Reranker promueve 1 de 2 gold docs -> recall=0.5, hit=1.0."""
        qrd = QueryRetrievalDetail(
            retrieved_doc_ids=["a", "b", "c"],
            retrieved_contents=["ca", "cb", "cc"],
            retrieval_scores=[0.9, 0.8, 0.7],
            expected_doc_ids=["x", "y"],
            generation_doc_ids=["x", "a", "b"],
            generation_contents=["cx", "ca", "cb"],
        )
        assert qrd.generation_recall == 0.5
        assert qrd.generation_hit == 1.0

    def test_no_gold_in_generation(self):
        """Ni retrieval ni reranker encuentran gold -> recall=0, hit=0."""
        qrd = QueryRetrievalDetail(
            retrieved_doc_ids=["a", "b"],
            retrieved_contents=["ca", "cb"],
            retrieval_scores=[0.9, 0.8],
            expected_doc_ids=["x", "y"],
            generation_doc_ids=["a", "b", "c"],
            generation_contents=["ca", "cb", "cc"],
        )
        assert qrd.generation_recall == 0.0
        assert qrd.generation_hit == 0.0

    def test_no_reranker_leaves_defaults(self):
        """Sin reranker (generation_doc_ids vacio) -> metricas quedan en 0.0."""
        qrd = QueryRetrievalDetail(
            retrieved_doc_ids=["x", "b"],
            retrieved_contents=["cx", "cb"],
            retrieval_scores=[0.9, 0.8],
            expected_doc_ids=["x", "y"],
        )
        assert qrd.generation_doc_ids == []
        assert qrd.generation_recall == 0.0
        assert qrd.generation_hit == 0.0

    def test_rescue_scenario(self):
        """Gold docs fuera de top-K retrieval pero en generation (rescate)."""
        qrd = QueryRetrievalDetail(
            retrieved_doc_ids=["a", "b", "c"],
            retrieved_contents=["ca", "cb", "cc"],
            retrieval_scores=[0.9, 0.8, 0.7],
            expected_doc_ids=["x", "y"],
            generation_doc_ids=["x", "y", "a"],
            generation_contents=["cx", "cy", "ca"],
        )
        # retrieval: 0 gold in top-3
        assert qrd.recall_at_k[3] == 0.0
        # generation: both gold
        assert qrd.generation_recall == 1.0

    def test_single_expected_doc(self):
        """Una sola expected doc -> recall es 0.0 o 1.0."""
        qrd = QueryRetrievalDetail(
            retrieved_doc_ids=["a"],
            retrieved_contents=["ca"],
            retrieval_scores=[0.9],
            expected_doc_ids=["x"],
            generation_doc_ids=["x"],
            generation_contents=["cx"],
        )
        assert qrd.generation_recall == 1.0
        assert qrd.generation_hit == 1.0


# =========================================================================
# QueryEvaluationResult.to_dict(): serializacion condicional
# =========================================================================


class TestToDictSerialization:
    """Serializacion de generation_recall/hit en to_dict()."""

    def _make_qer(self, generation_doc_ids=None, expected_doc_ids=None):
        gen_ids = generation_doc_ids or []
        exp_ids = expected_doc_ids or ["x", "y"]
        return QueryEvaluationResult(
            query_id="q_test",
            query_text="test query",
            dataset_name="test",
            dataset_type=DatasetType.HYBRID,
            retrieval=QueryRetrievalDetail(
                retrieved_doc_ids=["a", "b"],
                retrieved_contents=["ca", "cb"],
                retrieval_scores=[0.9, 0.8],
                expected_doc_ids=exp_ids,
                generation_doc_ids=gen_ids,
                generation_contents=["c" + g for g in gen_ids],
            ),
            generation=GenerationResult("answer", 10.0),
            expected_response="expected",
            primary_metric_type=MetricType.F1_SCORE,
            primary_metric_value=0.8,
            status=EvaluationStatus.COMPLETED,
        )

    def test_to_dict_includes_gen_metrics_when_reranked(self):
        """Con reranker, to_dict incluye generation_recall y generation_hit."""
        qer = self._make_qer(generation_doc_ids=["x", "a"])
        d = qer.to_dict()
        assert "generation_recall" in d
        assert "generation_hit" in d
        assert d["generation_recall"] == 0.5  # 1/2 gold
        assert d["generation_hit"] == 1.0

    def test_to_dict_excludes_gen_metrics_without_reranker(self):
        """Sin reranker, to_dict NO incluye generation_recall ni generation_hit."""
        qer = self._make_qer(generation_doc_ids=[])
        d = qer.to_dict()
        assert "generation_recall" not in d
        assert "generation_hit" not in d


# =========================================================================
# EvaluationRun: campos agregados
# =========================================================================


class TestEvaluationRunAggregates:
    """Campos agregados post-rerank en EvaluationRun."""

    def test_defaults_none_without_reranker(self):
        """Sin datos de reranker, avg_generation_recall/hit son None."""
        run = EvaluationRun()
        assert run.avg_generation_recall is None
        assert run.avg_generation_hit is None
        assert run.reranker_rescue_count == 0

    def test_to_dict_includes_new_fields(self):
        """to_dict() siempre incluye los 3 nuevos campos."""
        run = EvaluationRun()
        d = run.to_dict()
        assert "avg_generation_recall" in d
        assert "avg_generation_hit" in d
        assert "reranker_rescue_count" in d

    def test_to_dict_rounds_values(self):
        """Valores se redondean a 4 decimales."""
        run = EvaluationRun(
            avg_generation_recall=0.123456789,
            avg_generation_hit=0.987654321,
            reranker_rescue_count=3,
        )
        d = run.to_dict()
        assert d["avg_generation_recall"] == 0.1235
        assert d["avg_generation_hit"] == 0.9877
        assert d["reranker_rescue_count"] == 3


# =========================================================================
# RunExporter: columnas CSV
# =========================================================================


class TestRunExporterCSV:
    """Columnas de generation retrieval en summary y detail CSV."""

    def _make_run(self, with_reranker=True):
        """Construye un EvaluationRun minimo con 2 queries."""
        qrs = []
        for i, (gen_ids, exp_ids) in enumerate([
            (["x", "a"], ["x", "y"]),  # partial rescue
            (["x", "y"], ["x", "y"]),  # full rescue
        ]):
            ret_ids = ["a", "b", "c"] if with_reranker else gen_ids
            qrd = QueryRetrievalDetail(
                retrieved_doc_ids=ret_ids,
                retrieved_contents=["c"] * len(ret_ids),
                retrieval_scores=[0.9, 0.8, 0.7][:len(ret_ids)],
                expected_doc_ids=exp_ids,
                generation_doc_ids=gen_ids if with_reranker else [],
                generation_contents=["c"] * len(gen_ids) if with_reranker else [],
            )
            qrs.append(QueryEvaluationResult(
                query_id=f"q_{i}",
                query_text=f"query {i}",
                dataset_name="test",
                dataset_type=DatasetType.HYBRID,
                retrieval=qrd,
                generation=GenerationResult("ans", 10.0),
                expected_response="exp",
                primary_metric_type=MetricType.F1_SCORE,
                primary_metric_value=0.5,
                status=EvaluationStatus.COMPLETED,
                metadata={"reranked": with_reranker},
            ))
        return EvaluationRun(
            run_id="test_run",
            dataset_name="test",
            embedding_model="test",
            retrieval_strategy="SIMPLE_VECTOR",
            num_queries_evaluated=2,
            avg_generation_recall=0.75 if with_reranker else None,
            avg_generation_hit=1.0 if with_reranker else None,
            reranker_rescue_count=2 if with_reranker else 0,
            query_results=qrs,
        )

    def test_summary_csv_with_reranker(self):
        """Summary CSV incluye gen_recall, gen_hit, reranker_rescue_count."""
        run = self._make_run(with_reranker=True)
        with tempfile.TemporaryDirectory() as tmp:
            exporter = RunExporter(output_dir=Path(tmp))
            path = exporter.to_summary_csv(run)
            with open(path) as f:
                reader = csv.DictReader(f)
                row = next(reader)
            assert row["gen_recall"] == "0.75"
            assert row["gen_hit"] == "1.0"
            assert row["reranker_rescue_count"] == "2"

    def test_summary_csv_without_reranker(self):
        """Summary CSV emite campos vacios sin reranker."""
        run = self._make_run(with_reranker=False)
        with tempfile.TemporaryDirectory() as tmp:
            exporter = RunExporter(output_dir=Path(tmp))
            path = exporter.to_summary_csv(run)
            with open(path) as f:
                reader = csv.DictReader(f)
                row = next(reader)
            assert row["gen_recall"] == ""
            assert row["gen_hit"] == ""
            assert row["reranker_rescue_count"] == ""

    def test_detail_csv_with_reranker(self):
        """Detail CSV incluye columnas gen_recall, gen_hit con reranker."""
        run = self._make_run(with_reranker=True)
        with tempfile.TemporaryDirectory() as tmp:
            exporter = RunExporter(output_dir=Path(tmp))
            path = exporter.to_detail_csv(run)
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert "gen_recall" in rows[0]
            assert "gen_hit" in rows[0]
            assert rows[0]["gen_recall"] == "0.5"   # 1/2 gold
            assert rows[1]["gen_recall"] == "1.0"    # 2/2 gold

    def test_detail_csv_without_reranker(self):
        """Detail CSV omite columnas gen_recall, gen_hit sin reranker."""
        run = self._make_run(with_reranker=False)
        with tempfile.TemporaryDirectory() as tmp:
            exporter = RunExporter(output_dir=Path(tmp))
            path = exporter.to_detail_csv(run)
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert "gen_recall" not in rows[0]
            assert "gen_hit" not in rows[0]
