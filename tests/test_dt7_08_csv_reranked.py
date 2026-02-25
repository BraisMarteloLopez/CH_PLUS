"""
Test DT-7 #8: metadata["reranked"] se propaga al detail CSV.
Columna 'reranked' con valor True, False, o vacio segun el caso.
"""
import csv
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from shared.types import (
    EvaluationRun, EvaluationStatus, QueryEvaluationResult,
    QueryRetrievalDetail, GenerationResult, DatasetType, MetricType,
)
from shared.report import RunExporter


def _make_query_result(query_id, reranked_value):
    """Crea un QueryEvaluationResult con metadata["reranked"] = reranked_value."""
    metadata = {}
    if reranked_value is not None:
        metadata["reranked"] = reranked_value

    return QueryEvaluationResult(
        query_id=query_id,
        query_text=f"query {query_id}",
        dataset_name="test",
        dataset_type=DatasetType.HYBRID,
        retrieval=QueryRetrievalDetail(
            retrieved_doc_ids=["doc_0"],
            retrieved_contents=["content"],
            retrieval_scores=[0.9],
            expected_doc_ids=["doc_0"],
        ),
        generation=GenerationResult(generated_response="answer"),
        expected_response="expected",
        primary_metric_type=MetricType.F1_SCORE,
        primary_metric_value=0.8,
        status=EvaluationStatus.COMPLETED,
        metadata=metadata,
    )


def test_reranked_column_in_csv():
    run = EvaluationRun(
        run_id="test_csv",
        dataset_name="test",
        embedding_model="mock",
        retrieval_strategy="SIMPLE_VECTOR",
        query_results=[
            _make_query_result("q1", True),    # reranked exitoso
            _make_query_result("q2", False),   # reranked fallido
            _make_query_result("q3", None),    # sin reranker
        ],
        num_queries_evaluated=3,
        status=EvaluationStatus.COMPLETED,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = RunExporter(output_dir=Path(tmpdir))
        csv_path = exporter.to_detail_csv(run)

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    assert len(rows) == 3, f"Esperado 3 filas, obtenido {len(rows)}"
    assert "reranked" in rows[0], "Columna 'reranked' ausente en CSV"

    assert rows[0]["reranked"] == "True", (
        f"q1: esperado 'True', obtenido '{rows[0]['reranked']}'"
    )
    assert rows[1]["reranked"] == "False", (
        f"q2: esperado 'False', obtenido '{rows[1]['reranked']}'"
    )
    assert rows[2]["reranked"] == "", (
        f"q3: esperado vacio, obtenido '{rows[2]['reranked']}'"
    )

    print("PASS: columna 'reranked' en detail CSV con valores True/False/vacio")


if __name__ == "__main__":
    test_reranked_column_in_csv()
