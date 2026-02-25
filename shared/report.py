"""
Modulo: Report Exporter
Descripcion: Exportador plano para resultados de evaluacion.

Ubicacion: shared/report.py

Reemplaza evaluation_report.py (~500 lineas de boilerplate
orientado a matrices de comparacion multi-modelo).

Un run = un JSON + un CSV resumen + un CSV detalle.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Optional

from shared.types import EvaluationRun

logger = logging.getLogger(__name__)


class RunExporter:
    """
    Exporta un EvaluationRun a JSON y CSV.

    Uso:
        exporter = RunExporter(output_dir=Path("./results"))
        exporter.export(run)
        # Genera:
        #   results/run_20250211_143022.json
        #   results/run_20250211_143022_summary.csv
        #   results/run_20250211_143022_detail.csv
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, run: EvaluationRun) -> dict:
        """
        Exporta el run a todos los formatos.

        Returns:
            Dict con paths de archivos generados:
            {"json": Path, "summary_csv": Path, "detail_csv": Path}
        """
        paths = {
            "json": self.to_json(run),
            "summary_csv": self.to_summary_csv(run),
            "detail_csv": self.to_detail_csv(run),
        }
        logger.info(
            f"Run {run.run_id} exportado a {self.output_dir}: "
            f"{', '.join(p.name for p in paths.values())}"
        )
        return paths

    def to_json(self, run: EvaluationRun, filename: Optional[str] = None) -> Path:
        """Exporta run completo (con query_results) a JSON."""
        fname = filename or f"{run.run_id}.json"
        path = self.output_dir / fname

        data = run.to_dict_full()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.debug(f"JSON exportado: {path}")
        return path

    def to_summary_csv(
        self, run: EvaluationRun, filename: Optional[str] = None
    ) -> Path:
        """Exporta metricas agregadas a CSV (una fila)."""
        fname = filename or f"{run.run_id}_summary.csv"
        path = self.output_dir / fname

        row = {
            "run_id": run.run_id,
            "dataset": run.dataset_name,
            "embedding_model": run.embedding_model,
            "strategy": run.retrieval_strategy,
            "queries_evaluated": run.num_queries_evaluated,
            "queries_failed": run.num_queries_failed,
            "total_documents": run.total_documents,
            "hit_rate_at_5": round(run.avg_hit_rate_at_5, 4),
            "mrr": round(run.avg_mrr, 4),
        }

        # Recall@K y NDCG@K
        for k, v in sorted(run.avg_recall_at_k.items()):
            row[f"recall_at_{k}"] = round(v, 4)
        for k, v in sorted(run.avg_ndcg_at_k.items()):
            row[f"ndcg_at_{k}"] = round(v, 4)

        row["avg_generation_score"] = (
            round(run.avg_generation_score, 4)
            if run.avg_generation_score is not None
            else ""
        )
        row["avg_retrieved_count"] = round(run.avg_retrieved_count, 1)
        row["avg_expected_count"] = round(run.avg_expected_count, 1)

        # Post-rerank retrieval metrics (solo con reranker)
        row["gen_recall"] = (
            round(run.avg_generation_recall, 4)
            if run.avg_generation_recall is not None
            else ""
        )
        row["gen_hit"] = (
            round(run.avg_generation_hit, 4)
            if run.avg_generation_hit is not None
            else ""
        )
        row["reranker_rescue_count"] = (
            run.reranker_rescue_count if run.avg_generation_recall is not None else ""
        )

        # Config snapshot fields
        snapshot = run.config_snapshot or {}
        row["retrieval_k"] = snapshot.get("retrieval_k", "")
        row["reranker_top_n"] = snapshot.get("reranker_top_n", "")
        row["corpus_shuffle_seed"] = snapshot.get("corpus_shuffle_seed", "")
        row["corpus_indexed"] = snapshot.get("corpus_indexed", "")
        row["corpus_total_available"] = snapshot.get("corpus_total_available", "")
        row["gen_zero_count"] = snapshot.get("gen_zero_count", "")
        row["gen_nonzero_count"] = snapshot.get("gen_nonzero_count", "")

        row["execution_time_s"] = round(run.execution_time_seconds, 2)
        row["timestamp"] = run.timestamp

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)

        logger.debug(f"Summary CSV exportado: {path}")
        return path

    def to_detail_csv(
        self, run: EvaluationRun, filename: Optional[str] = None
    ) -> Path:
        """Exporta detalle por query a CSV (una fila por query)."""
        fname = filename or f"{run.run_id}_detail.csv"
        path = self.output_dir / fname

        if not run.query_results:
            logger.warning("Sin query_results para exportar detalle")
            path.touch()
            return path

        # Determinar K values
        first = run.query_results[0]
        k_values = sorted(first.retrieval.hit_at_k.keys())

        # Collect all secondary metric keys across all queries
        all_secondary_keys: set[str] = set()
        for qr in run.query_results:
            all_secondary_keys.update(qr.secondary_metrics.keys())
        sorted_secondary_keys = sorted(all_secondary_keys)

        # Detect if reranker was active (any query has generation_doc_ids)
        has_reranker = any(
            qr.retrieval.generation_doc_ids for qr in run.query_results
        )

        fieldnames = [
            "query_id",
            "query_text",
            "status",
            "mrr",
        ]
        for k in k_values:
            fieldnames.append(f"hit_at_{k}")
        for k in k_values:
            fieldnames.append(f"recall_at_{k}")
        for k in k_values:
            fieldnames.append(f"ndcg_at_{k}")
        fieldnames.extend([
            "retrieval_time_ms",
            "n_retrieved",
            "n_generation_docs",
            "reranked",
        ])
        if has_reranker:
            fieldnames.extend(["gen_recall", "gen_hit"])
        fieldnames.extend([
            "n_expected",
            "primary_metric_type",
            "primary_metric_value",
        ])
        # Secondary metric columns
        for sk in sorted_secondary_keys:
            fieldnames.append(f"sec_{sk}")
        fieldnames.extend([
            "expected_response",
            "generated_response",
        ])

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for qr in run.query_results:
                row = {
                    "query_id": qr.query_id,
                    "query_text": qr.query_text[:200],
                    "status": qr.status.value,
                    "mrr": round(qr.retrieval.mrr, 4),
                }
                for k in k_values:
                    row[f"hit_at_{k}"] = round(
                        qr.retrieval.hit_at_k.get(k, 0.0), 4
                    )
                    row[f"recall_at_{k}"] = round(
                        qr.retrieval.recall_at_k.get(k, 0.0), 4
                    )
                    row[f"ndcg_at_{k}"] = round(
                        qr.retrieval.ndcg_at_k.get(k, 0.0), 4
                    )
                row["retrieval_time_ms"] = round(
                    qr.retrieval.retrieval_time_ms, 1
                )
                row["n_retrieved"] = len(qr.retrieval.retrieved_doc_ids)
                row["n_generation_docs"] = (
                    len(qr.retrieval.generation_doc_ids)
                    if qr.retrieval.generation_doc_ids
                    else len(qr.retrieval.retrieved_doc_ids)
                )
                # FIX DT-7: exponer estado de rerank per-query para diagnostico
                reranked_val = qr.metadata.get("reranked") if qr.metadata else None
                row["reranked"] = "" if reranked_val is None else reranked_val
                if has_reranker:
                    row["gen_recall"] = round(qr.retrieval.generation_recall, 4)
                    row["gen_hit"] = round(qr.retrieval.generation_hit, 4)
                row["n_expected"] = len(qr.retrieval.expected_doc_ids)
                row["primary_metric_type"] = qr.primary_metric_type.value
                row["primary_metric_value"] = round(
                    qr.primary_metric_value, 4
                )
                # Secondary metrics
                for sk in sorted_secondary_keys:
                    row[f"sec_{sk}"] = round(
                        qr.secondary_metrics.get(sk, 0.0), 4
                    )
                row["expected_response"] = (
                    (qr.expected_response or "")[:200]
                )
                row["generated_response"] = (
                    (qr.generation.generated_response if qr.generation else "")[:200]
                )
                writer.writerow(row)

        logger.debug(f"Detail CSV exportado: {path} ({len(run.query_results)} queries)")
        return path


__all__ = ["RunExporter"]
