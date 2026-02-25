"""
Test de pipeline completo en modo mini (DEV_MODE).

Ejecuta MTEBEvaluator.run() con un subset minimo para verificar
que el flujo end-to-end funciona: carga -> indexacion -> retrieval ->
generacion -> metricas -> EvaluationRun.

Tiempo estimado: ~30-60 segundos con 10 queries.

Prerequisito: test_connectivity.py y test_components.py pasan.

Ejecucion:
    pytest tests/integration/test_pipeline.py -v -s
"""

import os
import tempfile

import pytest


@pytest.mark.integration
class TestPipelineDevMode:

    @pytest.fixture()
    def mini_config(self, mteb_config):
        """Config sobreescrita para pipeline mini: 10 queries, 100 corpus."""
        # Clonar config y sobreescribir para mini-run
        from sandbox_mteb.config import MTEBConfig, MinIOStorageConfig
        from dataclasses import replace

        # Usar tmpdir para resultados del test
        tmp_results = tempfile.mkdtemp(prefix="ragp_inttest_results_")
        tmp_vectordb = tempfile.mkdtemp(prefix="ragp_inttest_vectordb_")

        storage = replace(
            mteb_config.storage,
            evaluation_results_dir=__import__("pathlib").Path(tmp_results),
            vector_db_dir=__import__("pathlib").Path(tmp_vectordb),
        )

        mini = replace(
            mteb_config,
            storage=storage,
            dev_mode=True,
            dev_queries=10,
            dev_corpus_size=100,
            generation_enabled=True,
        )
        mini.ensure_directories()
        return mini

    def test_pipeline_completes(self, mini_config):
        """Pipeline DEV_MODE con 10 queries completa sin errores."""
        from sandbox_mteb.evaluator import MTEBEvaluator
        from shared.types import EvaluationStatus

        evaluator = MTEBEvaluator(mini_config)
        run = evaluator.run()

        # Run completado
        assert run.status == EvaluationStatus.COMPLETED

        # Evaluo al menos 1 query
        assert run.num_queries_evaluated > 0, "0 queries evaluadas"

        # Metricas de retrieval en rango valido
        assert 0.0 <= run.avg_hit_rate_at_5 <= 1.0
        assert 0.0 <= run.avg_mrr <= 1.0

        # Metricas de generacion presentes (generation_enabled=True)
        assert run.avg_generation_score is not None, (
            "avg_generation_score es None con generation_enabled=True"
        )
        assert 0.0 <= run.avg_generation_score <= 1.0

    def test_pipeline_query_results_structure(self, mini_config):
        """Cada QueryEvaluationResult tiene la estructura esperada."""
        from sandbox_mteb.evaluator import MTEBEvaluator
        from shared.types import EvaluationStatus

        evaluator = MTEBEvaluator(mini_config)
        run = evaluator.run()

        for qr in run.query_results:
            # Campos basicos presentes
            assert qr.query_id, "query_id vacio"
            assert qr.query_text, "query_text vacio"
            assert qr.dataset_name == mini_config.dataset_name

            if qr.status == EvaluationStatus.COMPLETED:
                # Retrieval
                assert len(qr.retrieval.retrieved_doc_ids) > 0, (
                    f"Query '{qr.query_id}' sin docs recuperados"
                )
                assert len(qr.retrieval.retrieval_scores) == len(
                    qr.retrieval.retrieved_doc_ids
                )

                # Generacion
                assert qr.generation is not None
                assert len(qr.generation.generated_response) > 0

                # Metrica primaria
                assert 0.0 <= qr.primary_metric_value <= 1.0

    def test_pipeline_export(self, mini_config):
        """Los 3 ficheros de salida se generan correctamente."""
        from sandbox_mteb.evaluator import MTEBEvaluator
        from shared.report import RunExporter

        evaluator = MTEBEvaluator(mini_config)
        run = evaluator.run()

        exporter = RunExporter(
            output_dir=mini_config.storage.evaluation_results_dir
        )
        paths = exporter.export(run)

        # Se generan los 3 tipos de fichero
        assert "json" in paths, "Falta fichero JSON"
        assert "summary_csv" in paths, "Falta CSV summary"
        assert "detail_csv" in paths, "Falta CSV detail"

        # Ficheros existen y no estan vacios
        for kind, path in paths.items():
            p = __import__("pathlib").Path(path)
            assert p.exists(), f"Fichero {kind} no existe: {path}"
            assert p.stat().st_size > 0, f"Fichero {kind} esta vacio: {path}"
