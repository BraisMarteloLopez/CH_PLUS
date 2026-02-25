"""
Test DTm-5 #12: Metrica secundaria que falla produce MetricResult con error
              (no desaparece silenciosamente).
Test DTm-5 #13: El fallo emite logger.warning con tipo de metrica y query.
"""
import asyncio
import logging
from unittest.mock import MagicMock

from shared.types import DatasetType, MetricType, get_dataset_config
from shared.metrics import MetricsCalculator, MetricResult


# ---------------------------------------------------------------
# Mock de MetricsCalculator que falla en una metrica especifica
# ---------------------------------------------------------------

class FailingMetricsCalculator(MetricsCalculator):
    """Calculator que falla en FAITHFULNESS pero funciona en el resto."""

    def __init__(self):
        super().__init__(llm_judge=MagicMock(), embedding_model=None)

    async def calculate_async(self, metric_type, generated, expected=None,
                              context=None, query=None):
        if metric_type == MetricType.FAITHFULNESS:
            raise RuntimeError("LLM judge timeout simulado")
        # F1 y EM funcionan normalmente
        if metric_type == MetricType.F1_SCORE:
            return MetricResult(metric_type=MetricType.F1_SCORE, value=0.65)
        if metric_type == MetricType.EXACT_MATCH:
            return MetricResult(metric_type=MetricType.EXACT_MATCH, value=0.0)
        if metric_type == MetricType.ACCURACY:
            return MetricResult(metric_type=MetricType.ACCURACY, value=1.0)
        raise ValueError(f"Metrica no esperada en mock: {metric_type}")


# ---------------------------------------------------------------
# Helper: ejecutar _calculate_metrics_async aislado
# ---------------------------------------------------------------

def _make_evaluator_with_failing_calc():
    from sandbox_mteb.evaluator import MTEBEvaluator
    from sandbox_mteb.config import MTEBConfig, MinIOStorageConfig
    from shared.config_base import InfraConfig, RerankerConfig
    from shared.retrieval.core import RetrievalConfig

    config = MTEBConfig(
        infra=InfraConfig(),
        storage=MinIOStorageConfig(),
        retrieval=RetrievalConfig(),
        reranker=RerankerConfig(enabled=False),
        generation_enabled=True,
    )
    evaluator = MTEBEvaluator(config)
    evaluator._metrics_calculator = FailingMetricsCalculator()
    return evaluator


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------

def test_failed_secondary_metric_produces_result():
    """
    #12: Cuando faithfulness falla como metrica secundaria,
    el resultado debe contener la key con value=0.0 (no desaparecer).
    """
    evaluator = _make_evaluator_with_failing_calc()

    primary, secondary = asyncio.run(
        evaluator._calculate_metrics_async(
            generated="yes",
            expected_answer="yes",
            answer_type="label",          # primary = ACCURACY
            context="some context here",
            query_text="Is this a test?",
            dataset_type=DatasetType.HYBRID,
            dataset_name="hotpotqa",
        )
    )

    # Primary debe ser ACCURACY (answer_type=label)
    assert primary.metric_type == MetricType.ACCURACY, (
        f"Primary esperado ACCURACY, obtenido {primary.metric_type}"
    )

    # Secundarias: F1, EM, FAITHFULNESS deberian estar presentes
    secondary_types = {r.metric_type for r in secondary}
    assert MetricType.FAITHFULNESS in secondary_types, (
        f"FAITHFULNESS ausente en secundarias. Presentes: "
        f"{[r.metric_type.value for r in secondary]}"
    )

    # El MetricResult de faithfulness debe tener error y value=0.0
    faith_result = next(
        r for r in secondary if r.metric_type == MetricType.FAITHFULNESS
    )
    assert faith_result.value == 0.0, (
        f"Esperado value=0.0 para metrica fallida, obtenido {faith_result.value}"
    )
    assert faith_result.error is not None, (
        "Esperado error no-None para metrica fallida"
    )
    assert "secondary_metric_error" in faith_result.error, (
        f"Error no contiene prefijo esperado: '{faith_result.error}'"
    )

    # F1 y EM deben estar presentes sin error
    f1_result = next(
        r for r in secondary if r.metric_type == MetricType.F1_SCORE
    )
    assert f1_result.error is None, f"F1 no deberia tener error: {f1_result.error}"
    assert f1_result.value == 0.65

    em_result = next(
        r for r in secondary if r.metric_type == MetricType.EXACT_MATCH
    )
    assert em_result.error is None

    print(
        "PASS: metrica secundaria fallida produce MetricResult con "
        "error y value=0.0 (no desaparece)"
    )


def test_failed_secondary_metric_logs_warning():
    """
    #13: El fallo de metrica secundaria emite logger.warning
    con el tipo de metrica y fragmento de la query.
    """
    evaluator = _make_evaluator_with_failing_calc()

    # Capturar logs
    log_records = []
    handler = logging.Handler()
    handler.emit = lambda record: log_records.append(record)
    eval_logger = logging.getLogger("sandbox_mteb.evaluator")
    eval_logger.addHandler(handler)
    eval_logger.setLevel(logging.WARNING)

    try:
        asyncio.run(
            evaluator._calculate_metrics_async(
                generated="yes",
                expected_answer="yes",
                answer_type="label",
                context="some context",
                query_text="Is this a warning test query?",
                dataset_type=DatasetType.HYBRID,
                dataset_name="hotpotqa",
            )
        )

        # Buscar warning sobre faithfulness
        faith_warnings = [
            r for r in log_records
            if r.levelno == logging.WARNING and "faithfulness" in r.getMessage().lower()
        ]
        assert len(faith_warnings) >= 1, (
            f"Esperado al menos 1 warning sobre faithfulness, "
            f"encontrados {len(faith_warnings)}. "
            f"Todos los warnings: {[r.getMessage() for r in log_records]}"
        )

        msg = faith_warnings[0].getMessage()
        assert "LLM judge timeout" in msg, (
            f"Warning no contiene mensaje de error original: '{msg}'"
        )
        assert "Is this a warning" in msg, (
            f"Warning no contiene fragmento de query: '{msg}'"
        )

        print("PASS: fallo de metrica secundaria emite warning con tipo y query")

    finally:
        eval_logger.removeHandler(handler)


def test_all_secondary_fail_returns_all_with_errors():
    """
    Caso borde: todas las metricas secundarias fallan.
    Todas deben estar presentes con error, ninguna desaparece.
    """

    class AllFailCalculator(MetricsCalculator):
        def __init__(self):
            super().__init__(llm_judge=MagicMock(), embedding_model=None)

        async def calculate_async(self, metric_type, generated, expected=None,
                                  context=None, query=None):
            if metric_type == MetricType.ACCURACY:
                # Primary funciona
                return MetricResult(metric_type=MetricType.ACCURACY, value=1.0)
            raise RuntimeError(f"Fallo simulado en {metric_type.value}")

    from sandbox_mteb.evaluator import MTEBEvaluator
    from sandbox_mteb.config import MTEBConfig, MinIOStorageConfig
    from shared.config_base import InfraConfig, RerankerConfig
    from shared.retrieval.core import RetrievalConfig

    config = MTEBConfig(
        infra=InfraConfig(),
        storage=MinIOStorageConfig(),
        retrieval=RetrievalConfig(),
        reranker=RerankerConfig(enabled=False),
        generation_enabled=True,
    )
    evaluator = MTEBEvaluator(config)
    evaluator._metrics_calculator = AllFailCalculator()

    primary, secondary = asyncio.run(
        evaluator._calculate_metrics_async(
            generated="yes",
            expected_answer="yes",
            answer_type="label",
            context="context",
            query_text="query",
            dataset_type=DatasetType.HYBRID,
            dataset_name="hotpotqa",
        )
    )

    # hotpotqa con answer_type=label: secondary_types = [F1, EM, FAITHFULNESS]
    assert len(secondary) == 3, (
        f"Esperado 3 secundarias (todas fallidas), obtenido {len(secondary)}"
    )
    for r in secondary:
        assert r.value == 0.0, f"{r.metric_type.value} deberia tener value=0.0"
        assert r.error is not None, f"{r.metric_type.value} deberia tener error"

    print("PASS: todas las secundarias fallan -> todas presentes con error")


if __name__ == "__main__":
    test_failed_secondary_metric_produces_result()
    test_failed_secondary_metric_logs_warning()
    test_all_secondary_fail_returns_all_with_errors()
