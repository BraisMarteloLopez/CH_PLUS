"""
Test DT-6 #3: context_utilization sync y async reciben contexto >4000 chars
sin truncar. Caso borde: contexto vacio retorna 0.0 sin invocar judge.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.metrics import LLMJudgeMetrics, MetricType


class MockJudge:
    def __init__(self):
        self.captured_prompt = None
        self.call_count = 0

    def invoke(self, user_prompt, system_prompt=None):
        self.captured_prompt = user_prompt
        self.call_count += 1
        return '{"score": 0.6, "justification": "test"}'

    async def invoke_async(self, user_prompt, system_prompt=None):
        self.captured_prompt = user_prompt
        self.call_count += 1
        return '{"score": 0.6, "justification": "test async"}'


def test_context_utilization_sync_no_truncation():
    judge = MockJudge()
    context = "C" * 8000
    generated = "some answer"
    query = "some question"

    result = LLMJudgeMetrics.context_utilization(generated, context, query, judge)

    assert judge.captured_prompt is not None, "Judge nunca fue invocado"
    assert context in judge.captured_prompt, "Contexto truncado en context_utilization sync"
    assert result.metric_type == MetricType.CONTEXT_UTILIZATION
    print("PASS: context_utilization sync pasa contexto integro")


def test_context_utilization_async_no_truncation():
    judge = MockJudge()
    context = "D" * 8000
    generated = "some answer"
    query = "some question"

    result = asyncio.run(
        LLMJudgeMetrics.context_utilization_async(generated, context, query, judge)
    )

    assert judge.captured_prompt is not None, "Judge async nunca fue invocado"
    assert context in judge.captured_prompt, "Contexto truncado en context_utilization async"
    assert result.metric_type == MetricType.CONTEXT_UTILIZATION
    print("PASS: context_utilization async pasa contexto integro")


def test_empty_context_returns_zero_no_invoke():
    judge = MockJudge()

    # faithfulness sync con contexto vacio
    r1 = LLMJudgeMetrics.faithfulness("answer", "", judge)
    assert r1.value == 0.0, f"Esperado 0.0, obtenido {r1.value}"
    assert judge.call_count == 0, "Judge no deberia ser invocado con contexto vacio"

    # context_utilization sync con contexto vacio
    r2 = LLMJudgeMetrics.context_utilization("answer", "", "query", judge)
    assert r2.value == 0.0, f"Esperado 0.0, obtenido {r2.value}"
    assert judge.call_count == 0, "Judge no deberia ser invocado con contexto vacio"

    print("PASS: contexto vacio retorna 0.0 sin invocar judge")


if __name__ == "__main__":
    test_context_utilization_sync_no_truncation()
    test_context_utilization_async_no_truncation()
    test_empty_context_returns_zero_no_invoke()
