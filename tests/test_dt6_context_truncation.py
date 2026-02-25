"""
Tests DT-6: contexto >4000 chars pasa integro al LLM judge (sin truncar).

Cubre faithfulness (sync/async) y context_utilization (sync/async) en un
solo archivo parametrizado. Adicionalmente: contexto vacio retorna 0.0
sin invocar judge.
"""
import asyncio

import pytest

from shared.metrics import LLMJudgeMetrics, MetricType


class MockJudge:
    """Mock sync+async que captura el prompt recibido."""

    def __init__(self):
        self.captured_prompt = None
        self.call_count = 0

    def invoke(self, user_prompt, system_prompt=None):
        self.captured_prompt = user_prompt
        self.call_count += 1
        return '{"score": 0.8, "justification": "test"}'

    async def invoke_async(self, user_prompt, system_prompt=None):
        self.captured_prompt = user_prompt
        self.call_count += 1
        return '{"score": 0.8, "justification": "test"}'


@pytest.mark.parametrize("method,is_async", [
    ("faithfulness", False),
    ("faithfulness_async", True),
    ("context_utilization", False),
    ("context_utilization_async", True),
])
def test_context_passed_without_truncation(method, is_async):
    """Contexto de 8000 chars llega integro al judge (no se trunca a 4000)."""
    judge = MockJudge()
    context = "A" * 8000

    fn = getattr(LLMJudgeMetrics, method)
    if "context_utilization" in method:
        args = ("some answer", context, "some question", judge)
    else:
        args = ("some answer", context, judge)

    if is_async:
        result = asyncio.run(fn(*args))
    else:
        result = fn(*args)

    assert judge.captured_prompt is not None, f"{method}: judge nunca invocado"
    assert context in judge.captured_prompt, f"{method}: contexto truncado"


def test_empty_context_returns_zero_without_invoking_judge():
    """Contexto vacio retorna 0.0 sin invocar judge."""
    judge = MockJudge()

    r1 = LLMJudgeMetrics.faithfulness("answer", "", judge)
    assert r1.value == 0.0
    r2 = LLMJudgeMetrics.context_utilization("answer", "", "query", judge)
    assert r2.value == 0.0
    assert judge.call_count == 0
