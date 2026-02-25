"""
Test DT-6 #2: faithfulness_async() recibe contexto >4000 chars
y lo pasa integro al LLM judge (sin truncar).
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.metrics import LLMJudgeMetrics, MetricType


class MockJudgeAsync:
    """Mock async que captura el user_prompt recibido."""

    def __init__(self):
        self.captured_prompt = None

    async def invoke_async(self, user_prompt, system_prompt=None):
        self.captured_prompt = user_prompt
        return '{"score": 0.75, "justification": "test async"}'


def test_faithfulness_async_no_truncation():
    judge = MockJudgeAsync()

    context = "B" * 8000
    generated = "some answer"

    result = asyncio.run(
        LLMJudgeMetrics.faithfulness_async(generated, context, judge)
    )

    assert judge.captured_prompt is not None, "Judge async nunca fue invocado"
    assert context in judge.captured_prompt, (
        f"Contexto truncado en async: esperado 8000 chars en prompt, "
        f"encontrado {len(judge.captured_prompt)} chars totales"
    )
    assert result.metric_type == MetricType.FAITHFULNESS
    assert result.value == 0.75

    print("PASS: faithfulness_async pasa contexto integro (8000 chars, sin truncar)")


if __name__ == "__main__":
    test_faithfulness_async_no_truncation()
