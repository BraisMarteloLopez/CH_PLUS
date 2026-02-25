"""
Test DT-6 #1: faithfulness() sync recibe contexto >4000 chars
y lo pasa integro al LLM judge (sin truncar).
"""
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.metrics import LLMJudgeMetrics, MetricType


class MockJudge:
    """Mock que captura el user_prompt recibido."""

    def __init__(self):
        self.captured_prompt = None

    def invoke(self, user_prompt, system_prompt=None):
        self.captured_prompt = user_prompt
        return '{"score": 0.8, "justification": "test"}'


def test_faithfulness_sync_no_truncation():
    judge = MockJudge()

    # Contexto de 8000 chars (el doble del antiguo limite de 4000)
    context = "A" * 8000
    generated = "some answer"

    result = LLMJudgeMetrics.faithfulness(generated, context, judge)

    # El prompt capturado debe contener el contexto integro
    assert judge.captured_prompt is not None, "Judge nunca fue invocado"
    assert context in judge.captured_prompt, (
        f"Contexto truncado: esperado 8000 chars en prompt, "
        f"encontrado {len(judge.captured_prompt)} chars totales"
    )
    assert result.metric_type == MetricType.FAITHFULNESS
    assert result.value == 0.8

    print("PASS: faithfulness sync pasa contexto integro (8000 chars, sin truncar)")


if __name__ == "__main__":
    test_faithfulness_sync_no_truncation()
