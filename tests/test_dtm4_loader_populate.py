"""
Tests para MinIOLoader._populate_from_dataframes() en loader.py.

Cobertura DTm-4:
  - Validar que el metodo estatico popula correctamente un LoadedDataset
  - Queries con campos completos (query_id, text, answer, answer_type)
  - Corpus con doc_id, title, text
  - Qrels linking queries a docs
  - DataFrames vacios/None no causan fallo
  - answer_type inferido si ausente pero answer presente
  - question_type fallback a campo 'type'

Sin dependencias de MinIO ni S3. Usa mock DataFrames.
"""
from unittest.mock import MagicMock

from shared.types import LoadedDataset, DatasetType, MetricType
from sandbox_mteb.loader import MinIOLoader


# =================================================================
# Helper: Mock DataFrame
# =================================================================

class MockDataFrame:
    """DataFrame minimo compatible con iterrows()."""

    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def iterrows(self):
        for i, row in enumerate(self._rows):
            yield i, MockRow(row)


class MockRow(dict):
    """Row que soporta .get() como dict."""

    def get(self, key, default=""):
        return super().get(key, default)


# =================================================================
# Helper: resultado base
# =================================================================

def _make_empty_result():
    return LoadedDataset(
        name="test",
        dataset_type=DatasetType.HYBRID,
        primary_metric=MetricType.F1_SCORE,
    )


# =================================================================
# Tests
# =================================================================

def test_basic_populate():
    """Popula queries, corpus y qrels correctamente."""
    queries_df = MockDataFrame([
        {"query_id": "q1", "text": "What is AI?", "answer": "Artificial Intelligence",
         "answer_type": "text", "question_type": "bridge", "level": "easy"},
        {"query_id": "q2", "text": "Is Python good?", "answer": "yes",
         "answer_type": "label", "question_type": "comparison", "level": "medium"},
    ])
    corpus_df = MockDataFrame([
        {"doc_id": "d1", "title": "AI Intro", "text": "AI is the simulation of intelligence."},
        {"doc_id": "d2", "title": "Python", "text": "Python is a programming language."},
    ])
    qrels_df = MockDataFrame([
        {"query_id": "q1", "doc_id": "d1"},
        {"query_id": "q2", "doc_id": "d2"},
    ])

    result = _make_empty_result()
    MinIOLoader._populate_from_dataframes(result, queries_df, corpus_df, qrels_df)

    # Queries
    assert len(result.queries) == 2
    assert result.queries[0].query_id == "q1"
    assert result.queries[0].query_text == "What is AI?"
    assert result.queries[0].expected_answer == "Artificial Intelligence"
    assert result.queries[0].answer_type == "text"
    assert result.queries[1].answer_type == "label"

    # Corpus
    assert len(result.corpus) == 2
    assert "d1" in result.corpus
    assert result.corpus["d1"].title == "AI Intro"
    assert "simulation" in result.corpus["d1"].content

    # Qrels
    assert result.queries[0].relevant_doc_ids == ["d1"]
    assert result.queries[1].relevant_doc_ids == ["d2"]

    # Totals
    assert result.total_queries == 2
    assert result.total_corpus == 2


def test_multiple_qrels_per_query():
    """Una query con multiples docs relevantes."""
    queries_df = MockDataFrame([
        {"query_id": "q1", "text": "Multi-hop question"},
    ])
    corpus_df = MockDataFrame([
        {"doc_id": "d1", "title": "Doc A", "text": "Content A"},
        {"doc_id": "d2", "title": "Doc B", "text": "Content B"},
    ])
    qrels_df = MockDataFrame([
        {"query_id": "q1", "doc_id": "d1"},
        {"query_id": "q1", "doc_id": "d2"},
    ])

    result = _make_empty_result()
    MinIOLoader._populate_from_dataframes(result, queries_df, corpus_df, qrels_df)

    assert len(result.queries[0].relevant_doc_ids) == 2
    assert "d1" in result.queries[0].relevant_doc_ids
    assert "d2" in result.queries[0].relevant_doc_ids


def test_none_dataframes_no_crash():
    """DataFrames None no causan error."""
    result = _make_empty_result()
    MinIOLoader._populate_from_dataframes(result, None, None, None)

    assert len(result.queries) == 0
    assert len(result.corpus) == 0


def test_empty_dataframes():
    """DataFrames vacios producen listas vacias."""
    result = _make_empty_result()
    MinIOLoader._populate_from_dataframes(
        result, MockDataFrame([]), MockDataFrame([]), MockDataFrame([])
    )

    assert len(result.queries) == 0
    assert len(result.corpus) == 0
    assert result.total_queries == 0
    assert result.total_corpus == 0


def test_answer_type_inferred_when_missing():
    """Si answer_type ausente pero answer presente, se infiere 'text'."""
    queries_df = MockDataFrame([
        {"query_id": "q1", "text": "Question?", "answer": "Some answer"},
    ])

    result = _make_empty_result()
    MinIOLoader._populate_from_dataframes(
        result, queries_df, MockDataFrame([]), None
    )

    assert result.queries[0].answer_type == "text", (
        f"answer_type deberia ser 'text' inferido, obtenido: "
        f"'{result.queries[0].answer_type}'"
    )


def test_no_answer_no_answer_type():
    """Sin answer ni answer_type, expected_answer es None."""
    queries_df = MockDataFrame([
        {"query_id": "q1", "text": "Question?"},
    ])

    result = _make_empty_result()
    MinIOLoader._populate_from_dataframes(
        result, queries_df, MockDataFrame([]), None
    )

    assert result.queries[0].expected_answer is None
    assert result.queries[0].answer_type is None


def test_question_type_metadata():
    """question_type se guarda en metadata."""
    queries_df = MockDataFrame([
        {"query_id": "q1", "text": "Q?", "question_type": "bridge", "level": "hard"},
    ])

    result = _make_empty_result()
    MinIOLoader._populate_from_dataframes(
        result, queries_df, MockDataFrame([]), None
    )

    assert result.queries[0].metadata["question_type"] == "bridge"
    assert result.queries[0].metadata["level"] == "hard"


def test_question_type_fallback_to_type_field():
    """Si question_type no existe, usa campo 'type'."""
    queries_df = MockDataFrame([
        {"query_id": "q1", "text": "Q?", "type": "comparison"},
    ])

    result = _make_empty_result()
    MinIOLoader._populate_from_dataframes(
        result, queries_df, MockDataFrame([]), None
    )

    assert result.queries[0].metadata["question_type"] == "comparison"


def test_query_without_qrels_has_empty_relevant_ids():
    """Query sin qrels correspondiente tiene relevant_doc_ids vacio."""
    queries_df = MockDataFrame([
        {"query_id": "q1", "text": "Orphan query"},
    ])

    result = _make_empty_result()
    MinIOLoader._populate_from_dataframes(
        result, queries_df, MockDataFrame([]), MockDataFrame([])
    )

    assert result.queries[0].relevant_doc_ids == []


if __name__ == "__main__":
    test_basic_populate()
    test_multiple_qrels_per_query()
    test_none_dataframes_no_crash()
    test_empty_dataframes()
    test_answer_type_inferred_when_missing()
    test_no_answer_no_answer_type()
    test_question_type_metadata()
    test_question_type_fallback_to_type_field()
    test_query_without_qrels_has_empty_relevant_ids()
