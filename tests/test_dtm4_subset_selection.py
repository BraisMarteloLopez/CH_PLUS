"""
Tests para _select_subset_dev() en evaluator.py.

Cobertura DTm-4:
  - Gold docs presentes en corpus resultante
  - Distractores rellenan hasta dev_corpus_size
  - Seed determinista (misma seed = mismo resultado)
  - gold_ids > dev_corpus_size lanza ValueError
  - dev_queries >= total queries usa todas
  - Gold docs ausentes en corpus se reportan (no fallan)

Sin dependencias externas (usa mocks).
"""
import pytest
from unittest.mock import MagicMock

from shared.types import (
    LoadedDataset,
    NormalizedQuery,
    NormalizedDocument,
    DatasetType,
    MetricType,
)
from sandbox_mteb.evaluator import MTEBEvaluator
from sandbox_mteb.config import MTEBConfig, MinIOStorageConfig
from shared.config_base import InfraConfig, RerankerConfig
from shared.retrieval.core import RetrievalConfig


# =================================================================
# Helpers
# =================================================================

def _make_evaluator(
    dev_mode: bool = True,
    dev_queries: int = 5,
    dev_corpus_size: int = 20,
    seed: int = 42,
) -> MTEBEvaluator:
    config = MTEBConfig(
        infra=InfraConfig(),
        storage=MinIOStorageConfig(),
        retrieval=RetrievalConfig(),
        reranker=RerankerConfig(enabled=False),
        generation_enabled=False,
        dev_mode=dev_mode,
        dev_queries=dev_queries,
        dev_corpus_size=dev_corpus_size,
        corpus_shuffle_seed=seed,
    )
    return MTEBEvaluator(config)


def _make_dataset(
    n_queries: int = 20,
    n_corpus: int = 100,
    gold_per_query: int = 2,
) -> LoadedDataset:
    """Crea un dataset sintetico con queries, corpus y qrels."""
    queries = []
    corpus = {}

    # Crear corpus: docs d_0 .. d_{n_corpus-1}
    for i in range(n_corpus):
        doc_id = f"d_{i}"
        corpus[doc_id] = NormalizedDocument(
            doc_id=doc_id,
            content=f"Contenido del documento {i}",
            title=f"Titulo {i}",
        )

    # Crear queries: cada una con gold_per_query docs relevantes
    for q_idx in range(n_queries):
        gold_ids = [
            f"d_{q_idx * gold_per_query + g}"
            for g in range(gold_per_query)
            if (q_idx * gold_per_query + g) < n_corpus
        ]
        queries.append(NormalizedQuery(
            query_id=f"q_{q_idx}",
            query_text=f"Pregunta numero {q_idx}",
            relevant_doc_ids=gold_ids,
            expected_answer=f"Respuesta {q_idx}",
            answer_type="text",
        ))

    dataset = LoadedDataset(
        name="test_dataset",
        dataset_type=DatasetType.HYBRID,
        queries=queries,
        corpus=corpus,
        primary_metric=MetricType.F1_SCORE,
    )
    return dataset


# =================================================================
# Tests
# =================================================================

def test_gold_docs_present_in_corpus():
    """Todos los gold docs de las queries seleccionadas estan en el corpus."""
    evaluator = _make_evaluator(dev_queries=5, dev_corpus_size=30)
    dataset = _make_dataset(n_queries=20, n_corpus=100, gold_per_query=2)

    queries, corpus = evaluator._select_subset_dev(dataset)

    for q in queries:
        for gold_id in q.relevant_doc_ids:
            if gold_id in dataset.corpus:  # solo si existe en dataset original
                assert gold_id in corpus, (
                    f"Gold doc {gold_id} de query {q.query_id} "
                    f"no esta en corpus resultante"
                )


def test_distractors_fill_to_dev_corpus_size():
    """Corpus resultante tiene exactamente dev_corpus_size docs."""
    dev_corpus_size = 30
    evaluator = _make_evaluator(dev_queries=5, dev_corpus_size=dev_corpus_size)
    dataset = _make_dataset(n_queries=20, n_corpus=100, gold_per_query=2)

    queries, corpus = evaluator._select_subset_dev(dataset)

    assert len(corpus) == dev_corpus_size, (
        f"Esperado {dev_corpus_size} docs, obtenido {len(corpus)}"
    )


def test_correct_number_of_queries():
    """Se seleccionan exactamente dev_queries queries."""
    dev_queries = 5
    evaluator = _make_evaluator(dev_queries=dev_queries, dev_corpus_size=30)
    dataset = _make_dataset(n_queries=20, n_corpus=100)

    queries, corpus = evaluator._select_subset_dev(dataset)

    assert len(queries) == dev_queries, (
        f"Esperado {dev_queries} queries, obtenido {len(queries)}"
    )


def test_seed_deterministic():
    """Misma seed produce mismos queries y corpus."""
    seed = 42
    dataset = _make_dataset(n_queries=20, n_corpus=100)

    ev1 = _make_evaluator(dev_queries=5, dev_corpus_size=30, seed=seed)
    queries1, corpus1 = ev1._select_subset_dev(dataset)

    ev2 = _make_evaluator(dev_queries=5, dev_corpus_size=30, seed=seed)
    queries2, corpus2 = ev2._select_subset_dev(dataset)

    ids1 = [q.query_id for q in queries1]
    ids2 = [q.query_id for q in queries2]
    assert ids1 == ids2, f"Queries difieren: {ids1} vs {ids2}"

    corpus_ids1 = sorted(corpus1.keys())
    corpus_ids2 = sorted(corpus2.keys())
    assert corpus_ids1 == corpus_ids2, "Corpus difieren con misma seed"


def test_different_seed_different_result():
    """Seeds diferentes producen diferentes selecciones."""
    dataset = _make_dataset(n_queries=20, n_corpus=100)

    ev1 = _make_evaluator(dev_queries=5, dev_corpus_size=30, seed=42)
    queries1, _ = ev1._select_subset_dev(dataset)

    ev2 = _make_evaluator(dev_queries=5, dev_corpus_size=30, seed=99)
    queries2, _ = ev2._select_subset_dev(dataset)

    ids1 = [q.query_id for q in queries1]
    ids2 = [q.query_id for q in queries2]
    # Con alta probabilidad, seeds distintas dan queries distintas
    assert ids1 != ids2, "Seeds diferentes deberian dar resultados diferentes"


def test_gold_exceeds_corpus_size_raises_error():
    """Si gold docs > dev_corpus_size, lanza ValueError."""
    # 10 queries * 2 gold = 20 gold docs, pero dev_corpus_size=10
    evaluator = _make_evaluator(dev_queries=10, dev_corpus_size=10)
    dataset = _make_dataset(n_queries=10, n_corpus=100, gold_per_query=2)

    with pytest.raises(ValueError, match="gold docs"):
        evaluator._select_subset_dev(dataset)


def test_dev_queries_exceeds_total_uses_all():
    """Si dev_queries >= total queries, usa todas las queries."""
    evaluator = _make_evaluator(dev_queries=100, dev_corpus_size=200)
    dataset = _make_dataset(n_queries=10, n_corpus=200)

    queries, corpus = evaluator._select_subset_dev(dataset)

    assert len(queries) == 10, (
        f"Esperado todas (10) queries, obtenido {len(queries)}"
    )


def test_gold_docs_missing_from_corpus_handled():
    """Gold docs que no existen en corpus no causan fallo."""
    evaluator = _make_evaluator(dev_queries=3, dev_corpus_size=20)
    dataset = _make_dataset(n_queries=5, n_corpus=50)

    # Agregar un query con gold doc inexistente
    dataset.queries.append(NormalizedQuery(
        query_id="q_missing",
        query_text="Pregunta con gold inexistente",
        relevant_doc_ids=["d_nonexistent"],
        expected_answer="Respuesta",
    ))

    # No debe fallar
    queries, corpus = evaluator._select_subset_dev(dataset)
    assert len(queries) <= evaluator.config.dev_queries


def test_corpus_contains_mix_of_gold_and_distractors():
    """Corpus tiene gold docs + distractores (no solo gold)."""
    evaluator = _make_evaluator(dev_queries=3, dev_corpus_size=20)
    dataset = _make_dataset(n_queries=20, n_corpus=100, gold_per_query=2)

    queries, corpus = evaluator._select_subset_dev(dataset)

    gold_ids = set()
    for q in queries:
        gold_ids.update(q.relevant_doc_ids)

    gold_in_corpus = gold_ids & set(corpus.keys())
    non_gold_in_corpus = set(corpus.keys()) - gold_ids

    assert len(gold_in_corpus) > 0, "Debe haber gold docs en corpus"
    assert len(non_gold_in_corpus) > 0, "Debe haber distractores en corpus"


if __name__ == "__main__":
    test_gold_docs_present_in_corpus()
    test_distractors_fill_to_dev_corpus_size()
    test_correct_number_of_queries()
    test_seed_deterministic()
    test_different_seed_different_result()
    test_gold_exceeds_corpus_size_raises_error()
    test_dev_queries_exceeds_total_uses_all()
    test_gold_docs_missing_from_corpus_handled()
    test_corpus_contains_mix_of_gold_and_distractors()
