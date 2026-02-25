"""
Tests de componentes individuales con infraestructura real.

Verifican que cada pieza funciona correctamente con datos reales,
sin ejecutar el pipeline completo.

Prerequisito: test_connectivity.py pasa.

Ejecucion:
    pytest tests/integration/test_components.py -v
"""

import json
import urllib.request

import pytest


# =========================================================================
# EMBEDDING BATCH
# =========================================================================

@pytest.mark.integration
class TestEmbeddingBatch:

    def test_batch_embed_produces_vectors(self, embedding_base_url, mteb_config):
        """Embeber 3 queries reales produce vectores de dimension consistente."""
        queries = [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "When was the first moon landing?",
        ]

        url = f"{embedding_base_url}/embeddings"
        payload = {
            "input": queries,
            "model": mteb_config.infra.embedding_model_name,
        }
        if mteb_config.infra.embedding_model_type == "asymmetric":
            payload["input_type"] = "query"

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        items = sorted(data["data"], key=lambda x: x["index"])
        assert len(items) == 3, f"Esperados 3 vectores, recibidos {len(items)}"

        # Todos deben tener la misma dimension
        dims = [len(item["embedding"]) for item in items]
        assert len(set(dims)) == 1, f"Dimensiones inconsistentes: {dims}"
        assert dims[0] > 0, "Dimension del vector es 0"

    def test_embedding_dimension_reasonable(self, embedding_base_url, mteb_config):
        """La dimension del embedding esta en un rango razonable (64-8192)."""
        url = f"{embedding_base_url}/embeddings"
        payload = {
            "input": ["test"],
            "model": mteb_config.infra.embedding_model_name,
        }
        if mteb_config.infra.embedding_model_type == "asymmetric":
            payload["input_type"] = "query"

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        dim = len(data["data"][0]["embedding"])
        assert 64 <= dim <= 8192, (
            f"Dimension {dim} fuera de rango razonable (64-8192)"
        )


# =========================================================================
# LLM GENERATION
# =========================================================================

@pytest.mark.integration
class TestLLMGeneration:

    def test_llm_generates_response(self, llm_base_url, mteb_config):
        """El LLM genera una respuesta no vacia para un prompt trivial."""
        url = f"{llm_base_url}/chat/completions"
        payload = {
            "model": mteb_config.infra.llm_model_name,
            "messages": [
                {"role": "user", "content": "What is 2 + 2? Answer with just the number."}
            ],
            "max_tokens": 128,
            "temperature": 0.0,
        }

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        choices = data.get("choices", [])
        assert len(choices) > 0, "LLM no retorno choices"
        choice = choices[0]
        content = choice.get("message", {}).get("content") or ""
        finish = choice.get("finish_reason", "unknown")
        assert len(content.strip()) > 0, (
            f"LLM retorno respuesta vacia (finish_reason={finish}, "
            f"model={mteb_config.infra.llm_model_name})"
        )

    def test_context_window_detection(self, mteb_config):
        """_query_model_context_window() retorna un entero positivo."""
        from sandbox_mteb.evaluator import MTEBEvaluator

        evaluator = MTEBEvaluator(mteb_config)
        ctx_window = evaluator._query_model_context_window()

        assert ctx_window is not None, (
            "No se pudo detectar context window. "
            "Verificar que el NIM LLM expone max_model_len en GET /v1/models."
        )
        assert ctx_window > 0, f"Context window invalido: {ctx_window}"


# =========================================================================
# MINIO DATASET
# =========================================================================

@pytest.mark.integration
class TestMinIODataset:

    def test_load_dataset_schema(self, minio_loader, mteb_config):
        """Cargar el dataset produce queries, corpus y qrels con estructura correcta."""
        dataset = minio_loader.load_dataset(mteb_config.dataset_name)

        assert dataset.load_status == "success", (
            f"Carga fallida: {dataset.error_message}"
        )
        assert len(dataset.queries) > 0, "No hay queries"
        assert len(dataset.corpus) > 0, "No hay corpus"

        # Verificar estructura de una query
        q = dataset.queries[0]
        assert q.query_id, "query_id vacio"
        assert q.query_text, "query_text vacio"
        assert len(q.relevant_doc_ids) > 0, (
            f"Query '{q.query_id}' sin relevant_doc_ids (qrels vacios?)"
        )

        # Verificar que los gold docs existen en el corpus
        for doc_id in q.relevant_doc_ids:
            assert doc_id in dataset.corpus, (
                f"Gold doc '{doc_id}' de query '{q.query_id}' no esta en corpus"
            )

    def test_dataset_counts_match_metadata(self, minio_loader, mteb_config):
        """Los conteos del dataset coinciden con total_queries y total_corpus."""
        dataset = minio_loader.load_dataset(mteb_config.dataset_name)

        assert dataset.load_status == "success"
        assert dataset.total_queries == len(dataset.queries)
        assert dataset.total_corpus == len(dataset.corpus)
