"""
Tests de conectividad: verifican que los servicios estan accesibles.

Son el prerequisito para los demas tests de integracion.
Si estos fallan, el resto no tiene sentido ejecutarlo.

Ejecucion:
    pytest tests/integration/test_connectivity.py -v
"""

import json
import urllib.request

import pytest


# =========================================================================
# NIM EMBEDDING
# =========================================================================

@pytest.mark.integration
class TestEmbeddingConnectivity:

    def test_embedding_health(self, embedding_base_url):
        """GET /v1/models del NIM de embeddings responde 200."""
        url = f"{embedding_base_url}/models"
        req = urllib.request.Request(url, method="GET")
        req.add_header("Accept", "application/json")

        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "data" in data
            assert len(data["data"]) > 0, "No hay modelos disponibles"

    def test_embedding_model_listed(self, embedding_base_url, mteb_config):
        """El modelo configurado aparece en la lista de modelos."""
        url = f"{embedding_base_url}/models"
        req = urllib.request.Request(url, method="GET")
        req.add_header("Accept", "application/json")

        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        model_ids = [m.get("id", "") for m in data.get("data", [])]
        expected = mteb_config.infra.embedding_model_name
        assert expected in model_ids, (
            f"Modelo '{expected}' no encontrado. Disponibles: {model_ids}"
        )


# =========================================================================
# NIM LLM
# =========================================================================

@pytest.mark.integration
class TestLLMConnectivity:

    def test_llm_health(self, llm_base_url):
        """GET /v1/models del NIM LLM responde 200."""
        url = f"{llm_base_url}/models"
        req = urllib.request.Request(url, method="GET")
        req.add_header("Accept", "application/json")

        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "data" in data
            assert len(data["data"]) > 0

    def test_llm_context_window_available(self, llm_base_url):
        """El modelo LLM expone max_model_len (necesario para auto-deteccion)."""
        url = f"{llm_base_url}/models"
        req = urllib.request.Request(url, method="GET")
        req.add_header("Accept", "application/json")

        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        model = data["data"][0]
        max_len = model.get("max_model_len")
        assert max_len is not None, (
            f"Modelo '{model.get('id')}' no expone max_model_len. "
            "Configurar GENERATION_MAX_CONTEXT_CHARS manualmente en .env."
        )
        assert isinstance(max_len, (int, float)) and max_len > 0


# =========================================================================
# MINIO
# =========================================================================

@pytest.mark.integration
class TestMinIOConnectivity:

    def test_minio_connection(self, minio_loader):
        """MinIO responde a head_bucket (bucket existe y es accesible)."""
        assert minio_loader.check_connection(), (
            f"No se puede conectar a MinIO: {minio_loader.endpoint}/{minio_loader.bucket}"
        )

    def test_minio_dataset_exists(self, minio_loader, mteb_config):
        """El dataset configurado existe en MinIO."""
        datasets = minio_loader.list_available_datasets()
        expected = mteb_config.dataset_name
        assert expected in datasets, (
            f"Dataset '{expected}' no encontrado en MinIO. "
            f"Disponibles: {datasets}"
        )
