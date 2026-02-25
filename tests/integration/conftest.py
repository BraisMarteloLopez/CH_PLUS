"""
Conftest para tests de integracion.

Carga la configuracion REAL desde sandbox_mteb/.env y verifica
que las variables criticas existen. Si no hay .env o faltan variables,
todos los tests de integracion se saltan automaticamente.

Requiere: NIM endpoints + MinIO accesibles.

Ejecucion:
    pytest tests/integration/ -v       # solo integracion
    pytest tests/                      # todo junto (unit + integracion)
"""

import os
import logging
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)


def _load_env() -> bool:
    """Carga .env del sandbox y verifica que las variables criticas existen."""
    env_path = Path(__file__).resolve().parent.parent.parent / "sandbox_mteb" / ".env"
    if not env_path.exists():
        return False

    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(env_path), override=True)

    required_vars = [
        "EMBEDDING_BASE_URL",
        "EMBEDDING_MODEL_NAME",
        "LLM_BASE_URL",
        "LLM_MODEL_NAME",
        "MINIO_ENDPOINT",
        "MINIO_BUCKET_NAME",
    ]
    for var in required_vars:
        if not os.getenv(var):
            logger.warning(f"Variable {var} no definida en {env_path}")
            return False
    return True


# ---------------------------------------------------------------------------
# SKIP GLOBAL si el entorno no esta preparado
# ---------------------------------------------------------------------------

if not _load_env():
    pytest.skip(
        "Integration tests: .env no encontrado o incompleto. "
        "Copiar sandbox_mteb/env.example a sandbox_mteb/.env con valores reales.",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mteb_config():
    """Config real cargada desde .env."""
    from sandbox_mteb.config import MTEBConfig
    env_path = Path(__file__).resolve().parent.parent.parent / "sandbox_mteb" / ".env"
    config = MTEBConfig.from_env(str(env_path))
    errors = config.validate()
    if errors:
        pytest.skip(f"Config invalida: {errors}")
    return config


@pytest.fixture(scope="session")
def minio_loader(mteb_config):
    """MinIOLoader con conexion real."""
    from sandbox_mteb.loader import MinIOLoader
    return MinIOLoader(mteb_config.storage)


@pytest.fixture(scope="session")
def embedding_base_url():
    return os.getenv("EMBEDDING_BASE_URL", "").rstrip("/")


@pytest.fixture(scope="session")
def llm_base_url():
    return os.getenv("LLM_BASE_URL", "").rstrip("/")
