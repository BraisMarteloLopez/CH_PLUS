"""
Modulo: Config Base
Descripcion: Helpers para lectura de .env y sub-configs compartidos.

Ubicacion: shared/config_base.py

NO contiene instancias globales. Cada sandbox construye su config
en su propio entry point via from_env().

Sub-configs compartidos:
  - InfraConfig: LLM, embeddings, concurrencia NIM
  - RerankerConfig: cross-encoder post-retrieval
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS .env
# =============================================================================

def load_env_file(env_path: Optional[str] = None) -> None:
    """Carga .env una unica vez. Llamar desde el entry point del sandbox."""
    if env_path:
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv()


def _env(var: str, default: str = "") -> str:
    return os.getenv(var, default).strip()


def _env_int(var: str, default: int) -> int:
    raw = os.getenv(var)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"{var}='{raw}' no es entero. Usando default={default}")
        return default


def _env_float(var: str, default: float) -> float:
    raw = os.getenv(var)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"{var}='{raw}' no es float. Usando default={default}")
        return default


def _env_bool(var: str, default: bool = False) -> bool:
    raw = os.getenv(var)
    if raw is None:
        return default
    return raw.strip().lower() in ("true", "1", "yes")


def _env_path(var: str, default: str = "") -> Path:
    return Path(os.getenv(var, default).strip())


# Alias
load_dotenv_file = load_env_file


# =============================================================================
# SUB-CONFIGS COMPARTIDOS
# =============================================================================

@dataclass
class InfraConfig:
    """Infraestructura NIM: LLM, embeddings, concurrencia."""
    llm_base_url: str = ""
    llm_model_name: str = ""
    embedding_base_url: str = ""
    embedding_model_name: str = ""
    embedding_model_type: str = "symmetric"  # "symmetric" | "asymmetric"
    embedding_batch_size: int = 50
    nim_max_concurrent: int = 32
    nim_timeout: int = 120
    nim_max_retries: int = 3

    @classmethod
    def from_env(cls) -> "InfraConfig":
        return cls(
            llm_base_url=_env("LLM_BASE_URL"),
            llm_model_name=_env("LLM_MODEL_NAME"),
            embedding_base_url=_env("EMBEDDING_BASE_URL"),
            embedding_model_name=_env("EMBEDDING_MODEL_NAME"),
            embedding_model_type=_env("EMBEDDING_MODEL_TYPE", "symmetric"),
            embedding_batch_size=_env_int("EMBEDDING_BATCH_SIZE", 50),
            nim_max_concurrent=_env_int("NIM_MAX_CONCURRENT_REQUESTS", 32),
            nim_timeout=_env_int("NIM_REQUEST_TIMEOUT", 120),
            nim_max_retries=_env_int("NIM_MAX_RETRIES", 3),
        )

    def validate(self) -> List[str]:
        errors = []
        if not self.llm_base_url:
            errors.append("LLM_BASE_URL no definida")
        if not self.llm_model_name:
            errors.append("LLM_MODEL_NAME no definida")
        if not self.embedding_base_url or not self.embedding_model_name:
            errors.append("EMBEDDING_BASE_URL y EMBEDDING_MODEL_NAME requeridos")
        if self.embedding_model_type not in ("symmetric", "asymmetric"):
            errors.append(
                f"EMBEDDING_MODEL_TYPE='{self.embedding_model_type}' no valido. "
                "Usar 'symmetric' o 'asymmetric'"
            )
        if not 1 <= self.nim_max_concurrent <= 128:
            errors.append(
                f"NIM_MAX_CONCURRENT={self.nim_max_concurrent} fuera de rango (1-128)"
            )
        return errors


# NOTA: GenerationConfig y EvaluationConfig eliminadas (codigo muerto).
# - GenerationConfig: parametros (temperature, max_tokens) se pasan
#   directamente a AsyncLLMService, no via config.
# - EvaluationConfig: max_queries/max_corpus estan en MTEBConfig;
#   db_type/continue_on_error/verbose/log_every_n nunca se usaron.


@dataclass
class RerankerConfig:
    """Cross-encoder reranking post-retrieval."""
    enabled: bool = False
    base_url: str = ""
    model_name: str = ""
    top_n: int = 20

    @classmethod
    def from_env(cls) -> "RerankerConfig":
        return cls(
            enabled=_env_bool("RERANKER_ENABLED"),
            base_url=_env("RERANKER_BASE_URL"),
            model_name=_env("RERANKER_MODEL_NAME"),
            top_n=_env_int("RERANKER_TOP_N", 20),
        )

    def validate(self) -> List[str]:
        errors = []
        if self.enabled:
            if not self.base_url or not self.model_name:
                errors.append("Reranker habilitado pero falta base_url o model_name")
        return errors


def ensure_directories(*dirs: Path) -> None:
    """Crea directorios si no existen."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


__all__ = [
    "load_env_file",
    "load_dotenv_file",
    "InfraConfig",
    "RerankerConfig",
    "ensure_directories",
]
