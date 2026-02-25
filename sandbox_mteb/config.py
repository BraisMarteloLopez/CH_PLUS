"""
Configuracion para sandbox MTEB.

Toda la parametrizacion viene del .env. El entry point (run.py)
construye MTEBConfig.from_env() una sola vez.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional

from shared.config_base import (
    InfraConfig,
    RerankerConfig,
    _env,
    _env_int,
    _env_float,
    _env_bool,
    _env_path,
    load_dotenv_file,
)
from shared.retrieval.core import RetrievalConfig


# =========================================================================
# STORAGE (MinIO)
# =========================================================================

@dataclass
class MinIOStorageConfig:
    """Config de almacenamiento MinIO para datasets MTEB pre-descargados."""
    minio_endpoint: str = ""
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = ""
    s3_datasets_prefix: str = "datasets/evaluation"
    datasets_cache_dir: Path = Path("./data/datasets_cache")
    evaluation_results_dir: Path = Path("./data/results")
    vector_db_dir: Path = Path("./data/vector_db")

    @classmethod
    def from_env(cls) -> "MinIOStorageConfig":
        return cls(
            minio_endpoint=_env("MINIO_ENDPOINT", ""),
            minio_access_key=_env("MINIO_ACCESS_KEY", ""),
            minio_secret_key=_env("MINIO_SECRET_KEY", ""),
            minio_bucket=_env("MINIO_BUCKET_NAME", ""),
            s3_datasets_prefix=_env("S3_DATASETS_PREFIX", "datasets/evaluation"),
            datasets_cache_dir=_env_path("DATASETS_CACHE_DIR", "./data/datasets_cache"),
            evaluation_results_dir=_env_path("EVALUATION_RESULTS_DIR", "./data/results"),
            vector_db_dir=_env_path("VECTOR_DB_DIR", "./data/vector_db"),
        )

    def validate(self) -> List[str]:
        errors = []
        if not self.minio_endpoint:
            errors.append("MINIO_ENDPOINT no configurado")
        if not self.minio_bucket:
            errors.append("MINIO_BUCKET_NAME no configurado")
        return errors


# =========================================================================
# CONFIG PRINCIPAL
# =========================================================================

@dataclass
class MTEBConfig:
    """
    Configuracion completa para un run de evaluacion sobre datasets MTEB/BeIR.

    Composicion de sub-configs de shared/ + config especifico del sandbox.
    Se construye una sola vez en el entry point via from_env().
    """
    infra: InfraConfig = field(default_factory=InfraConfig)
    storage: MinIOStorageConfig = field(default_factory=MinIOStorageConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)

    # Especifico MTEB
    dataset_name: str = "hotpotqa"
    max_queries: int = 50
    max_corpus: int = 1000

    # Generacion (opcional, desactivable)
    generation_enabled: bool = True

    # Limite de caracteres para contexto de generacion.
    # Si 0: se deriva automaticamente del context window del modelo LLM
    # via GET /v1/models. Si el query falla, se usa 4000 como fallback.
    # Si >0: se usa directamente como override manual.
    generation_max_context_chars: int = 0

    # Shuffle del corpus para evitar sesgo de orden (None = no shuffle)
    corpus_shuffle_seed: Optional[int] = 42

    # Modo desarrollo: subset con gold docs garantizados.
    # Ignora max_queries y max_corpus cuando esta activo.
    dev_mode: bool = False
    dev_queries: int = 200
    dev_corpus_size: int = 4000

    @classmethod
    def from_env(cls, env_path: str = ".env") -> "MTEBConfig":
        """Construye config completa desde .env."""
        load_dotenv_file(env_path)

        return cls(
            infra=InfraConfig.from_env(),
            storage=MinIOStorageConfig.from_env(),
            retrieval=RetrievalConfig.from_env(),
            reranker=RerankerConfig.from_env(),
            dataset_name=_env("MTEB_DATASET_NAME", "hotpotqa"),
            max_queries=_env_int("EVAL_MAX_QUERIES", 50),
            max_corpus=_env_int("EVAL_MAX_CORPUS", 1000),
            generation_enabled=_env_bool("GENERATION_ENABLED", True),
            generation_max_context_chars=_env_int("GENERATION_MAX_CONTEXT_CHARS", 0),
            corpus_shuffle_seed=_env_int("CORPUS_SHUFFLE_SEED", 42) if _env_int("CORPUS_SHUFFLE_SEED", 42) >= 0 else None,
            dev_mode=_env_bool("DEV_MODE", False),
            dev_queries=_env_int("DEV_QUERIES", 200),
            dev_corpus_size=_env_int("DEV_CORPUS_SIZE", 4000),
        )

    def validate(self) -> List[str]:
        """Valida la configuracion. Retorna lista de errores (vacia = OK)."""
        errors = []
        errors.extend(self.storage.validate())

        if not self.infra.embedding_base_url:
            errors.append("EMBEDDING_BASE_URL no configurado")
        if not self.infra.embedding_model_name:
            errors.append("EMBEDDING_MODEL_NAME no configurado")

        # Estrategias validas para este sandbox
        from shared.retrieval.core import RetrievalStrategy
        VALID_STRATEGIES = (RetrievalStrategy.SIMPLE_VECTOR, RetrievalStrategy.CONTEXTUAL_HYBRID)
        if self.retrieval.strategy not in VALID_STRATEGIES:
            valid_names = ", ".join(s.name for s in VALID_STRATEGIES)
            errors.append(
                f"RETRIEVAL_STRATEGY={self.retrieval.strategy.name} no soportada "
                f"en sandbox_mteb. Valores validos: {valid_names}"
            )

        # LLM requerido si generacion activa O si estrategia es CONTEXTUAL_HYBRID
        needs_llm = (
            self.generation_enabled
            or self.retrieval.strategy == RetrievalStrategy.CONTEXTUAL_HYBRID
        )
        if needs_llm:
            if not self.infra.llm_base_url:
                reason = (
                    "CONTEXTUAL_HYBRID requiere LLM para enriquecimiento"
                    if self.retrieval.strategy == RetrievalStrategy.CONTEXTUAL_HYBRID
                    else "GENERATION_ENABLED=true"
                )
                errors.append(f"LLM_BASE_URL requerido ({reason})")
            if not self.infra.llm_model_name:
                errors.append("LLM_MODEL_NAME requerido")

        if self.max_queries < 0:
            errors.append(f"EVAL_MAX_QUERIES={self.max_queries} debe ser >= 0 (0=all)")
        if self.max_corpus < 0:
            errors.append(f"EVAL_MAX_CORPUS={self.max_corpus} debe ser >= 0 (0=all)")

        if self.dev_mode:
            if self.dev_queries <= 0:
                errors.append(f"DEV_QUERIES={self.dev_queries} debe ser > 0 cuando DEV_MODE=true")
            if self.dev_corpus_size <= 0:
                errors.append(f"DEV_CORPUS_SIZE={self.dev_corpus_size} debe ser > 0 cuando DEV_MODE=true")

        return errors

    def ensure_directories(self) -> None:
        """Crea directorios necesarios."""
        self.storage.datasets_cache_dir.mkdir(parents=True, exist_ok=True)
        self.storage.evaluation_results_dir.mkdir(parents=True, exist_ok=True)
        self.storage.vector_db_dir.mkdir(parents=True, exist_ok=True)

    def summary(self) -> str:
        """Resumen legible de la configuracion."""
        lines = [
            "=== MTEB Sandbox Config ===",
            f"  Dataset:    {self.dataset_name}",
            f"  Embedding:  {self.infra.embedding_model_name} ({self.infra.embedding_model_type})",
            f"  Strategy:   {self.retrieval.strategy.name}",
            f"  Reranker:   {'ON' if self.reranker.enabled else 'OFF'}",
            f"  Generation: {'ON' if self.generation_enabled else 'OFF'}",
        ]
        if self.dev_mode:
            lines.append(f"  DEV_MODE:   ON ({self.dev_queries} queries, {self.dev_corpus_size} corpus, gold docs garantizados)")
        else:
            lines.append(f"  Queries:    {self.max_queries if self.max_queries > 0 else 'ALL'}")
            lines.append(f"  Corpus:     {self.max_corpus if self.max_corpus > 0 else 'ALL'}")
        lines.extend([
            f"  Shuffle:    seed={self.corpus_shuffle_seed}" if self.corpus_shuffle_seed is not None else "  Shuffle:    OFF (WARNING: ordering bias risk)",
            f"  MinIO:      {self.storage.minio_endpoint}/{self.storage.minio_bucket}",
            f"  Results:    {self.storage.evaluation_results_dir}",
        ])
        return "\n".join(lines)


# =========================================================================
# PROMPTS DE GENERACION POR DATASET
# =========================================================================

GENERATION_PROMPTS: Dict[str, Dict[str, str]] = {
    "hotpotqa": {
        "system": (
            "You are a helpful assistant. Use the provided context to answer the question. "
            "Be concise and direct. For yes/no questions, start with yes or no."
        ),
        "user_template": "CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:",
    },
    # Datasets adicionales: agregar prompt cuando tengan ETL y datos en MinIO.
    "default": {
        "system": "You are a helpful assistant. Use the provided context to answer the question.",
        "user_template": "CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:",
    },
}


__all__ = [
    "MTEBConfig",
    "MinIOStorageConfig",
    "GENERATION_PROMPTS",
]
