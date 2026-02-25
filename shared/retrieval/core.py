"""
Modulo: Retrieval Core
Descripcion: Tipos base, configuracion, y SimpleVectorRetriever.

Ubicacion: shared/retrieval/core.py

Consolida base.py + simple_retriever.py en un unico archivo.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from shared.types import EmbeddingModelProtocol

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERACIONES
# =============================================================================

class RetrievalStrategy(Enum):
    """Estrategias de retrieval disponibles."""
    SIMPLE_VECTOR = auto()
    CONTEXTUAL_HYBRID = auto()


# =============================================================================
# CONFIGURACION
# =============================================================================

@dataclass
class RetrievalConfig:
    """
    Configuracion para una estrategia de retrieval.
    Todos los parametros operativos viajan en este objeto.
    """
    strategy: RetrievalStrategy = RetrievalStrategy.SIMPLE_VECTOR
    retrieval_k: int = 20

    # Pesos RRF (CONTEXTUAL_HYBRID)
    bm25_weight: float = 0.5
    vector_weight: float = 0.5
    rrf_k: int = 60
    pre_fusion_k: int = 150

    # Contextual retrieval (optimizado para modelos nano)
    context_max_tokens: int = 100
    context_batch_size: int = 50

    # BM25
    bm25_language: str = "en"

    # HNSW (ChromaDB): num_threads=1 reduce no-determinismo del grafo
    # (elimina variabilidad de threading). No garantiza reproducibilidad
    # total: ChromaDB no soporta hnsw:random_seed. Ver DTm-13.
    hnsw_num_threads: int = 1

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        from shared.config_base import _env, _env_int, _env_float
        return cls(
            strategy=RetrievalStrategy[_env("RETRIEVAL_STRATEGY", "SIMPLE_VECTOR")],
            retrieval_k=_env_int("RETRIEVAL_K", 20),
            bm25_weight=_env_float("RETRIEVAL_BM25_WEIGHT", 0.5),
            vector_weight=_env_float("RETRIEVAL_VECTOR_WEIGHT", 0.5),
            pre_fusion_k=_env_int("RETRIEVAL_PRE_FUSION_K", 150),
            rrf_k=_env_int("RETRIEVAL_RRF_K", 60),
            context_max_tokens=_env_int("RETRIEVAL_CONTEXT_MAX_TOKENS", 100),
            context_batch_size=_env_int("RETRIEVAL_CONTEXT_BATCH_SIZE", 50),
            bm25_language=_env("RETRIEVAL_BM25_LANGUAGE", "en"),
            hnsw_num_threads=_env_int("HNSW_NUM_THREADS", 1),
        )


# =============================================================================
# RESULTADO
# =============================================================================

@dataclass
class RetrievalResult:
    """Resultado de una operacion de retrieval."""
    doc_ids: List[str]
    contents: List[str]
    scores: List[float]

    bm25_scores: Optional[List[float]] = None
    vector_scores: Optional[List[float]] = None

    retrieval_time_ms: float = 0.0
    strategy_used: RetrievalStrategy = RetrievalStrategy.SIMPLE_VECTOR
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# CLASE BASE ABSTRACTA
# =============================================================================

class BaseRetriever(ABC):
    """Clase base abstracta para implementaciones de retrieval."""

    def __init__(self, config: RetrievalConfig):
        self.config = config
        self._is_indexed = False

    @abstractmethod
    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> RetrievalResult:
        pass

    def retrieve_by_vector(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Retrieval usando vector pre-computado.

        Evita la llamada REST al NIM de embeddings por query.
        Default: fallback a retrieve() (subclases pueden optimizar).
        """
        return self.retrieve(query_text, top_k)

    @abstractmethod
    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> bool:
        pass

    def clear_index(self) -> None:
        self._is_indexed = False

    @property
    def is_indexed(self) -> bool:
        return self._is_indexed


# =============================================================================
# SIMPLE VECTOR RETRIEVER
# =============================================================================

class SimpleVectorRetriever(BaseRetriever):
    """Retriever usando solo busqueda vectorial (ChromaDB)."""

    def __init__(
        self,
        config: RetrievalConfig,
        embedding_model: EmbeddingModelProtocol,
        collection_name: Optional[str] = None,
        embedding_batch_size: int = 0,
    ):
        super().__init__(config)
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.embedding_batch_size = embedding_batch_size
        self._vector_store: Any = None  # ChromaVectorStore, lazy import

    def _init_vector_store(self, collection_name: str) -> None:
        from shared.vector_store import ChromaVectorStore

        store_config = {
            "CHROMA_COLLECTION_NAME": collection_name,
            "EMBEDDING_BATCH_SIZE": self.embedding_batch_size,
            "HNSW_NUM_THREADS": self.config.hnsw_num_threads,
        }
        self._vector_store = ChromaVectorStore(
            store_config, self.embedding_model
        )
        logger.debug(f"Vector store inicializado: {collection_name}")

    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> bool:
        try:
            from langchain_core.documents import Document

            name = (
                collection_name or self.collection_name or "default_collection"
            )
            self._init_vector_store(name)

            lc_docs = [
                Document(
                    page_content=doc.get("content", ""),
                    metadata={
                        "doc_id": doc.get("doc_id", ""),
                        "title": doc.get("title", ""),
                    },
                )
                for doc in documents
            ]

            self._vector_store.add_documents(lc_docs)
            self._is_indexed = True
            logger.info(f"Indexados {len(lc_docs)} documentos en '{name}'")
            return True

        except Exception as e:
            logger.error(f"Error indexando documentos: {e}")
            return False

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        k = top_k or self.config.retrieval_k

        if not self._vector_store:
            logger.warning("Vector store no inicializado")
            return RetrievalResult(
                doc_ids=[], contents=[], scores=[],
                strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
            )

        start_time = time.perf_counter()

        try:
            results = self._vector_store.similarity_search_with_score(
                query, k=k
            )

            doc_ids = []
            contents = []
            scores = []

            for doc, score in results:
                doc_ids.append(doc.metadata.get("doc_id", ""))
                contents.append(doc.page_content)
                scores.append(float(score))

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return RetrievalResult(
                doc_ids=doc_ids,
                contents=contents,
                scores=scores,
                vector_scores=scores,
                retrieval_time_ms=elapsed_ms,
                strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
            )

        except Exception as e:
            logger.error(f"Error en retrieval: {e}")
            return RetrievalResult(
                doc_ids=[], contents=[], scores=[],
                strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
                metadata={"error": str(e)},
            )

    def retrieve_by_vector(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """Busqueda vectorial usando embedding pre-computado."""
        k = top_k or self.config.retrieval_k

        if not self._vector_store:
            logger.warning("Vector store no inicializado")
            return RetrievalResult(
                doc_ids=[], contents=[], scores=[],
                strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
            )

        start_time = time.perf_counter()

        try:
            results = self._vector_store.similarity_search_by_vector_with_score(
                query_vector, k=k
            )

            doc_ids = []
            contents = []
            scores = []

            for doc, score in results:
                doc_ids.append(doc.metadata.get("doc_id", ""))
                contents.append(doc.page_content)
                scores.append(float(score))

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return RetrievalResult(
                doc_ids=doc_ids,
                contents=contents,
                scores=scores,
                vector_scores=scores,
                retrieval_time_ms=elapsed_ms,
                strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
            )

        except Exception as e:
            logger.error(f"Error en retrieval por vector: {e}")
            return RetrievalResult(
                doc_ids=[], contents=[], scores=[],
                strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
                metadata={"error": str(e)},
            )

    def clear_index(self) -> None:
        if self._vector_store:
            self._vector_store.delete_all_documents()
        self._is_indexed = False
        logger.debug("Indice limpiado")


__all__ = [
    "RetrievalStrategy",
    "RetrievalConfig",
    "RetrievalResult",
    "BaseRetriever",
    "SimpleVectorRetriever",
]
