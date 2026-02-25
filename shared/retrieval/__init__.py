"""
Retrieval strategies for RAG evaluation.

Estrategias soportadas:
  - SIMPLE_VECTOR: embedding search puro via ChromaDB
  - CONTEXTUAL_HYBRID: enriquecimiento LLM (Anthropic pattern) + BM25+Vector+RRF
"""

import logging
from typing import Optional

from shared.types import EmbeddingModelProtocol, LLMJudgeProtocol

from .core import (
    RetrievalStrategy,
    RetrievalConfig,
    RetrievalResult,
    BaseRetriever,
    SimpleVectorRetriever,
)
from .hybrid_retriever import HybridRetriever, HAS_BM25, HAS_TANTIVY
from .tantivy_index import TantivyIndex
from .contextual_retriever import (
    ContextualRetriever,
    LLMContextGenerator,
    EnrichedChunk,
)

logger = logging.getLogger(__name__)


def get_retriever(
    config: RetrievalConfig,
    embedding_model: EmbeddingModelProtocol,
    collection_name: Optional[str] = None,
    embedding_batch_size: int = 0,
    llm_service: Optional[LLMJudgeProtocol] = None,
) -> BaseRetriever:
    """
    Factory para obtener un retriever segun la estrategia en config.

    Args:
        config: Configuracion de retrieval con estrategia seleccionada.
        embedding_model: Modelo de embeddings (NVIDIAEmbeddings o compatible).
        collection_name: Nombre de la coleccion ChromaDB.
        embedding_batch_size: Batch size para embeddings (0 = default).
        llm_service: Requerido para CONTEXTUAL_HYBRID (AsyncLLMService).
                     Usado para generar contextos de enriquecimiento.

    Returns:
        BaseRetriever configurado segun la estrategia.
    """
    strategy = config.strategy
    logger.info(f"Factory: creando retriever {strategy.name}")

    if strategy == RetrievalStrategy.SIMPLE_VECTOR:
        return SimpleVectorRetriever(
            config, embedding_model, collection_name,
            embedding_batch_size=embedding_batch_size,
        )

    if strategy == RetrievalStrategy.CONTEXTUAL_HYBRID:
        if llm_service is None:
            raise ValueError(
                "CONTEXTUAL_HYBRID requiere llm_service para generar "
                "contextos de enriquecimiento durante la indexacion."
            )
        context_generator = LLMContextGenerator(
            llm_service=llm_service,
            max_tokens=config.context_max_tokens,
        )
        return ContextualRetriever(
            config=config,
            embedding_model=embedding_model,
            context_generator=context_generator,
            collection_name=collection_name,
            embedding_batch_size=embedding_batch_size,
        )

    raise ValueError(f"Estrategia no soportada: {strategy}")


__all__ = [
    "RetrievalStrategy",
    "RetrievalConfig",
    "RetrievalResult",
    "BaseRetriever",
    "SimpleVectorRetriever",
    "HybridRetriever",
    "ContextualRetriever",
    "LLMContextGenerator",
    "get_retriever",
]
