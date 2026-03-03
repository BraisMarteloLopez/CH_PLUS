"""
Retrieval strategies for RAG evaluation.

Estrategias soportadas:
  - SIMPLE_VECTOR: embedding search puro via ChromaDB
  - CONTEXTUAL_HYBRID: enriquecimiento LLM (Anthropic pattern) + BM25+Vector+RRF
  - CONTEXTUAL_HYBRID_PLUS: CONTEXTUAL_HYBRID + entity cross-linking (spaCy NER)
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
    ContextualRetrieverPlus,
    LLMContextGenerator,
    EnrichedChunk,
)
from .entity_linker import HAS_SPACY

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
        llm_service: Requerido para CONTEXTUAL_HYBRID / CONTEXTUAL_HYBRID_PLUS.
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

    if strategy == RetrievalStrategy.CONTEXTUAL_HYBRID_PLUS:
        if llm_service is None:
            raise ValueError(
                "CONTEXTUAL_HYBRID_PLUS requiere llm_service para "
                "generar contextos de enriquecimiento durante la indexacion."
            )
        context_generator = LLMContextGenerator(
            llm_service=llm_service,
            max_tokens=config.context_max_tokens,
        )
        return ContextualRetrieverPlus(
            config=config,
            embedding_model=embedding_model,
            context_generator=context_generator,
            collection_name=collection_name,
            embedding_batch_size=embedding_batch_size,
            max_cross_refs=config.entity_max_cross_refs,
            min_shared_entities=config.entity_min_shared,
            max_entity_doc_fraction=config.entity_max_doc_fraction,
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
    "ContextualRetrieverPlus",
    "LLMContextGenerator",
    "HAS_SPACY",
    "get_retriever",
]
