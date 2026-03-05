"""
Retrieval strategies for RAG evaluation.

Estrategias soportadas:
  - SIMPLE_VECTOR: embedding search puro via ChromaDB
  - HYBRID_PLUS: BM25+Vector+RRF + entity cross-linking (spaCy NER)
  - LIGHT_RAG: Vector + Knowledge Graph dual-level (LLM triplet extraction)
"""

import logging
from typing import Optional

from shared.types import EmbeddingModelProtocol

from .core import (
    RetrievalStrategy,
    RetrievalConfig,
    RetrievalResult,
    BaseRetriever,
    SimpleVectorRetriever,
)
from .hybrid_retriever import HybridRetriever, HAS_BM25, HAS_TANTIVY
from .hybrid_plus_retriever import HybridPlusRetriever
from .lightrag_retriever import LightRAGRetriever
from .tantivy_index import TantivyIndex
from .entity_linker import HAS_SPACY
from .knowledge_graph import HAS_NETWORKX

logger = logging.getLogger(__name__)


def get_retriever(
    config: RetrievalConfig,
    embedding_model: EmbeddingModelProtocol,
    collection_name: Optional[str] = None,
    embedding_batch_size: int = 0,
    llm_service: Optional[object] = None,
) -> BaseRetriever:
    """
    Factory para obtener un retriever segun la estrategia en config.

    Args:
        config: Configuracion de retrieval con estrategia seleccionada.
        embedding_model: Modelo de embeddings (NVIDIAEmbeddings o compatible).
        collection_name: Nombre de la coleccion ChromaDB.
        embedding_batch_size: Batch size para embeddings (0 = default).
        llm_service: AsyncLLMService (requerido para LIGHT_RAG).

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

    if strategy == RetrievalStrategy.HYBRID_PLUS:
        return HybridPlusRetriever(
            config=config,
            embedding_model=embedding_model,
            collection_name=collection_name,
            embedding_batch_size=embedding_batch_size,
            max_cross_refs=config.entity_max_cross_refs,
            min_shared_entities=config.entity_min_shared,
            max_entity_doc_fraction=config.entity_max_doc_fraction,
        )

    if strategy == RetrievalStrategy.LIGHT_RAG:
        return LightRAGRetriever(
            config=config,
            embedding_model=embedding_model,
            llm_service=llm_service,
            collection_name=collection_name,
            embedding_batch_size=embedding_batch_size,
            kg_max_hops=config.kg_max_hops,
            kg_max_text_chars=config.kg_max_text_chars,
            graph_weight=config.kg_graph_weight,
            vector_weight=config.kg_vector_weight,
        )

    raise ValueError(f"Estrategia no soportada: {strategy}")


__all__ = [
    "RetrievalStrategy",
    "RetrievalConfig",
    "RetrievalResult",
    "BaseRetriever",
    "SimpleVectorRetriever",
    "HybridRetriever",
    "HybridPlusRetriever",
    "LightRAGRetriever",
    "HAS_SPACY",
    "HAS_NETWORKX",
    "get_retriever",
]
