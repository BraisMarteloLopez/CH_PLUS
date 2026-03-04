"""
Modulo: Hybrid Plus Retriever
Descripcion: Busqueda hibrida BM25+Vector+RRF con entity cross-linking (spaCy NER).
             Sin LLM enrichment. Sin dependencia de NIM para indexacion.

Ubicacion: shared/retrieval/hybrid_plus_retriever.py

Flujo:
    1. NER sobre contenido original de cada doc (si spaCy disponible)
    2. EntityLinker.build_index() -> cross-refs por doc
    3. Texto indexado = contenido original + cross-refs
    4. HybridRetriever indexa texto final (BM25+Vector)
    5. Retrieval: swap a contenido original para generacion
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from shared.types import EmbeddingModelProtocol

from .core import (
    BaseRetriever,
    RetrievalConfig,
    RetrievalResult,
    RetrievalStrategy,
)

logger = logging.getLogger(__name__)


class HybridPlusRetriever(BaseRetriever):
    """
    Hybrid retrieval (BM25+Vector+RRF) con entity cross-linking (spaCy NER).

    Sin LLM enrichment. Sin dependencia de NIM para indexacion.
    Sin spaCy: se comporta como HybridRetriever puro (BM25+Vector+RRF).
    """

    def __init__(
        self,
        config: RetrievalConfig,
        embedding_model: EmbeddingModelProtocol,
        collection_name: Optional[str] = None,
        embedding_batch_size: int = 0,
        max_cross_refs: int = 3,
        min_shared_entities: int = 1,
        max_entity_doc_fraction: float = 0.05,
    ):
        super().__init__(config)
        self.embedding_model = embedding_model
        self._original_contents: Dict[str, str] = {}

        # Entity linker config
        self._max_cross_refs = max_cross_refs
        self._min_shared_entities = min_shared_entities
        self._max_entity_doc_fraction = max_entity_doc_fraction
        self._linker_stats: Dict[str, Any] = {}

        # Inner retriever (HybridRetriever: BM25+Vector+RRF)
        from .hybrid_retriever import HybridRetriever, HAS_BM25, HAS_TANTIVY
        if HAS_TANTIVY or HAS_BM25:
            self._inner_retriever = HybridRetriever(
                config, embedding_model, collection_name,
                embedding_batch_size=embedding_batch_size,
            )
        else:
            from .core import SimpleVectorRetriever
            logger.warning(
                "HYBRID_PLUS: ni tantivy ni rank-bm25 disponible, "
                "usando SimpleVector como inner retriever"
            )
            self._inner_retriever = SimpleVectorRetriever(
                config, embedding_model, collection_name,
                embedding_batch_size=embedding_batch_size,
            )

    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> bool:
        if not documents:
            logger.warning("index_documents llamado con lista vacia")
            return False

        start_time = time.perf_counter()
        logger.info(
            f"HybridPlusRetriever: indexando {len(documents)} documentos..."
        )

        try:
            # Paso 1: NER + cross-linking (solo si spaCy disponible)
            cross_refs: Dict[str, str] = {}
            from .entity_linker import HAS_SPACY
            if HAS_SPACY:
                from .entity_linker import EntityLinker
                linker = EntityLinker(
                    max_cross_refs=self._max_cross_refs,
                    min_shared_entities=self._min_shared_entities,
                    max_entity_doc_fraction=self._max_entity_doc_fraction,
                )
                cross_refs = linker.compute_cross_refs(documents)
                self._linker_stats = linker.get_stats()
            else:
                logger.warning(
                    "HYBRID_PLUS: spaCy no disponible. "
                    "Indexando sin cross-refs (BM25+Vector+RRF puro)."
                )

            # Paso 2: Construir docs para indexacion
            enriched_docs = []
            for doc in documents:
                doc_id = doc.get("doc_id", "")
                content = doc.get("content", "")
                title = doc.get("title", "")

                # Guardar original para swap durante generacion
                self._original_contents[doc_id] = content

                # Texto indexado = original + cross-refs
                refs = cross_refs.get(doc_id, "")
                if refs:
                    indexed_content = f"{content}\n\n{refs}"
                else:
                    indexed_content = content

                enriched_docs.append({
                    "doc_id": doc_id,
                    "content": indexed_content,
                    "title": title,
                })

            # Paso 3: Indexar en inner retriever
            result = self._inner_retriever.index_documents(
                enriched_docs, collection_name=collection_name
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._is_indexed = result

            logger.info(
                f"HybridPlusRetriever: indexacion {elapsed_ms:.0f}ms. "
                f"Linker stats: {self._linker_stats}"
            )
            return result

        except Exception as e:
            logger.error(f"Error en indexacion hybrid plus: {e}")
            return False

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        result = self._inner_retriever.retrieve(query, top_k)
        return self._swap_to_original_contents(result)

    def retrieve_by_vector(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        result = self._inner_retriever.retrieve_by_vector(
            query_text, query_vector, top_k
        )
        return self._swap_to_original_contents(result)

    def _swap_to_original_contents(
        self, result: RetrievalResult
    ) -> RetrievalResult:
        result.contents = [
            self._original_contents.get(doc_id, content)
            for doc_id, content in zip(result.doc_ids, result.contents)
        ]
        result.strategy_used = RetrievalStrategy.HYBRID_PLUS
        result.metadata["entity_cross_linking"] = True
        result.metadata["linker_stats"] = self._linker_stats
        return result

    def clear_index(self) -> None:
        self._inner_retriever.clear_index()
        self._original_contents.clear()
        self._linker_stats = {}
        self._is_indexed = False
        logger.debug("HybridPlusRetriever: indice y mapa limpiados")


__all__ = [
    "HybridPlusRetriever",
]
