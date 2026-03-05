"""
Modulo: Hybrid Plus Retriever
Descripcion: Busqueda hibrida BM25+Vector+RRF con LiteGraph inline +
             graph expansion via entity cross-linking (spaCy NER).

Ubicacion: shared/retrieval/hybrid_plus_retriever.py

Flujo:
    1. NER sobre contenido original de cada doc (si spaCy disponible)
    2. EntityLinker genera cross-refs textuales Y grafo estructurado
    3. Indexacion DUAL:
       - Vector index: contenido ORIGINAL (embeddings limpios)
       - BM25 index: contenido + cross-refs inline (bridge terms)
    4. Retrieval: BM25+Vector+RRF -> graph expansion (capped) -> swap
       a original para generacion

Doble mecanismo cross-ref:
    - INLINE: cross-refs en texto indexado mejoran BM25 (bridge terms).
      Embeddings vectoriales permanecen limpios (sin contaminacion).
    - GRAPH EXPANSION: vecinos del grafo se anaden al pool de candidatos
      post-retrieval (limitado por max_graph_expansion). Safety net para
      bridge questions donde un hop no aparece en el top-K inicial.
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
    Hybrid retrieval (BM25+Vector+RRF) con LiteGraph inline + graph expansion.

    Doble uso de entity cross-linking:
      1. INLINE: cross-refs textuales en contenido indexado (BM25 + embeddings)
      2. EXPANSION: grafo de vecinos para ampliar pool de candidatos post-retrieval

    Contenido para generacion: siempre ORIGINAL (sin cross-refs).
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

        # Cross-ref graph: doc_id -> [related_doc_ids]
        self._cross_ref_graph: Dict[str, List[str]] = {}

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
                # Texto inline para indexacion
                cross_refs = linker.compute_cross_refs(documents)
                # Grafo estructurado para expansion en retrieval
                # (build_index ya fue llamado por compute_cross_refs)
                for doc in documents:
                    doc_id = doc.get("doc_id", "")
                    neighbors = linker.get_cross_ref_graph(doc_id)
                    if neighbors:
                        self._cross_ref_graph[doc_id] = neighbors
                self._linker_stats = linker.get_stats()
                logger.info(
                    f"HybridPlusRetriever: {len(cross_refs)} docs con inline refs, "
                    f"{len(self._cross_ref_graph)} nodos con vecinos en grafo"
                )
            else:
                logger.warning(
                    "HYBRID_PLUS: spaCy no disponible. "
                    "Indexando sin cross-refs (BM25+Vector+RRF puro)."
                )

            # Paso 2: Construir docs para indexacion dual
            # - original_docs: embeddings limpios (vector index)
            # - enriched_docs: contenido + cross-refs (BM25 index)
            original_docs = []
            enriched_docs = []
            has_any_refs = False

            for doc in documents:
                doc_id = doc.get("doc_id", "")
                content = doc.get("content", "")
                title = doc.get("title", "")

                # Guardar original para swap durante generacion
                self._original_contents[doc_id] = content

                original_docs.append({
                    "doc_id": doc_id,
                    "content": content,
                    "title": title,
                })

                # BM25: original + cross-refs inline
                refs = cross_refs.get(doc_id, "")
                if refs:
                    enriched_content = f"{content} {refs}"
                    has_any_refs = True
                else:
                    enriched_content = content

                enriched_docs.append({
                    "doc_id": doc_id,
                    "content": enriched_content,
                    "title": title,
                })

            # Paso 3: Indexar en inner retriever (dual indexing)
            # Vector recibe contenido original (embeddings limpios),
            # BM25 recibe contenido enriquecido (bridge terms para keyword match)
            bm25_arg = enriched_docs if has_any_refs else None
            result = self._inner_retriever.index_documents(
                original_docs,
                collection_name=collection_name,
                bm25_documents=bm25_arg,
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
        result = self._expand_with_graph(result)
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
        result = self._expand_with_graph(result)
        return self._swap_to_original_contents(result)

    def _expand_with_graph(self, result: RetrievalResult) -> RetrievalResult:
        """Expande resultados con vecinos del grafo de cross-refs.

        Para cada doc en el resultado, busca vecinos en el grafo y los
        anade al pool si no estan ya presentes. Vecinos entran con score
        bajo para no desplazar candidatos originales, pero si para entrar
        en el pool que el reranker evaluara.

        Expansion limitada por ``config.max_graph_expansion`` (0 = sin limite).
        """
        if not self._cross_ref_graph:
            result.metadata["graph_expanded"] = 0
            return result

        max_expand = self.config.max_graph_expansion

        existing_ids = set(result.doc_ids)
        expanded_ids: List[str] = []
        expanded_contents: List[str] = []
        expanded_scores: List[float] = []

        min_score = min(result.scores) if result.scores else 0.0
        neighbor_base_score = min_score * 0.5

        for doc_id in result.doc_ids:
            if max_expand > 0 and len(expanded_ids) >= max_expand:
                break
            neighbors = self._cross_ref_graph.get(doc_id, [])
            for neighbor_id in neighbors:
                if max_expand > 0 and len(expanded_ids) >= max_expand:
                    break
                if neighbor_id not in existing_ids:
                    content = self._original_contents.get(neighbor_id, "")
                    if content:
                        expanded_ids.append(neighbor_id)
                        expanded_contents.append(content)
                        expanded_scores.append(neighbor_base_score)
                        existing_ids.add(neighbor_id)
                        neighbor_base_score *= 0.9

        result.doc_ids.extend(expanded_ids)
        result.contents.extend(expanded_contents)
        result.scores.extend(expanded_scores)

        if result.vector_scores:
            result.vector_scores.extend([0.0] * len(expanded_ids))
        if result.bm25_scores:
            result.bm25_scores.extend([0.0] * len(expanded_ids))

        result.metadata["graph_expanded"] = len(expanded_ids)

        if expanded_ids:
            logger.debug(
                f"Graph expansion: +{len(expanded_ids)} vecinos "
                f"(total {len(result.doc_ids)} candidatos)"
            )

        return result

    def _swap_to_original_contents(
        self, result: RetrievalResult
    ) -> RetrievalResult:
        """Reemplaza contenido enriquecido por original para generacion."""
        result.contents = [
            self._original_contents.get(doc_id, content)
            for doc_id, content in zip(result.doc_ids, result.contents)
        ]
        result.strategy_used = RetrievalStrategy.HYBRID_PLUS
        result.metadata["entity_cross_linking"] = bool(self._cross_ref_graph) or bool(self._linker_stats)
        result.metadata["linker_stats"] = self._linker_stats
        return result

    def clear_index(self) -> None:
        self._inner_retriever.clear_index()
        self._original_contents.clear()
        self._cross_ref_graph.clear()
        self._linker_stats = {}
        self._is_indexed = False
        logger.debug("HybridPlusRetriever: indice, grafo y mapa limpiados")


__all__ = [
    "HybridPlusRetriever",
]
