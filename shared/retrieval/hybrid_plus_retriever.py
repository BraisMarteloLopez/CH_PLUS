"""
Modulo: Hybrid Plus Retriever
Descripcion: Busqueda hibrida BM25+Vector+RRF con graph expansion via
             entity cross-linking (spaCy NER).

Ubicacion: shared/retrieval/hybrid_plus_retriever.py

Flujo:
    1. NER sobre contenido original de cada doc (si spaCy disponible)
    2. EntityLinker.compute_cross_ref_graph() -> grafo doc_id -> [vecinos]
    3. Contenido indexado = contenido ORIGINAL (sin contaminar)
    4. HybridRetriever indexa texto limpio (BM25+Vector)
    5. Retrieval: BM25+Vector+RRF -> graph expansion -> candidatos ampliados
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
    Hybrid retrieval (BM25+Vector+RRF) con graph expansion via entity cross-linking.

    A diferencia del diseno anterior que contaminaba el contenido indexado con
    texto de cross-refs, este retriever:
      - Indexa contenido LIMPIO (embeddings y BM25 sin ruido)
      - Usa el grafo de entidades para EXPANDIR resultados durante retrieval
      - Los vecinos del grafo se anaden al pool de candidatos antes del reranker

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

        # Entity linker config
        self._max_cross_refs = max_cross_refs
        self._min_shared_entities = min_shared_entities
        self._max_entity_doc_fraction = max_entity_doc_fraction
        self._linker_stats: Dict[str, Any] = {}

        # Cross-ref graph: doc_id -> [related_doc_ids]
        self._cross_ref_graph: Dict[str, List[str]] = {}

        # Doc content map for graph expansion lookups
        self._doc_content_map: Dict[str, str] = {}

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
            # Guardar mapa de contenidos para graph expansion
            for doc in documents:
                self._doc_content_map[doc.get("doc_id", "")] = doc.get("content", "")

            # Paso 1: Construir grafo de cross-refs (solo si spaCy disponible)
            from .entity_linker import HAS_SPACY
            if HAS_SPACY:
                from .entity_linker import EntityLinker
                linker = EntityLinker(
                    max_cross_refs=self._max_cross_refs,
                    min_shared_entities=self._min_shared_entities,
                    max_entity_doc_fraction=self._max_entity_doc_fraction,
                )
                self._cross_ref_graph = linker.compute_cross_ref_graph(documents)
                self._linker_stats = linker.get_stats()
                logger.info(
                    f"HybridPlusRetriever: grafo con "
                    f"{len(self._cross_ref_graph)} nodos con vecinos"
                )
            else:
                logger.warning(
                    "HYBRID_PLUS: spaCy no disponible. "
                    "Indexando sin cross-refs (BM25+Vector+RRF puro)."
                )

            # Paso 2: Indexar contenido LIMPIO en inner retriever
            result = self._inner_retriever.index_documents(
                documents, collection_name=collection_name
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
        return self._expand_with_graph(result, top_k)

    def retrieve_by_vector(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        result = self._inner_retriever.retrieve_by_vector(
            query_text, query_vector, top_k
        )
        return self._expand_with_graph(result, top_k)

    def _expand_with_graph(
        self, result: RetrievalResult, top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """Expande resultados de retrieval usando el grafo de cross-refs.

        Para cada doc en el top-K inicial, busca vecinos en el grafo y los
        anade al pool de candidatos si no estan ya presentes.
        Los vecinos se insertan con score decrementado para no desplazar
        a los candidatos originales de alto score, pero si para entrar
        en el pool que el reranker evaluara.
        """
        if not self._cross_ref_graph:
            result.strategy_used = RetrievalStrategy.HYBRID_PLUS
            result.metadata["entity_cross_linking"] = False
            result.metadata["graph_expanded"] = 0
            return result

        existing_ids = set(result.doc_ids)
        expanded_ids: List[str] = []
        expanded_contents: List[str] = []
        expanded_scores: List[float] = []

        # Score minimo de los resultados originales (para asignar a vecinos)
        min_score = min(result.scores) if result.scores else 0.0
        neighbor_base_score = min_score * 0.5  # Vecinos entran con score bajo

        # Para cada doc recuperado, traer sus vecinos del grafo
        for doc_id in result.doc_ids:
            neighbors = self._cross_ref_graph.get(doc_id, [])
            for neighbor_id in neighbors:
                if neighbor_id not in existing_ids:
                    content = self._doc_content_map.get(neighbor_id, "")
                    if content:
                        expanded_ids.append(neighbor_id)
                        expanded_contents.append(content)
                        expanded_scores.append(neighbor_base_score)
                        existing_ids.add(neighbor_id)
                        neighbor_base_score *= 0.9  # Decay para cada vecino extra

        # Append vecinos al final del resultado
        result.doc_ids.extend(expanded_ids)
        result.contents.extend(expanded_contents)
        result.scores.extend(expanded_scores)

        # Extender vector_scores y bm25_scores si existen
        if result.vector_scores:
            result.vector_scores.extend([0.0] * len(expanded_ids))
        if result.bm25_scores:
            result.bm25_scores.extend([0.0] * len(expanded_ids))

        result.strategy_used = RetrievalStrategy.HYBRID_PLUS
        result.metadata["entity_cross_linking"] = True
        result.metadata["linker_stats"] = self._linker_stats
        result.metadata["graph_expanded"] = len(expanded_ids)

        if expanded_ids:
            logger.debug(
                f"Graph expansion: +{len(expanded_ids)} vecinos anadidos "
                f"(total {len(result.doc_ids)} candidatos)"
            )

        return result

    def clear_index(self) -> None:
        self._inner_retriever.clear_index()
        self._cross_ref_graph.clear()
        self._doc_content_map.clear()
        self._linker_stats = {}
        self._is_indexed = False
        logger.debug("HybridPlusRetriever: indice y grafo limpiados")


__all__ = [
    "HybridPlusRetriever",
]
