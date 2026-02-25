"""
Modulo: Reranker
Descripcion: Cross-encoder reranking para post-filtrado de candidatos.

Ubicacion: shared/retrieval/reranker.py

Flujo:
    Retriever -> top-N candidatos -> Reranker -> top-K reordenados
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from .core import RetrievalResult

logger = logging.getLogger(__name__)

try:
    from langchain_nvidia_ai_endpoints import NVIDIARerank
    from langchain_core.documents import Document
    HAS_NVIDIA_RERANK = True
except ImportError:
    HAS_NVIDIA_RERANK = False
    NVIDIARerank = None
    Document = None


class CrossEncoderReranker:
    """
    Reranker basado en cross-encoder (NVIDIA NIM).

    El cross-encoder evalua cada par (query, documento) conjuntamente,
    produciendo scores mas precisos que bi-encoders, pero mas lento.
    Solo se aplica sobre candidatos pre-filtrados.
    """

    def __init__(self, base_url: str, model_name: str):
        if not HAS_NVIDIA_RERANK:
            raise ImportError(
                "langchain-nvidia-ai-endpoints no soporta NVIDIARerank. "
                "Verificar version: pip install -U langchain-nvidia-ai-endpoints"
            )

        self.base_url = base_url
        self.model_name = model_name

        self._reranker = NVIDIARerank(
            model=model_name,
            base_url=base_url,
            mode="nim",
        )

        logger.info(f"CrossEncoderReranker: {model_name} @ {base_url}")

    def rerank(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        top_n: int = 20,
    ) -> RetrievalResult:
        """
        Reordena los documentos de un RetrievalResult usando el cross-encoder.

        Args:
            query: Query original
            retrieval_result: Resultado de retrieval con candidatos
            top_n: Numero de documentos a retener tras reranking
        """
        if not retrieval_result.doc_ids:
            return retrieval_result

        start_time = time.perf_counter()
        n_candidates = len(retrieval_result.doc_ids)

        lc_docs = []
        for i, (doc_id, content) in enumerate(
            zip(retrieval_result.doc_ids, retrieval_result.contents)
        ):
            lc_docs.append(
                Document(
                    page_content=content,
                    metadata={"doc_id": doc_id, "original_index": i},
                )
            )

        try:
            # NVIDIARerank.top_n controla cuantos docs devuelve la API.
            # Default interno es 5; lo seteamos por llamada.
            self._reranker.top_n = top_n
            reranked_docs = self._reranker.compress_documents(lc_docs, query)

            # FIX DT-8: ordenar explicitamente por relevance_score descendente.
            # compress_documents() no garantiza orden formalmente en su contrato.
            reranked_docs = sorted(
                reranked_docs,
                key=lambda d: d.metadata.get("relevance_score", 0.0),
                reverse=True,
            )

            new_doc_ids = []
            new_contents = []
            new_scores = []

            for doc in reranked_docs[:top_n]:
                new_doc_ids.append(doc.metadata["doc_id"])
                new_contents.append(doc.page_content)
                new_scores.append(
                    doc.metadata.get("relevance_score", 0.0)
                )

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            logger.debug(
                f"Reranked {n_candidates} -> {len(new_doc_ids)} docs "
                f"en {elapsed_ms:.0f}ms"
            )

            return RetrievalResult(
                doc_ids=new_doc_ids,
                contents=new_contents,
                scores=new_scores,
                retrieval_time_ms=(
                    retrieval_result.retrieval_time_ms + elapsed_ms
                ),
                strategy_used=retrieval_result.strategy_used,
                metadata={
                    **retrieval_result.metadata,
                    "reranked": True,
                    "reranker_model": self.model_name,
                    "candidates_before_rerank": n_candidates,
                    "rerank_time_ms": elapsed_ms,
                },
            )

        except Exception as e:
            logger.error(
                f"Error en reranking: {e}. Retornando sin rerank."
            )
            return RetrievalResult(
                doc_ids=retrieval_result.doc_ids[:top_n],
                contents=retrieval_result.contents[:top_n],
                scores=retrieval_result.scores[:top_n],
                retrieval_time_ms=retrieval_result.retrieval_time_ms,
                strategy_used=retrieval_result.strategy_used,
                metadata={
                    **retrieval_result.metadata,
                    "reranked": False,
                    "rerank_error": str(e),
                },
            )


__all__ = ["CrossEncoderReranker", "HAS_NVIDIA_RERANK"]
