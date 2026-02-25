"""
Módulo: Contextual Retriever
Descripción: Contextual Retrieval según paper de Anthropic.
             Enriquece documentos con contexto LLM antes de indexar.

Ubicación: shared/retrieval/contextual_retriever.py

Flujo:
    1. Pre-indexación: Documento -> LLM -> Contexto breve (50-100 tokens)
    2. Enriquecimiento: Contexto + Documento original -> Documento enriquecido
    3. Indexación: Documento enriquecido -> inner retriever
    4. Retrieval: Query -> inner retriever -> Resultados
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from shared.types import EmbeddingModelProtocol, LLMJudgeProtocol

from .core import (
    BaseRetriever,
    RetrievalConfig,
    RetrievalResult,
    RetrievalStrategy,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ESTRUCTURAS
# =============================================================================

@dataclass
class EnrichedChunk:
    """Chunk enriquecido con contexto generado por LLM."""
    chunk_id: str
    original_content: str
    generated_context: str
    full_document_title: Optional[str] = None

    def get_enriched_text(self) -> str:
        """Contexto generado + contenido original (orden Anthropic: context first)."""
        return f"{self.generated_context}\n\n{self.original_content}"


# =============================================================================
# PROMPTS — Optimizados para modelos pequeños (nemotron-3-nano ~1B params)
# =============================================================================
# Los modelos nano no manejan bien XML complejo ni instrucciones abstractas.
# Se usan headers en texto plano (Document: / Chunk:) e instrucciones directas.
#
# Modo A (con documento padre):
#   El LLM recibe el documento padre (truncado) y el chunk específico.
#   Genera 1-2 frases que sitúan el chunk dentro del documento.
#
# Modo B (sin documento padre — fallback):
#   Solo se dispone del chunk y opcionalmente un título.
#   El LLM genera un contexto basado en el contenido del chunk.
# =============================================================================

CONTEXT_SYSTEM_PROMPT = (
    "You are a helpful assistant. You always reply with a short sentence. "
    "Never reply with an empty message."
)

DOCUMENT_CONTEXT_PROMPT = """Document:
{doc_content}

Chunk:
{chunk_content}

Write 1-2 sentences describing what this chunk is about and how it fits in the document. Be specific, mention names and topics."""

FALLBACK_CONTEXT_PROMPT = """Document title: {title}

Chunk:
{chunk_content}

Write 1-2 sentences describing what this chunk is about. Be specific, mention names and topics."""


# =============================================================================
# GENERADOR DE CONTEXTO CON LLM
# =============================================================================

class LLMContextGenerator:
    """
    Genera contexto descriptivo para chunks usando un LLM.

    Implementa la técnica Contextual Retrieval de Anthropic:
    - Modo A (con parent): pasa documento padre + chunk al LLM.
    - Modo B (fallback): pasa solo chunk + título al LLM.

    Solo expone la API async (batch). El flujo siempre es:
    ContextualRetriever -> _run_batch_generation -> generate_contexts_batch
    """

    def __init__(
        self,
        llm_service: LLMJudgeProtocol,
        max_tokens: int = 1000,
    ):
        self.llm_service = llm_service
        self.max_tokens = max_tokens
        self._cache: Dict[str, str] = {}
        self._total_generated = 0
        self._total_cache_hits = 0
        self._total_errors = 0
        self._total_with_parent = 0
        self._total_fallback = 0

    async def _generate_one(
        self,
        chunk_content: str,
        parent_content: Optional[str] = None,
        document_title: Optional[str] = None,
    ) -> str:
        """
        Genera contexto para un chunk de forma asíncrona.

        Si parent_content está disponible, usa el prompt con documento padre
        (truncado a 2000 chars). Si no, usa el fallback (solo chunk + título).
        Los inputs se truncan para respetar el context window de modelos nano.
        """
        cache_key = self._make_cache_key(
            chunk_content, parent_content or document_title or ""
        )

        if cache_key in self._cache:
            self._total_cache_hits += 1
            return self._cache[cache_key]

        # Truncar inputs para modelos con context window limitado
        truncated_chunk = chunk_content[:1000]

        if parent_content:
            # Modo A: prompt con documento padre (truncado)
            truncated_parent = parent_content[:2000]
            user_prompt = DOCUMENT_CONTEXT_PROMPT.format(
                doc_content=truncated_parent,
                chunk_content=truncated_chunk,
            )
            self._total_with_parent += 1
        else:
            # Modo B: fallback sin documento padre
            title = document_title or "Untitled"
            user_prompt = FALLBACK_CONTEXT_PROMPT.format(
                title=title, chunk_content=truncated_chunk
            )
            self._total_fallback += 1

        try:
            response = await self.llm_service.invoke_async(
                user_prompt,
                system_prompt=CONTEXT_SYSTEM_PROMPT,
                max_tokens=self.max_tokens,
            )
            context = str(response).strip()

            # Validación post-strip: captura respuestas de solo espacios/newlines
            if not context:
                raise ValueError("LLM returned empty context after strip")

            self._cache[cache_key] = context
            self._total_generated += 1
            return context

        except Exception as e:
            logger.warning(f"Error generando contexto: {e}. Usando fallback.")
            self._total_errors += 1
            # Fallback mejorado: incluye keywords reales del chunk para retrieval
            chunk_preview = chunk_content[:200].replace("\n", " ")
            fallback = f"Document: {document_title or 'Untitled'}. Content: {chunk_preview}"
            self._cache[cache_key] = fallback
            return fallback

    async def generate_contexts_batch(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 10,
    ) -> List[EnrichedChunk]:
        """
        Genera contextos para un lote de documentos concurrentemente.

        Cada dict en documents debe tener:
          - doc_id: str
          - content: str (texto del chunk)
          - title: str (opcional)
          - parent_content: str (opcional, texto del documento padre)
        """
        enriched_chunks: List[EnrichedChunk] = []
        total = len(documents)

        logger.info(f"Generando contextos para {total} chunks (batch={batch_size})...")

        for batch_start in range(0, total, batch_size):
            batch = documents[batch_start : batch_start + batch_size]

            tasks = []
            for doc in batch:
                tasks.append(self._generate_one(
                    chunk_content=doc.get("content", ""),
                    parent_content=doc.get("parent_content"),
                    document_title=doc.get("title"),
                ))

            contexts = await asyncio.gather(*tasks, return_exceptions=True)

            for doc, ctx in zip(batch, contexts):
                if isinstance(ctx, Exception):
                    logger.warning(f"Excepción en batch: {ctx}")
                    ctx = f"Document: {doc.get('title', 'Untitled')}."

                enriched_chunks.append(EnrichedChunk(
                    chunk_id=doc.get("doc_id", ""),
                    original_content=doc.get("content", ""),
                    generated_context=str(ctx),
                    full_document_title=doc.get("title"),
                ))

            batch_end = batch_start + len(batch)
            if batch_end % 50 == 0 or batch_end == total:
                logger.info(f"  Contextos generados: {batch_end}/{total}")

        logger.info(
            f"Generación completada: {self._total_generated} generados, "
            f"{self._total_cache_hits} cache hits, {self._total_errors} errores"
        )
        return enriched_chunks

    @staticmethod
    def _make_cache_key(chunk: str, full_document: str) -> str:
        raw = f"{chunk[:500]}|{full_document[:500]}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def get_stats(self) -> Dict[str, int]:
        return {
            "total_generated": self._total_generated,
            "cache_hits": self._total_cache_hits,
            "errors": self._total_errors,
            "cache_size": len(self._cache),
            "with_parent": self._total_with_parent,
            "fallback": self._total_fallback,
        }

    def clear_cache(self) -> None:
        self._cache.clear()


# =============================================================================
# CONTEXTUAL RETRIEVER
# =============================================================================

class ContextualRetriever(BaseRetriever):
    """
    Retriever que enriquece documentos con contexto LLM antes de indexar.

    Patrón decorador: envuelve un inner retriever (Simple o Hybrid)
    y le pasa documentos enriquecidos.
    """

    def __init__(
        self,
        config: RetrievalConfig,
        embedding_model: EmbeddingModelProtocol,
        context_generator: LLMContextGenerator,
        inner_retriever: Optional[BaseRetriever] = None,
        collection_name: Optional[str] = None,
        embedding_batch_size: int = 0,
    ):
        super().__init__(config)
        self.embedding_model = embedding_model
        self.context_generator = context_generator

        # Mapa doc_id -> contenido original (sin enriquecimiento).
        # El texto enriquecido se usa para indexación/búsqueda;
        # el original se devuelve para generación de respuesta.
        self._original_contents: Dict[str, str] = {}

        if inner_retriever is not None:
            self._inner_retriever = inner_retriever
        else:
            # CONTEXTUAL_HYBRID: enrichment + hybrid (BM25+Vector+RRF)
            from .hybrid_retriever import HybridRetriever, HAS_BM25, HAS_TANTIVY
            if HAS_TANTIVY or HAS_BM25:
                self._inner_retriever = HybridRetriever(
                    config, embedding_model, collection_name,
                    embedding_batch_size=embedding_batch_size,
                )
            else:
                from .core import SimpleVectorRetriever
                logger.warning(
                    "Ni tantivy ni rank-bm25 disponible, CONTEXTUAL_HYBRID usara "
                    "SimpleVector como inner retriever (sin BM25)"
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
        """Enriquece documentos con contexto LLM e indexa en el inner retriever."""
        if not documents:
            logger.warning("index_documents llamado con lista vacía")
            return False

        start_time = time.perf_counter()
        logger.info(f"ContextualRetriever: enriqueciendo {len(documents)} documentos...")

        try:
            enriched_chunks = self._run_batch_generation(documents)

            enriched_docs = []
            for chunk in enriched_chunks:
                self._original_contents[chunk.chunk_id] = chunk.original_content
                enriched_docs.append({
                    "doc_id": chunk.chunk_id,
                    "content": chunk.get_enriched_text(),
                    "title": chunk.full_document_title or "",
                })

            result = self._inner_retriever.index_documents(
                enriched_docs, collection_name=collection_name
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._is_indexed = result

            logger.info(
                f"ContextualRetriever: indexación {elapsed_ms:.0f}ms. "
                f"Stats: {self.context_generator.get_stats()}"
            )
            return result

        except Exception as e:
            logger.error(f"Error en indexación contextual: {e}")
            return False

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Delega búsqueda al inner retriever (indexado con texto enriquecido)
        pero devuelve contenido original para generación de respuesta.
        """
        result = self._inner_retriever.retrieve(query, top_k)
        return self._swap_to_original_contents(result)

    def retrieve_by_vector(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Retrieval con vector pre-computado, delegando al inner retriever.
        """
        result = self._inner_retriever.retrieve_by_vector(
            query_text, query_vector, top_k
        )
        return self._swap_to_original_contents(result)

    def _swap_to_original_contents(
        self, result: RetrievalResult
    ) -> RetrievalResult:
        """
        Sustituye contenido enriquecido por original.

        El enriquecimiento mejora matching de embeddings/BM25,
        pero el LLM generador necesita el texto limpio.
        """
        result.contents = [
            self._original_contents.get(doc_id, content)
            for doc_id, content in zip(result.doc_ids, result.contents)
        ]

        result.strategy_used = RetrievalStrategy.CONTEXTUAL_HYBRID

        result.metadata["contextual_enrichment"] = True
        result.metadata["context_generator_stats"] = self.context_generator.get_stats()
        return result

    def clear_index(self) -> None:
        self._inner_retriever.clear_index()
        self.context_generator.clear_cache()
        self._original_contents.clear()
        self._is_indexed = False
        logger.debug("ContextualRetriever: índice, cache y mapa de originales limpiados")

    def _run_batch_generation(
        self, documents: List[Dict[str, Any]]
    ) -> List[EnrichedChunk]:
        """Ejecuta generacion batch de forma sincrona."""
        from shared.llm import run_sync
        batch_size = self.config.context_batch_size
        return run_sync(
            self.context_generator.generate_contexts_batch(
                documents, batch_size=batch_size
            )
        )


__all__ = [
    "LLMContextGenerator",
    "ContextualRetriever",
    "EnrichedChunk",
]
