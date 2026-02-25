"""
Modulo: Shared Types
Descripcion: Estructuras de datos normalizadas para evaluacion RAG.

Ubicacion: shared/types.py

Cambios respecto a la version original:
  - EvaluationRun (plano) reemplaza GlobalEvaluationReport (matrix multi-modelo)
  - LoadedDataset usa Dict index para queries (O(1) lookup vs O(n))
  - NormalizedDocument sin parent_content (eso vive en sandbox_anthropic)
  - Eliminado DatasetEvaluationResult (redundante con EvaluationRun)
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERACIONES
# =============================================================================

class DatasetType(Enum):
    """Clasificacion de datasets segun disponibilidad de Ground Truth."""
    HYBRID = auto()          # Tiene respuesta textual (F1, EM, Accuracy)
    RETRIEVAL_ONLY = auto()  # Solo doc_id relevante (metricas heuristicas)
    ADAPTED = auto()         # Requiere transformacion (ej: ArguAna)


class MetricType(Enum):
    """Tipos de metricas soportadas."""
    # Con referencia
    EXACT_MATCH = "exact_match"
    F1_SCORE = "f1_score"
    ACCURACY = "accuracy"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    # Sin referencia (LLM-Judge)
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"
    CONTEXT_UTILIZATION = "context_utilization"


class EvaluationStatus(Enum):
    """Estado de una evaluacion."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# =============================================================================
# ESTRUCTURAS NORMALIZADAS - DATOS
# =============================================================================

@dataclass
class NormalizedQuery:
    """Representacion normalizada de una query/pregunta."""
    query_id: str
    query_text: str
    relevant_doc_ids: List[str] = field(default_factory=list)
    expected_answer: Optional[str] = None
    answer_type: Optional[str] = None  # "text", "label", "counter_argument"
    supporting_facts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_text_answer(self) -> bool:
        return self.expected_answer is not None and self.answer_type in [
            "text", "counter_argument"
        ]

    def has_label_answer(self) -> bool:
        return self.answer_type == "label"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "relevant_doc_ids": self.relevant_doc_ids,
            "expected_answer": self.expected_answer,
            "answer_type": self.answer_type,
            "supporting_facts": self.supporting_facts,
            "metadata": self.metadata,
        }


@dataclass
class NormalizedDocument:
    """
    Representacion normalizada de un documento/chunk del corpus.

    Este tipo base NO contiene parent_content. Para Contextual Retrieval
    (Anthropic), usar ContextualDocument en sandbox_anthropic que hereda
    de este y agrega parent_content.
    """
    doc_id: str
    content: str
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "title": self.title,
            "metadata": self.metadata,
        }

    def get_full_text(self) -> str:
        """Retorna titulo + contenido concatenados para indexacion.

        Formato: "{titulo}\\n\\n{contenido}" si hay titulo, solo contenido si no.

        Nota: el titulo se antepone al contenido, lo que sesga el embedding
        hacia los terminos del titulo. Para titulos largos o genericos
        (ej: "Wikipedia"), el embedding representara mas el titulo que el
        contenido sustantivo. Esta es una decision de diseno aceptada para
        el caso de uso actual (Wikipedia con titulos cortos y descriptivos).
        """
        if self.title:
            return f"{self.title}\n\n{self.content}"
        return self.content


@dataclass
class LoadedDataset:
    """
    Contenedor para un dataset completamente cargado y normalizado.

    Cambio clave: queries indexadas en _query_index (Dict) para
    lookup O(1) en lugar de iteracion lineal O(n).
    """
    name: str
    dataset_type: DatasetType = DatasetType.HYBRID
    queries: List[NormalizedQuery] = field(default_factory=list)
    corpus: Dict[str, NormalizedDocument] = field(default_factory=dict)
    primary_metric: MetricType = MetricType.F1_SCORE
    secondary_metrics: List[MetricType] = field(default_factory=list)
    total_queries: int = 0
    total_corpus: int = 0
    load_status: str = "pending"
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Indice para lookup O(1)
    _query_index: Dict[str, NormalizedQuery] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        self._rebuild_query_index()

    def _rebuild_query_index(self) -> None:
        self._query_index = {q.query_id: q for q in self.queries}

    def get_query_by_id(self, query_id: str) -> Optional[NormalizedQuery]:
        """Busca una query por su ID. O(1)."""
        return self._query_index.get(query_id)

    def get_document_by_id(self, doc_id: str) -> Optional[NormalizedDocument]:
        return self.corpus.get(doc_id)

    def get_relevant_documents(
        self, query_id: str
    ) -> List[NormalizedDocument]:
        query = self._query_index.get(query_id)
        if not query:
            return []
        return [
            self.corpus[doc_id]
            for doc_id in query.relevant_doc_ids
            if doc_id in self.corpus
        ]

    def iterate_evaluation_pairs(
        self,
    ) -> Iterator[Tuple[NormalizedQuery, List[NormalizedDocument]]]:
        """Genera pares (query, documentos_relevantes) para evaluacion."""
        for query in self.queries:
            relevant_docs = [
                self.corpus[doc_id]
                for doc_id in query.relevant_doc_ids
                if doc_id in self.corpus
            ]
            yield query, relevant_docs

    def get_statistics(self) -> Dict[str, Any]:
        queries_with_answer = sum(
            1 for q in self.queries if q.expected_answer
        )
        queries_with_docs = sum(
            1 for q in self.queries if q.relevant_doc_ids
        )
        return {
            "name": self.name,
            "dataset_type": self.dataset_type.name,
            "queries_loaded": len(self.queries),
            "queries_total": self.total_queries,
            "queries_with_text_answer": queries_with_answer,
            "queries_with_relevant_docs": queries_with_docs,
            "corpus_loaded": len(self.corpus),
            "corpus_total": self.total_corpus,
            "primary_metric": self.primary_metric.value,
            "load_status": self.load_status,
        }


# =============================================================================
# RESULTADOS - NIVEL QUERY
# =============================================================================

@dataclass
class QueryRetrievalDetail:
    """
    Resultado de retrieval para una query individual.

    Separacion retrieval vs generacion (cuando reranker activo):
      - retrieved_doc_ids/contents/scores: resultado del retriever (pre-rerank),
        truncado a RETRIEVAL_K. Se usa para calcular metricas de retrieval.
      - generation_doc_ids/contents: resultado post-rerank, truncado a
        RERANKER_TOP_N. Se usa para generar respuesta con el LLM.

    NOTA: generation_doc_ids NO es necesariamente subconjunto de retrieved_doc_ids.
    El reranker opera sobre PRE_FUSION_K candidatos (ej: 150), mientras que
    retrieved_doc_ids solo almacena los top RETRIEVAL_K (ej: 20) para metricas.
    El reranker puede promover un doc de posicion 21-150 al top 5 de generacion.

    Sin reranker: generation_doc_ids queda vacio y get_generation_contents()
    devuelve retrieved_contents (comportamiento identico al anterior).
    """
    retrieved_doc_ids: List[str]
    retrieved_contents: List[str]
    retrieval_scores: List[float]
    expected_doc_ids: List[str]
    hit_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    retrieval_time_ms: float = 0.0

    # Docs para generacion (post-rerank). Vacio si no hay reranker.
    generation_doc_ids: List[str] = field(default_factory=list)
    generation_contents: List[str] = field(default_factory=list)

    # FIX DT-5: IDs de los candidatos que el reranker recibio (PRE_FUSION_K).
    # Solo IDs, sin contenidos (~3KB/query). Permite verificar post-hoc
    # que generation_doc_ids provienen del pool de candidatos, y analizar
    # que posiciones originales fueron promovidas por el reranker.
    pre_rerank_candidate_ids: List[str] = field(default_factory=list)

    # Metricas de retrieval efectivo (post-rerank). Solo cuando reranker activo.
    # Miden la calidad de los docs que realmente llegan al LLM de generacion.
    generation_recall: float = 0.0
    generation_hit: float = 0.0

    # Constante de clase, no se instancia ni serializa por query
    EVAL_K_VALUES: ClassVar[List[int]] = [1, 3, 5, 10, 20]

    def __post_init__(self) -> None:
        if self.expected_doc_ids and self.retrieved_doc_ids:
            self._calculate_all_metrics()
        if self.expected_doc_ids and self.generation_doc_ids:
            self._calculate_generation_metrics()

    def get_generation_contents(self) -> List[str]:
        """Contenidos para generacion: post-rerank si disponible, pre-rerank si no."""
        return self.generation_contents if self.generation_contents else self.retrieved_contents

    def _calculate_all_metrics(self) -> None:
        expected_set = set(self.expected_doc_ids)
        n_relevant = len(expected_set)

        for k in self.EVAL_K_VALUES:
            top_k_ids = set(self.retrieved_doc_ids[:k])

            # Hit@K
            self.hit_at_k[k] = 1.0 if top_k_ids & expected_set else 0.0

            # Recall@K
            self.recall_at_k[k] = (
                len(top_k_ids & expected_set) / n_relevant
                if n_relevant > 0
                else 0.0
            )

            # NDCG@K (relevancia binaria)
            dcg = sum(
                1.0 / math.log2(i + 2)
                for i, doc_id in enumerate(self.retrieved_doc_ids[:k])
                if doc_id in expected_set
            )
            idcg = sum(
                1.0 / math.log2(i + 2) for i in range(min(k, n_relevant))
            )
            self.ndcg_at_k[k] = dcg / idcg if idcg > 0 else 0.0

        # MRR
        self.mrr = 0.0
        for rank, doc_id in enumerate(self.retrieved_doc_ids, start=1):
            if doc_id in expected_set:
                self.mrr = 1.0 / rank
                break

    def _calculate_generation_metrics(self) -> None:
        """Metricas sobre generation_doc_ids (post-rerank)."""
        expected_set = set(self.expected_doc_ids)
        gen_set = set(self.generation_doc_ids)
        n_relevant = len(expected_set)
        self.generation_hit = 1.0 if gen_set & expected_set else 0.0
        self.generation_recall = (
            len(gen_set & expected_set) / n_relevant
            if n_relevant > 0
            else 0.0
        )


@dataclass
class GenerationResult:
    """Resultado de generacion para una query."""
    generated_response: str
    generation_time_ms: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    model_name: str = "unknown"


@dataclass
class QueryEvaluationResult:
    """Resultado completo de evaluacion para una query individual."""
    query_id: str
    query_text: str
    dataset_name: str
    dataset_type: DatasetType
    retrieval: QueryRetrievalDetail
    generation: Optional[GenerationResult] = None
    expected_response: Optional[str] = None
    primary_metric_type: MetricType = MetricType.F1_SCORE
    primary_metric_value: float = 0.0
    secondary_metrics: Dict[str, float] = field(default_factory=dict)
    status: EvaluationStatus = EvaluationStatus.COMPLETED
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_retrieval_success(self) -> bool:
        return self.retrieval.hit_at_k.get(5, 0.0) > 0.0

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "dataset_name": self.dataset_name,
            "dataset_type": self.dataset_type.name,
            "retrieved_doc_ids": self.retrieval.retrieved_doc_ids,
            "hit_at_k": self.retrieval.hit_at_k,
            "recall_at_k": self.retrieval.recall_at_k,
            "ndcg_at_k": self.retrieval.ndcg_at_k,
            "mrr": self.retrieval.mrr,
            "generated_response": (
                self.generation.generated_response if self.generation else ""
            ),
            "expected_response": self.expected_response,
            "primary_metric_type": self.primary_metric_type.value,
            "primary_metric_value": round(self.primary_metric_value, 4),
            "secondary_metrics": {
                k: round(v, 4) for k, v in self.secondary_metrics.items()
            },
            "status": self.status.value,
        }
        # Solo incluir generation_doc_ids si hay reranking (evita ruido en JSON)
        if self.retrieval.generation_doc_ids:
            result["generation_doc_ids"] = self.retrieval.generation_doc_ids
            result["generation_recall"] = round(self.retrieval.generation_recall, 4)
            result["generation_hit"] = round(self.retrieval.generation_hit, 4)
        # FIX DT-5: incluir candidatos pre-rerank para trazabilidad
        if self.retrieval.pre_rerank_candidate_ids:
            result["pre_rerank_candidate_ids"] = self.retrieval.pre_rerank_candidate_ids
        # FIX DT-7: incluir metadata (contiene reranked status, etc.)
        if self.metadata:
            result["metadata"] = self.metadata
        return result


# =============================================================================
# RESULTADO - NIVEL RUN (reemplaza GlobalEvaluationReport)
# =============================================================================

@dataclass
class EvaluationRun:
    """
    Resultado de UNA ejecucion: 1 dataset + 1 embedding + 1 estrategia.

    Reemplaza GlobalEvaluationReport y DatasetEvaluationResult.
    No hay results_matrix, no hay dimension multi-modelo.
    """
    # Identificacion del run
    run_id: str = ""
    dataset_name: str = ""
    embedding_model: str = ""
    retrieval_strategy: str = ""
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    # Contadores
    num_queries_total: int = 0
    num_queries_evaluated: int = 0
    num_queries_failed: int = 0
    total_documents: int = 0

    # Metricas de retrieval agregadas
    avg_hit_rate_at_5: float = 0.0
    avg_mrr: float = 0.0
    avg_recall_at_k: Dict[int, float] = field(default_factory=dict)
    avg_ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    retrieval_complement_recall_at_k: Dict[int, float] = field(
        default_factory=dict
    )

    # Metricas de generacion (opcionales, deuda tecnica LLM-as-judge)
    avg_generation_score: Optional[float] = None
    std_generation_score: Optional[float] = None

    # Diagnostico
    avg_retrieved_count: float = 0.0
    avg_expected_count: float = 0.0

    # Metricas de retrieval efectivo (post-rerank, solo con reranker activo)
    avg_generation_recall: Optional[float] = None
    avg_generation_hit: Optional[float] = None
    reranker_rescue_count: int = 0

    # Detalle por query
    query_results: List[QueryEvaluationResult] = field(default_factory=list)

    # Meta
    execution_time_seconds: float = 0.0
    timestamp: str = ""
    status: EvaluationStatus = EvaluationStatus.PENDING

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    # NOTE: calculate_aggregates() eliminado (v3.2 cleanup).
    # La agregacion se hace en MTEBEvaluator._build_run(), que es el unico
    # punto de construccion de EvaluationRun. No habia callers de este metodo.

    def to_dict(self) -> Dict[str, Any]:
        """Serializacion sin detalle de queries."""
        return {
            "run_id": self.run_id,
            "dataset_name": self.dataset_name,
            "embedding_model": self.embedding_model,
            "retrieval_strategy": self.retrieval_strategy,
            "config_snapshot": self.config_snapshot,
            "num_queries_evaluated": self.num_queries_evaluated,
            "num_queries_failed": self.num_queries_failed,
            "total_documents": self.total_documents,
            "avg_hit_rate_at_5": round(self.avg_hit_rate_at_5, 4),
            "avg_mrr": round(self.avg_mrr, 4),
            "avg_recall_at_k": {
                k: round(v, 4) for k, v in self.avg_recall_at_k.items()
            },
            "avg_ndcg_at_k": {
                k: round(v, 4) for k, v in self.avg_ndcg_at_k.items()
            },
            "retrieval_complement_recall_at_k": {
                k: round(v, 4)
                for k, v in self.retrieval_complement_recall_at_k.items()
            },
            "avg_retrieved_count": round(self.avg_retrieved_count, 1),
            "avg_expected_count": round(self.avg_expected_count, 1),
            "avg_generation_recall": (
                round(self.avg_generation_recall, 4)
                if self.avg_generation_recall is not None
                else None
            ),
            "avg_generation_hit": (
                round(self.avg_generation_hit, 4)
                if self.avg_generation_hit is not None
                else None
            ),
            "reranker_rescue_count": self.reranker_rescue_count,
            "avg_generation_score": (
                round(self.avg_generation_score, 4)
                if self.avg_generation_score is not None
                else None
            ),
            "execution_time_seconds": round(self.execution_time_seconds, 2),
            "timestamp": self.timestamp,
            "status": self.status.value,
        }

    def to_dict_full(self) -> Dict[str, Any]:
        """to_dict + query_results individuales."""
        d = self.to_dict()
        d["query_results"] = [qr.to_dict() for qr in self.query_results]
        return d


# =============================================================================
# CONFIGURACION DE DATASETS
# =============================================================================

DATASET_CONFIG: Dict[str, Dict[str, Any]] = {
    "hotpotqa": {
        "type": DatasetType.HYBRID,
        "primary_metric": MetricType.F1_SCORE,
        "secondary_metrics": [MetricType.EXACT_MATCH, MetricType.FAITHFULNESS],
        "has_supporting_facts": True,
        "answer_field": "answer",
        "description": "Preguntas multi-hop que requieren conectar multiples hechos",
    },
    # Datasets adicionales: agregar aqui cuando tengan ETL ejecutado y datos en MinIO.
    # get_dataset_config() devuelve defaults razonables para datasets no registrados.
}


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """Obtiene la configuracion para un dataset especifico."""
    normalized_name = dataset_name.lower().replace("-", "").replace("_", "")

    for key, config in DATASET_CONFIG.items():
        if key.replace("-", "").replace("_", "") == normalized_name:
            return config

    logger.warning(
        f"Dataset '{dataset_name}' sin configuracion predefinida. Usando defaults."
    )
    return {
        "type": DatasetType.RETRIEVAL_ONLY,
        "primary_metric": MetricType.FAITHFULNESS,
        "secondary_metrics": [MetricType.ANSWER_RELEVANCE],
        "has_supporting_facts": False,
        "answer_field": None,
        "description": "Dataset sin configuracion predefinida",
    }


# =============================================================================
# PROTOCOLOS
# =============================================================================

@runtime_checkable
class LLMJudgeProtocol(Protocol):
    """Protocolo para LLM que evalua metricas y genera contextos.
    
    FIX DT-4: max_tokens agregado para compatibilidad con
    contextual_retriever (que limita tokens de contexto generado).
    """

    def invoke(
        self, user_prompt: str, system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> str: ...

    async def invoke_async(
        self, user_prompt: str, system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> str: ...


@runtime_checkable
class EmbeddingModelProtocol(Protocol):
    """Protocolo para modelos de embedding (compatible con LangChain Embeddings).
    
    FIX DT-4: centralizado aqui (antes en metrics.py) para que retrieval/,
    vector_store.py y metrics.py importen del mismo lugar.
    """

    def embed_query(self, text: str) -> List[float]: ...

    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...


__all__ = [
    "DatasetType",
    "MetricType",
    "EvaluationStatus",
    "NormalizedQuery",
    "NormalizedDocument",
    "LoadedDataset",
    "QueryRetrievalDetail",
    "GenerationResult",
    "QueryEvaluationResult",
    "EvaluationRun",
    "DATASET_CONFIG",
    "get_dataset_config",
    "LLMJudgeProtocol",
    "EmbeddingModelProtocol",
]
