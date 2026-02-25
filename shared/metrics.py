"""
Modulo: Evaluation Metrics
Descripcion: Metricas para evaluacion RAG.

Ubicacion: shared/metrics.py

Dos categorias:
  A) Con referencia: ExactMatch, F1, Accuracy, SemanticSimilarity
  B) Sin referencia (LLM-Judge): Faithfulness, AnswerRelevance, ContextUtilization

Todas retornan valores en [0.0, 1.0] via MetricResult.
"""

from __future__ import annotations

import logging
import re
import string
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

# Importacion condicional de numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

# Interfaces locales
from shared.types import MetricType, LLMJudgeProtocol, EmbeddingModelProtocol

# Limite de caracteres de la respuesta generada que se envia al LLM Judge.
# Guarda defensiva contra respuestas degeneradas (repeticiones, text overflow).
# Para HotpotQA las respuestas reales rara vez superan 200 chars, asi que
# 2000 chars (~500 tokens) es holgado sin desperdiciar contexto del judge.
_MAX_RESPONSE_CHARS_FOR_JUDGE = 2000

# Configuracion del logger
logger = logging.getLogger(__name__)


# =============================================================================
# SECCION 1: PROTOCOLOS Y TIPOS
# =============================================================================

# FIX DT-4: EmbeddingModelProtocol centralizado en types.py


@dataclass
class MetricResult:
    """Resultado de una evaluacion de metrica. Valor en [0.0, 1.0]."""
    metric_type: MetricType
    value: float
    details: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
    
    def __post_init__(self) -> None:
        if self.details is None:
            self.details = {}
        # Asegurar que el valor esta en rango valido
        self.value = max(0.0, min(1.0, self.value))
    
    def is_valid(self) -> bool:
        """Indica si el resultado es valido (sin errores)."""
        return self.error is None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "metric_type": self.metric_type.value,
            "value": round(self.value, 4),
            "details": self.details,
            "confidence": round(self.confidence, 4) if self.confidence else None,
            "error": self.error
        }


# =============================================================================
# SECCION 2: UTILIDADES DE NORMALIZACION DE TEXTO
# =============================================================================

class TextNormalizer:
    """Normalizacion de texto para comparacion de metricas."""
    
    # Articulos a eliminar
    ARTICLES_EN = {'a', 'an', 'the'}
    ARTICLES_ES = {'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas'}
    
    @staticmethod
    def normalize(
        text: str,
        lowercase: bool = True,
        remove_accents: bool = True,
        remove_punctuation: bool = True,
        remove_articles: bool = False,
        language: str = "en"
    ) -> str:
        """Aplica normalizacion completa: lowercase, acentos, puntuacion, espacios, articulos."""
        if not text:
            return ""
        
        result = text
        
        # 1. Minusculas
        if lowercase:
            result = result.lower()
        
        # 2. Eliminar acentos
        if remove_accents:
            result = TextNormalizer._remove_accents(result)
        
        # 3. Separar por dashes antes de eliminar puntuacion.
        #    Sin esto, "1969-1974" (en-dash) queda como 1 token,
        #    y "1969-1974" (hyphen) se fusiona en "19691974".
        #    Ambos causan F1=0.0 contra "1969 until 1974".
        result = TextNormalizer._dashes_to_spaces(result)
        
        # 4. Eliminar puntuacion
        if remove_punctuation:
            result = TextNormalizer._remove_punctuation(result)
        
        # 4. Normalizar espacios
        result = ' '.join(result.split())
        
        # 5. Eliminar articulos
        if remove_articles:
            articles = (
                TextNormalizer.ARTICLES_EN if language == "en" 
                else TextNormalizer.ARTICLES_ES
            )
            tokens = result.split()
            tokens = [t for t in tokens if t not in articles]
            result = ' '.join(tokens)
        
        return result.strip()
    
    @staticmethod
    def _remove_accents(text: str) -> str:
        """Elimina acentos y diacriticos via normalizacion Unicode NFD."""
        # NFD descompone caracteres acentuados en base + acento
        # Luego filtramos los caracteres de categoria "Mark, Nonspacing"
        normalized = unicodedata.normalize('NFD', text)
        return ''.join(
            char for char in normalized 
            if unicodedata.category(char) != 'Mn'
        )
    
    @staticmethod
    def _dashes_to_spaces(text: str) -> str:
        """Reemplaza variantes de dash/guion por espacios para evitar fusion de tokens."""
        return re.sub(r'[\u002D\u2013\u2014\u2015\u2212\uFE58\uFE63\uFF0D]', ' ', text)
    
    @staticmethod
    def _remove_punctuation(text: str) -> str:
        """Elimina signos de puntuacion."""
        # Crear tabla de traduccion que mapea puntuacion a None
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    
    @staticmethod
    def tokenize(text: str, normalize: bool = True) -> List[str]:
        """Tokeniza texto en palabras. Si normalize=True, aplica normalizacion previa."""
        if normalize:
            text = TextNormalizer.normalize(text)
        
        return text.split()
    
    @staticmethod
    def get_token_counts(text: str, normalize: bool = True) -> Counter:
        """Retorna Counter con frecuencia de tokens."""
        tokens = TextNormalizer.tokenize(text, normalize)
        return Counter(tokens)


# =============================================================================
# SECCION 3: METRICAS CON REFERENCIA (Reference-Based)
# =============================================================================

class ReferenceBasedMetrics:
    """Metricas que requieren ground truth textual."""
    
    @staticmethod
    def exact_match(
        generated: str,
        expected: str,
        normalize: bool = True
    ) -> MetricResult:
        """Coincidencia exacta normalizada. Retorna 0.0 o 1.0."""
        if not generated or not expected:
            return MetricResult(
                metric_type=MetricType.EXACT_MATCH,
                value=0.0,
                details={"reason": "empty_input"}
            )
        
        if normalize:
            gen_normalized = TextNormalizer.normalize(generated)
            exp_normalized = TextNormalizer.normalize(expected)
        else:
            gen_normalized = generated
            exp_normalized = expected
        
        is_match = gen_normalized == exp_normalized
        
        return MetricResult(
            metric_type=MetricType.EXACT_MATCH,
            value=1.0 if is_match else 0.0,
            details={
                "generated_normalized": gen_normalized[:100],
                "expected_normalized": exp_normalized[:100],
                "is_match": is_match
            }
        )
    
    @staticmethod
    def f1_score(
        generated: str,
        expected: str,
        normalize: bool = True
    ) -> MetricResult:
        """F1 por token-overlap (media armonica de precision y recall)."""
        if not generated or not expected:
            return MetricResult(
                metric_type=MetricType.F1_SCORE,
                value=0.0,
                details={"reason": "empty_input", "precision": 0.0, "recall": 0.0}
            )
        
        # Tokenizar
        gen_tokens = TextNormalizer.get_token_counts(generated, normalize)
        exp_tokens = TextNormalizer.get_token_counts(expected, normalize)
        
        if not gen_tokens or not exp_tokens:
            return MetricResult(
                metric_type=MetricType.F1_SCORE,
                value=0.0,
                details={"reason": "no_tokens", "precision": 0.0, "recall": 0.0}
            )
        
        # Calcular tokens comunes (interseccion con frecuencias)
        common_tokens = gen_tokens & exp_tokens  # Interseccion de Counters
        num_common = sum(common_tokens.values())
        
        num_generated = sum(gen_tokens.values())
        num_expected = sum(exp_tokens.values())
        
        # Calcular precision y recall
        precision = num_common / num_generated if num_generated > 0 else 0.0
        recall = num_common / num_expected if num_expected > 0 else 0.0
        
        # Calcular F1
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return MetricResult(
            metric_type=MetricType.F1_SCORE,
            value=f1,
            details={
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "num_common_tokens": num_common,
                "num_generated_tokens": num_generated,
                "num_expected_tokens": num_expected,
                "common_tokens": list(common_tokens.keys())[:10]  # Muestra primeros 10
            }
        )
    
    @staticmethod
    def accuracy(
        generated: str,
        expected: str,
        valid_labels: Optional[List[str]] = None,
        normalize: bool = True
    ) -> MetricResult:
        """Accuracy para clasificacion (ej: FEVER). Retorna 0.0 o 1.0."""
        if not generated or not expected:
            return MetricResult(
                metric_type=MetricType.ACCURACY,
                value=0.0,
                details={"reason": "empty_input"}
            )
        
        if normalize:
            gen_normalized = TextNormalizer.normalize(generated)
            exp_normalized = TextNormalizer.normalize(expected)
        else:
            gen_normalized = generated
            exp_normalized = expected
        
        # Validar si la etiqueta generada es valida
        is_valid_label = True
        if valid_labels:
            valid_normalized = [
                TextNormalizer.normalize(l) if normalize else l 
                for l in valid_labels
            ]
            is_valid_label = gen_normalized in valid_normalized
        
        is_correct = gen_normalized == exp_normalized
        
        return MetricResult(
            metric_type=MetricType.ACCURACY,
            value=1.0 if is_correct else 0.0,
            details={
                "generated_label": gen_normalized,
                "expected_label": exp_normalized,
                "is_correct": is_correct,
                "is_valid_label": is_valid_label
            }
        )
    
    @staticmethod
    def semantic_similarity(
        generated: str,
        expected: str,
        embedding_model: EmbeddingModelProtocol
    ) -> MetricResult:
        """Similitud coseno entre embeddings. Rango transformado de [-1,1] a [0,1]."""
        if not generated or not expected:
            return MetricResult(
                metric_type=MetricType.SEMANTIC_SIMILARITY,
                value=0.0,
                details={"reason": "empty_input"}
            )
        
        if not HAS_NUMPY:
            return MetricResult(
                metric_type=MetricType.SEMANTIC_SIMILARITY,
                value=0.0,
                error="numpy no esta instalado, requerido para similitud semantica"
            )
        
        try:
            # Generar embeddings
            gen_embedding = embedding_model.embed_query(generated)
            exp_embedding = embedding_model.embed_query(expected)
            
            # Convertir a arrays numpy
            gen_vec = np.array(gen_embedding)
            exp_vec = np.array(exp_embedding)
            
            # Calcular similitud coseno
            dot_product = np.dot(gen_vec, exp_vec)
            norm_gen = np.linalg.norm(gen_vec)
            norm_exp = np.linalg.norm(exp_vec)
            
            if norm_gen == 0 or norm_exp == 0:
                cosine_sim = 0.0
            else:
                cosine_sim = dot_product / (norm_gen * norm_exp)
            
            # Transformar de [-1, 1] a [0, 1]
            # (aunque en la practica textos reales raramente tienen similitud negativa)
            normalized_sim = (cosine_sim + 1) / 2
            
            return MetricResult(
                metric_type=MetricType.SEMANTIC_SIMILARITY,
                value=float(normalized_sim),
                details={
                    "raw_cosine_similarity": float(cosine_sim),
                    "embedding_dimension": len(gen_embedding),
                    "generated_preview": generated[:100],
                    "expected_preview": expected[:100]
                }
            )
            
        except Exception as e:
            logger.error(f"Error calculando similitud semantica: {e}")
            return MetricResult(
                metric_type=MetricType.SEMANTIC_SIMILARITY,
                value=0.0,
                error=str(e)
            )


# =============================================================================
# SECCION 4: METRICAS SIN REFERENCIA (LLM-Judge)
# =============================================================================

class LLMJudgeMetrics:
    """Metricas evaluadas por LLM como juez (sin ground truth textual)."""
    
    # =========================================================================
    # PROMPTS DEL SISTEMA PARA CADA METRICA
    # =========================================================================
    
    FAITHFULNESS_SYSTEM_PROMPT = """You are an expert RAG system evaluator.
Your task is to assess whether an ANSWER is derived EXCLUSIVELY from the
information present in the provided CONTEXT.

EVALUATION CRITERIA:
- SCORE 1.0: The answer uses ONLY information from the context. No fabrications.
- SCORE 0.7-0.9: The answer is mostly faithful, with minimal reasonable inferences.
- SCORE 0.4-0.6: The answer mixes context information with external knowledge.
- SCORE 0.1-0.3: The answer contains information not present in the context.
- SCORE 0.0: The answer contradicts the context or is entirely fabricated.

INSTRUCTIONS:
1. Read the CONTEXT carefully.
2. Compare each claim in the ANSWER against the CONTEXT.
3. Identify any information NOT present in the CONTEXT.
4. Assign a SCORE according to the criteria.

RESPONSE FORMAT (MANDATORY):
Respond ONLY with valid JSON:
{"score": <number between 0.0 and 1.0>, "justification": "<brief explanation>"}
"""

    ANSWER_RELEVANCE_SYSTEM_PROMPT = """You are an expert RAG system evaluator.
Your task is to assess whether an ANSWER is RELEVANT and PERTINENT to the QUESTION asked.

EVALUATION CRITERIA:
- SCORE 1.0: The answer directly addresses the question with useful information.
- SCORE 0.7-0.9: The answer is relevant but could be more direct or complete.
- SCORE 0.4-0.6: The answer is partially relevant, digresses, or is incomplete.
- SCORE 0.1-0.3: The answer barely relates to the question.
- SCORE 0.0: The answer is completely irrelevant or does not address the question.

INSTRUCTIONS:
1. Read the QUESTION and understand what information is being requested.
2. Assess whether the ANSWER provides that information.
3. Consider the completeness and usefulness of the answer.

RESPONSE FORMAT (MANDATORY):
Respond ONLY with valid JSON:
{"score": <number between 0.0 and 1.0>, "justification": "<brief explanation>"}
"""

    CONTEXT_UTILIZATION_SYSTEM_PROMPT = """You are an expert RAG system evaluator.
Your task is to assess how much of the provided CONTEXT was UTILIZED to generate the ANSWER.

EVALUATION CRITERIA:
- SCORE 1.0: The answer synthesizes and uses most of the relevant context.
- SCORE 0.7-0.9: The answer uses a good portion of the available context.
- SCORE 0.4-0.6: The answer uses only a fraction of the context.
- SCORE 0.1-0.3: The answer barely references the context.
- SCORE 0.0: The answer completely ignores the provided context.

NOTE: Do not penalize if the context contains information irrelevant to the question.
Only evaluate the use of RELEVANT context.

RESPONSE FORMAT (MANDATORY):
Respond ONLY with valid JSON:
{"score": <number between 0.0 and 1.0>, "justification": "<brief explanation>"}
"""

    # =========================================================================
    # METODOS DE EVALUACION
    # =========================================================================
    
    @staticmethod
    def faithfulness(
        generated: str,
        context: str,
        llm_judge: LLMJudgeProtocol
    ) -> MetricResult:
        """Evalua si la respuesta se deriva exclusivamente del contexto."""
        if not generated:
            return MetricResult(
                metric_type=MetricType.FAITHFULNESS,
                value=0.0,
                details={"reason": "empty_response"}
            )
        
        if not context:
            return MetricResult(
                metric_type=MetricType.FAITHFULNESS,
                value=0.0,
                details={"reason": "empty_context"}
            )
        
        # Construir prompt de evaluacion
        # FIX DT-6: no re-truncar contexto. El caller (evaluator) ya trunco
        # via _format_context() al limite del modelo. Truncar aqui a 4000 chars
        # causaba asimetria: el LLM de generacion veia el contexto completo
        # pero el judge solo los primeros 2-3 documentos.
        user_prompt = f"""CONTEXT:
{context}

ANSWER TO EVALUATE:
{generated[:_MAX_RESPONSE_CHARS_FOR_JUDGE]}

Evaluate the faithfulness of the ANSWER with respect to the CONTEXT."""
        
        return LLMJudgeMetrics._invoke_judge(
            llm_judge=llm_judge,
            system_prompt=LLMJudgeMetrics.FAITHFULNESS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            metric_type=MetricType.FAITHFULNESS
        )
    
    @staticmethod
    def answer_relevance(
        generated: str,
        query: str,
        llm_judge: LLMJudgeProtocol
    ) -> MetricResult:
        """Evalua si la respuesta es pertinente a la pregunta."""
        if not generated:
            return MetricResult(
                metric_type=MetricType.ANSWER_RELEVANCE,
                value=0.0,
                details={"reason": "empty_response"}
            )
        
        if not query:
            return MetricResult(
                metric_type=MetricType.ANSWER_RELEVANCE,
                value=0.0,
                details={"reason": "empty_query"}
            )
        
        user_prompt = f"""QUESTION:
{query}

ANSWER TO EVALUATE:
{generated[:_MAX_RESPONSE_CHARS_FOR_JUDGE]}

Evaluate the relevance of the ANSWER with respect to the QUESTION."""
        
        return LLMJudgeMetrics._invoke_judge(
            llm_judge=llm_judge,
            system_prompt=LLMJudgeMetrics.ANSWER_RELEVANCE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            metric_type=MetricType.ANSWER_RELEVANCE
        )
    
    @staticmethod
    def context_utilization(
        generated: str,
        context: str,
        query: str,
        llm_judge: LLMJudgeProtocol
    ) -> MetricResult:
        """Evalua que proporcion del contexto relevante se uso en la respuesta."""
        if not generated or not context:
            return MetricResult(
                metric_type=MetricType.CONTEXT_UTILIZATION,
                value=0.0,
                details={"reason": "empty_input"}
            )
        
        user_prompt = f"""ORIGINAL QUESTION:
{query}

PROVIDED CONTEXT:
{context}

GENERATED ANSWER:
{generated[:_MAX_RESPONSE_CHARS_FOR_JUDGE]}

Evaluate how much of the relevant CONTEXT was utilized in the ANSWER."""
        
        return LLMJudgeMetrics._invoke_judge(
            llm_judge=llm_judge,
            system_prompt=LLMJudgeMetrics.CONTEXT_UTILIZATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            metric_type=MetricType.CONTEXT_UTILIZATION
        )
    
    # =========================================================================
    # VERSIONES ASYNC (usan invoke_async del LLM)
    # =========================================================================

    @staticmethod
    async def faithfulness_async(
        generated: str, context: str, llm_judge: LLMJudgeProtocol
    ) -> MetricResult:
        """Version async de faithfulness."""
        if not generated:
            return MetricResult(metric_type=MetricType.FAITHFULNESS, value=0.0, details={"reason": "empty_response"})
        if not context:
            return MetricResult(metric_type=MetricType.FAITHFULNESS, value=0.0, details={"reason": "empty_context"})
        
        user_prompt = f"""CONTEXT:\n{context}\n\nANSWER TO EVALUATE:\n{generated[:_MAX_RESPONSE_CHARS_FOR_JUDGE]}\n\nEvaluate the faithfulness of the ANSWER with respect to the CONTEXT."""
        return await LLMJudgeMetrics._invoke_judge_async(
            llm_judge=llm_judge,
            system_prompt=LLMJudgeMetrics.FAITHFULNESS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            metric_type=MetricType.FAITHFULNESS,
        )

    @staticmethod
    async def answer_relevance_async(
        generated: str, query: str, llm_judge: LLMJudgeProtocol
    ) -> MetricResult:
        """Version async de answer_relevance."""
        if not generated:
            return MetricResult(metric_type=MetricType.ANSWER_RELEVANCE, value=0.0, details={"reason": "empty_response"})
        if not query:
            return MetricResult(metric_type=MetricType.ANSWER_RELEVANCE, value=0.0, details={"reason": "empty_query"})
        
        user_prompt = f"""QUESTION:\n{query}\n\nANSWER TO EVALUATE:\n{generated[:_MAX_RESPONSE_CHARS_FOR_JUDGE]}\n\nEvaluate the relevance of the ANSWER with respect to the QUESTION."""
        return await LLMJudgeMetrics._invoke_judge_async(
            llm_judge=llm_judge,
            system_prompt=LLMJudgeMetrics.ANSWER_RELEVANCE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            metric_type=MetricType.ANSWER_RELEVANCE,
        )

    @staticmethod
    async def context_utilization_async(
        generated: str, context: str, query: str, llm_judge: LLMJudgeProtocol
    ) -> MetricResult:
        """Version async de context_utilization."""
        if not generated or not context:
            return MetricResult(metric_type=MetricType.CONTEXT_UTILIZATION, value=0.0, details={"reason": "empty_input"})
        
        user_prompt = f"""ORIGINAL QUESTION:\n{query}\n\nPROVIDED CONTEXT:\n{context}\n\nGENERATED ANSWER:\n{generated[:_MAX_RESPONSE_CHARS_FOR_JUDGE]}\n\nEvaluate how much of the relevant CONTEXT was utilized in the ANSWER."""
        return await LLMJudgeMetrics._invoke_judge_async(
            llm_judge=llm_judge,
            system_prompt=LLMJudgeMetrics.CONTEXT_UTILIZATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            metric_type=MetricType.CONTEXT_UTILIZATION,
        )
    
    # =========================================================================
    # METODOS AUXILIARES
    # =========================================================================
    
    @staticmethod
    def _invoke_judge(
        llm_judge: LLMJudgeProtocol,
        system_prompt: str,
        user_prompt: str,
        metric_type: MetricType
    ) -> MetricResult:
        """Invoca LLM Judge sync y parsea respuesta JSON."""
        try:
            # Invocar LLM
            response = llm_judge.invoke(user_prompt, system_prompt=system_prompt)
            response_text = str(response).strip()
            return LLMJudgeMetrics._parse_judge_result(response_text, metric_type)
                
        except Exception as e:
            logger.error(f"Error en LLM Judge ({metric_type.value}): {e}")
            return MetricResult(
                metric_type=metric_type,
                value=0.0,
                error=str(e),
                confidence=0.0
            )

    @staticmethod
    async def _invoke_judge_async(
        llm_judge: LLMJudgeProtocol,
        system_prompt: str,
        user_prompt: str,
        metric_type: MetricType
    ) -> MetricResult:
        """Version async de _invoke_judge. Usa invoke_async del LLM."""
        try:
            response = await llm_judge.invoke_async(
                user_prompt, system_prompt=system_prompt
            )
            response_text = str(response).strip()
            return LLMJudgeMetrics._parse_judge_result(response_text, metric_type)
                
        except Exception as e:
            logger.error(f"Error en LLM Judge async ({metric_type.value}): {e}")
            return MetricResult(
                metric_type=metric_type,
                value=0.0,
                error=str(e),
                confidence=0.0
            )

    @staticmethod
    def _parse_judge_result(response_text: str, metric_type: MetricType) -> MetricResult:
        """Parsea respuesta del judge (compartido entre sync y async)."""
        parsed = LLMJudgeMetrics._parse_judge_response(response_text)
        
        if parsed:
            score = float(parsed.get("score", 0.0))
            justification = parsed.get("justification", "Sin justificacion")
            confidence = 0.9 if "score" in parsed and "justification" in parsed else 0.6
            
            return MetricResult(
                metric_type=metric_type,
                value=score,
                details={
                    "justification": justification,
                    "raw_response": response_text[:500]
                },
                confidence=confidence
            )
        else:
            score = LLMJudgeMetrics._extract_score_fallback(response_text)
            
            return MetricResult(
                metric_type=metric_type,
                value=score,
                details={
                    "justification": "Formato no estructurado",
                    "raw_response": response_text[:500]
                },
                confidence=0.4
            )
    
    @staticmethod
    def _parse_judge_response(response_text: str) -> Optional[Dict[str, Any]]:
        """Parsea respuesta JSON del LLM Judge. Maneja JSON embebido en texto."""
        import json
        
        # Intentar parseo directo
        try:
            result: Dict[str, Any] = json.loads(response_text)
            return result
        except json.JSONDecodeError:
            pass

        # Buscar JSON embebido con regex
        json_pattern = r'\{[^{}]*"score"[^{}]*\}'
        match = re.search(json_pattern, response_text, re.DOTALL)

        if match:
            try:
                result = json.loads(match.group())
                return result
            except json.JSONDecodeError:
                pass
        
        return None
    
    @staticmethod
    def _extract_score_fallback(response_text: str) -> float:
        """Extrae score numerico de respuesta no estructurada. Busca decimales y fracciones.
        
        FIX DT-9: Regex anterior ``(?:score[:\\s]*)?([01]\\.?\\d*)`` capturaba parciales
        de numeros mayores (ej: '10.5' -> '10' -> 1.0 falso positivo).
        Ahora usa word boundaries y patrones explicitos, en orden:
          1. Fraccion N/M (mas especifico, evita que "1/2" sea capturado como decimal "1")
          2. Decimal 0.X o 1.0 (rango 0-1 directo)
          3. Entero 1-10 con prefijo "score:" (normalizado a 0-1)
        """
        text = response_text.lower()
        
        # 1. Fracciones como 8/10 (evaluar primero: "1/2" no debe capturarse como "1")
        fraction_pattern = r'(\d+)\s*/\s*(\d+)'
        match = re.search(fraction_pattern, response_text)
        
        if match:
            try:
                numerator = float(match.group(1))
                denominator = float(match.group(2))
                if denominator > 0:
                    return min(1.0, numerator / denominator)
            except ValueError:
                pass
        
        # 2. Decimal explicito en rango 0-1: "0.85", "1.0", "0", "1"
        #    \b evita capturar "10" cuando es parte de un numero mayor
        decimal_pattern = r'(?:score[:\s]*)?\b(0(?:\.\d+)?|1(?:\.0+)?)\b'
        match = re.search(decimal_pattern, text)
        
        if match:
            try:
                value = float(match.group(1))
                if 0 <= value <= 1:
                    return value
            except ValueError:
                pass
        
        # 3. Entero standalone 2-10 (escala 1-10): "score: 8", "score:7"
        #    Solo con prefijo "score" para evitar falsos positivos
        int_scale_pattern = r'(?:score[:\s]*)\b(\d{1,2})\b'
        match = re.search(int_scale_pattern, text)
        
        if match:
            try:
                value = int(match.group(1))
                if 1 <= value <= 10:
                    return value / 10
            except ValueError:
                pass
        
        # Default si no se puede extraer
        return 0.5



# =============================================================================
# SECCION 5: CLASE ORQUESTADORA DE METRICAS
# =============================================================================

class MetricsCalculator:
    """Orquestador de metricas segun tipo de dataset y disponibilidad de ground truth."""
    
    def __init__(
        self,
        llm_judge: Optional[LLMJudgeProtocol] = None,
        embedding_model: Optional[EmbeddingModelProtocol] = None
    ):
        self.llm_judge = llm_judge
        self.embedding_model = embedding_model
        
        logger.debug(
            f"MetricsCalculator inicializado. "
            f"LLM Judge: {'OK' if llm_judge else 'No'}, "
            f"Embeddings: {'OK' if embedding_model else 'No'}"
        )
    
    def calculate(
        self,
        metric_type: MetricType,
        generated: str,
        expected: Optional[str] = None,
        context: Optional[str] = None,
        query: Optional[str] = None
    ) -> MetricResult:
        """Calcula una metrica. Raises ValueError si faltan argumentos requeridos."""
        # Metricas con referencia
        if metric_type == MetricType.EXACT_MATCH:
            if expected is None:
                raise ValueError("EXACT_MATCH requiere 'expected'")
            return ReferenceBasedMetrics.exact_match(generated, expected)
        
        elif metric_type == MetricType.F1_SCORE:
            if expected is None:
                raise ValueError("F1_SCORE requiere 'expected'")
            return ReferenceBasedMetrics.f1_score(generated, expected)
        
        elif metric_type == MetricType.ACCURACY:
            if expected is None:
                raise ValueError("ACCURACY requiere 'expected'")
            return ReferenceBasedMetrics.accuracy(generated, expected)
        
        elif metric_type == MetricType.SEMANTIC_SIMILARITY:
            if expected is None:
                raise ValueError("SEMANTIC_SIMILARITY requiere 'expected'")
            if self.embedding_model is None:
                raise ValueError("SEMANTIC_SIMILARITY requiere embedding_model configurado")
            return ReferenceBasedMetrics.semantic_similarity(
                generated, expected, self.embedding_model
            )
        
        # Metricas sin referencia (LLM Judge)
        elif metric_type == MetricType.FAITHFULNESS:
            if context is None:
                raise ValueError("FAITHFULNESS requiere 'context'")
            if self.llm_judge is None:
                raise ValueError("FAITHFULNESS requiere llm_judge configurado")
            return LLMJudgeMetrics.faithfulness(generated, context, self.llm_judge)
        
        elif metric_type == MetricType.ANSWER_RELEVANCE:
            if query is None:
                raise ValueError("ANSWER_RELEVANCE requiere 'query'")
            if self.llm_judge is None:
                raise ValueError("ANSWER_RELEVANCE requiere llm_judge configurado")
            return LLMJudgeMetrics.answer_relevance(generated, query, self.llm_judge)
        
        elif metric_type == MetricType.CONTEXT_UTILIZATION:
            if context is None or query is None:
                raise ValueError("CONTEXT_UTILIZATION requiere 'context' y 'query'")
            if self.llm_judge is None:
                raise ValueError("CONTEXT_UTILIZATION requiere llm_judge configurado")
            return LLMJudgeMetrics.context_utilization(
                generated, context, query, self.llm_judge
            )
        
        else:
            raise ValueError(f"Tipo de metrica no soportado: {metric_type}")
    
    def calculate_all(
        self,
        generated: str,
        metric_types: List[MetricType],
        expected: Optional[str] = None,
        context: Optional[str] = None,
        query: Optional[str] = None
    ) -> Dict[MetricType, MetricResult]:
        """Calcula multiples metricas. Retorna Dict[MetricType, MetricResult]."""
        results = {}
        
        for metric_type in metric_types:
            try:
                result = self.calculate(
                    metric_type=metric_type,
                    generated=generated,
                    expected=expected,
                    context=context,
                    query=query
                )
                results[metric_type] = result
                
            except ValueError as e:
                # Metrica no calculable por falta de datos
                logger.warning(f"No se pudo calcular {metric_type.value}: {e}")
                results[metric_type] = MetricResult(
                    metric_type=metric_type,
                    value=0.0,
                    error=str(e)
                )
            except Exception as e:
                logger.error(f"Error inesperado calculando {metric_type.value}: {e}")
                results[metric_type] = MetricResult(
                    metric_type=metric_type,
                    value=0.0,
                    error=str(e)
                )
        
        return results
    
    def get_available_metrics(self) -> Dict[str, bool]:
        """Retorna disponibilidad de cada metrica segun dependencias configuradas."""
        return {
            "exact_match": True,
            "f1_score": True,
            "accuracy": True,
            "semantic_similarity": self.embedding_model is not None,
            "faithfulness": self.llm_judge is not None,
            "answer_relevance": self.llm_judge is not None,
            "context_utilization": self.llm_judge is not None
        }

    async def calculate_async(
        self,
        metric_type: MetricType,
        generated: str,
        expected: Optional[str] = None,
        context: Optional[str] = None,
        query: Optional[str] = None,
    ) -> MetricResult:
        """Version async de calculate(). Metricas con referencia son sync; LLM-judge usa invoke_async."""
        # Metricas con referencia: compute sync (instantaneo)
        if metric_type in (
            MetricType.EXACT_MATCH,
            MetricType.F1_SCORE,
            MetricType.ACCURACY,
            MetricType.SEMANTIC_SIMILARITY,
        ):
            return self.calculate(metric_type, generated, expected, context, query)

        # Metricas LLM Judge: compute async
        if self.llm_judge is None:
            raise ValueError(f"{metric_type.value} requiere llm_judge configurado")

        if metric_type == MetricType.FAITHFULNESS:
            if context is None:
                raise ValueError("FAITHFULNESS requiere 'context'")
            return await LLMJudgeMetrics.faithfulness_async(
                generated, context, self.llm_judge
            )

        elif metric_type == MetricType.ANSWER_RELEVANCE:
            if query is None:
                raise ValueError("ANSWER_RELEVANCE requiere 'query'")
            return await LLMJudgeMetrics.answer_relevance_async(
                generated, query, self.llm_judge
            )

        elif metric_type == MetricType.CONTEXT_UTILIZATION:
            if context is None or query is None:
                raise ValueError("CONTEXT_UTILIZATION requiere 'context' y 'query'")
            return await LLMJudgeMetrics.context_utilization_async(
                generated, context, query, self.llm_judge
            )

        else:
            raise ValueError(f"Tipo de metrica no soportado: {metric_type}")


# =============================================================================
# SECCION 6: FUNCIONES DE CONVENIENCIA
# =============================================================================

def quick_f1(generated: str, expected: str) -> float:
    """Calcula F1 Score. Retorna solo el valor."""
    result = ReferenceBasedMetrics.f1_score(generated, expected)
    return result.value


def quick_exact_match(generated: str, expected: str) -> float:
    """Calcula Exact Match. Retorna solo el valor."""
    result = ReferenceBasedMetrics.exact_match(generated, expected)
    return result.value


def quick_semantic_similarity(
    generated: str,
    expected: str,
    embedding_model: EmbeddingModelProtocol
) -> float:
    """Calcula Semantic Similarity. Retorna solo el valor."""
    result = ReferenceBasedMetrics.semantic_similarity(generated, expected, embedding_model)
    return result.value



