"""
Modulo: LLM & Embeddings Service
Descripcion: Servicios NIM para inferencia LLM y embeddings.

Ubicacion: shared/llm.py

Consolida llm.py + embeddings.py.

Fixes aplicados:
  - asyncio.Lock se crea en __post_init__ (no en default de dataclass)
  - invoke() sync usa run_sync() con asyncio.run()
  - Eliminados from_env() y from_settings() (el sandbox construye explicitamente)
  - load_embedding_model acepta solo parametros explicitos (sin fallback)
"""

import asyncio
import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from shared.types import EmbeddingModelProtocol

logger = logging.getLogger(__name__)

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    from langchain_core.messages import HumanMessage, SystemMessage
    HAS_NVIDIA = True
except ImportError:
    HAS_NVIDIA = False
    ChatNVIDIA = None
    HumanMessage = None
    SystemMessage = None


# =============================================================================
# METRICAS
# =============================================================================

@dataclass
class LLMMetrics:
    """Metricas de rendimiento del servicio LLM."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0
    retries_total: int = 0

    # asyncio.Lock() no requiere event loop activo para instanciarse (Python 3.10+)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def record_request(
        self, success: bool, latency_ms: float, retries: int = 0
    ) -> None:
        async with self._lock:
            self.total_requests += 1
            self.total_latency_ms += latency_ms
            self.retries_total += retries
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0
        return self.total_latency_ms / self.total_requests

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0
        return self.successful_requests / self.total_requests

    def summary(self) -> str:
        return (
            f"Requests: {self.total_requests} | "
            f"Success: {self.success_rate:.1%} | "
            f"Avg Latency: {self.avg_latency_ms:.0f}ms | "
            f"Retries: {self.retries_total}"
        )

    # FIX DTm-11: crear Lock nuevo en copias para evitar compartir
    # estado de sincronizacion entre instancias independientes.
    def __copy__(self) -> "LLMMetrics":
        return LLMMetrics(
            total_requests=self.total_requests,
            successful_requests=self.successful_requests,
            failed_requests=self.failed_requests,
            total_latency_ms=self.total_latency_ms,
            retries_total=self.retries_total,
        )

    def __deepcopy__(self, memo: dict) -> "LLMMetrics":
        return self.__copy__()


# =============================================================================
# HELPER: ejecutar coroutine de forma segura
# =============================================================================

from typing import Any, Coroutine, TypeVar

_T = TypeVar("_T")


def run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """
    Ejecuta una coroutine de forma sincrona.

    - Sin event loop activo (CLI normal): usa asyncio.run().
    - Con event loop activo (Jupyter, frameworks async): crea un thread
      dedicado con su propio loop para evitar el error
      "cannot be called from a running event loop".

    FIX DTm-3: detecta loop activo y ejecuta en thread auxiliar.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        # Caso normal (CLI): no hay loop, asyncio.run() funciona directamente.
        return asyncio.run(coro)

    # Caso con loop activo (Jupyter, frameworks async):
    # ejecutar la coroutine en un thread dedicado con su propio loop.
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


# =============================================================================
# SERVICIO LLM ASINCRONO
# =============================================================================

class AsyncLLMService:
    """
    Servicio asincrono para inferencia de texto con NVIDIA NIM.

    Uso:
        service = AsyncLLMService(
            base_url="http://nim:8080/v1",
            model_name="meta/llama-3.1-70b-instruct",
        )
        response = await service.invoke_async("prompt")
        response = service.invoke("prompt")  # sync wrapper
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        max_concurrent: int = 32,
        timeout_seconds: int = 120,
        max_retries: int = 3,
        temperature: float = 0.1,
    ):
        if not HAS_NVIDIA:
            raise ImportError("pip install langchain-nvidia-ai-endpoints")

        self.base_url = base_url
        self.model_name = model_name
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.temperature = temperature

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client = ChatNVIDIA(
            base_url=base_url,
            model=model_name,
            temperature=temperature,
        )
        self.metrics = LLMMetrics()

        logger.info(
            f"AsyncLLMService: {model_name} @ {base_url} "
            f"(max_concurrent={max_concurrent})"
        )

    async def __aenter__(self) -> "AsyncLLMService":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    # -------------------------------------------------------------------------
    # INFERENCIA
    # -------------------------------------------------------------------------

    async def _invoke_with_retry(
        self, messages: list, max_tokens: int = 4096
    ) -> str:
        last_error = None
        retries = 0

        for attempt in range(self.max_retries + 1):
            start_time = time.perf_counter()
            try:
                async with self._semaphore:
                    response = await self._client.ainvoke(
                        messages,
                        max_tokens=max_tokens,
                        temperature=self.temperature,
                    )

                latency_ms = (time.perf_counter() - start_time) * 1000
                await self.metrics.record_request(True, latency_ms, retries)

                content = response.content
                if isinstance(content, list):
                    content = "\n".join(
                        part.get("text", str(part))
                        if isinstance(part, dict)
                        else str(part)
                        for part in content
                    )

                # Reasoning models (e.g. nemotron-3-nano) may return
                # content in reasoning_content while leaving content empty.
                if not content or not str(content).strip():
                    extra = getattr(response, "additional_kwargs", {})
                    reasoning = (
                        extra.get("reasoning_content")
                        or extra.get("reasoning")
                    )
                    if reasoning and str(reasoning).strip():
                        logger.debug(
                            "Content empty, using reasoning_content as fallback"
                        )
                        content = str(reasoning)

                if not content or not str(content).strip():
                    raise ValueError("LLM returned empty/null content")
                return str(content).strip()

            except Exception as e:
                last_error = e
                retries += 1
                latency_ms = (time.perf_counter() - start_time) * 1000

                if attempt < self.max_retries:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Intento {attempt + 1}/{self.max_retries + 1} fallo: {e}. "
                        f"Reintentando en {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    await self.metrics.record_request(
                        False, latency_ms, retries
                    )

        raise RuntimeError(
            f"Inferencia fallida tras {self.max_retries + 1} intentos. "
            f"Ultimo error: {last_error}"
        )

    async def invoke_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> str:
        """API async principal."""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        return await self._invoke_with_retry(messages, max_tokens)

    def invoke(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> str:
        """Wrapper sincrono. Usa run_sync para compatibilidad con loops activos."""
        return run_sync(
            self.invoke_async(user_prompt, system_prompt, max_tokens)
        )

    def get_metrics_summary(self) -> str:
        return self.metrics.summary()

    def reset_metrics(self) -> None:
        self.metrics = LLMMetrics()


# =============================================================================
# EMBEDDINGS (consolidado desde embeddings.py)
# =============================================================================

try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
    HAS_NVIDIA_EMBEDDINGS = True
except ImportError:
    HAS_NVIDIA_EMBEDDINGS = False
    NVIDIAEmbeddings = None


def load_embedding_model(
    base_url: str,
    model_name: str,
    model_type: str = "symmetric",
) -> EmbeddingModelProtocol:
    """
    Carga modelo de embeddings NIM para uso con LangChain/ChromaDB.

    Acepta SOLO parametros explicitos. Sin fallback a config ni env vars.
    El caller (cada sandbox) pasa los valores desde su propio config.

    Args:
        base_url: URL base del servidor NIM.
        model_name: Nombre del modelo de embedding.
        model_type: "symmetric" (gRPC/Triton) o "asymmetric" (REST OpenAI-compatible).

    Returns:
        Instancia NVIDIAEmbeddings compatible con LangChain.
    """
    if not HAS_NVIDIA_EMBEDDINGS:
        raise ImportError("pip install langchain-nvidia-ai-endpoints")

    if not base_url or not model_name:
        raise ValueError(
            "base_url y model_name son requeridos. "
            "Verificar configuracion en .env del sandbox."
        )

    if model_type not in ("symmetric", "asymmetric"):
        raise ValueError(
            f"model_type='{model_type}' no valido. Usar 'symmetric' o 'asymmetric'"
        )

    logger.info(
        f"Cargando embedding NIM: {model_name} @ {base_url} [tipo={model_type}]"
    )

    if model_type == "asymmetric":
        return NVIDIAEmbeddings(  # type: ignore[no-any-return]
            model=model_name,
            base_url=base_url,
            truncate="END",
        )
    else:
        return NVIDIAEmbeddings(  # type: ignore[no-any-return]
            model=model_name,
            base_url=base_url,
            truncate="END",
            mode="nim",
        )


__all__ = [
    "AsyncLLMService",
    "LLMMetrics",
    "HAS_NVIDIA",
    "run_sync",
    "load_embedding_model",
    "HAS_NVIDIA_EMBEDDINGS",
]
