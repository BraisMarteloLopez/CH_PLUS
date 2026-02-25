"""
Modulo: Structured Logging
Descripcion: Logging JSONL estructurado para analisis post-hoc de runs.

Ubicacion: shared/structured_logging.py

FIX DT-3: proporciona logging estructurado (JSONL) ademas del logging
human-readable existente. Controlado por variable de entorno LOG_FORMAT.

Uso:
    from shared.structured_logging import configure_logging, structured_log

    configure_logging()  # Lee LOG_FORMAT del entorno

    structured_log("run_start", run_id="abc", dataset="hotpotqa")
    structured_log("query_result", query_id="q1", hit_at_5=1.0, mrr=0.5)
    structured_log("run_complete", run_id="abc", elapsed_s=120.5)

El JSONL se escribe a un archivo dedicado (<run_id>.jsonl en data/results/)
o a stderr, segun configuracion. El logging human-readable existente
(logging.info/warning) no se modifica.
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)

# Formato controlado por env var: "text" (default, human-readable) o "jsonl"
_LOG_FORMAT: str = "text"
_JSONL_FILE = None


class JSONLFormatter(logging.Formatter):
    """Formatter que emite registros como lineas JSON."""

    def format(self, record: logging.LogRecord) -> str:
        entry: Dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def configure_logging(
    log_format: Optional[str] = None,
    level: int = logging.INFO,
) -> None:
    """
    Configura el logging del proceso.

    Args:
        log_format: "text" (default) o "jsonl". Si None, lee LOG_FORMAT del entorno.
        level: Nivel de logging (default INFO).
    """
    global _LOG_FORMAT

    fmt = log_format or os.environ.get("LOG_FORMAT", "text").lower()
    _LOG_FORMAT = fmt

    root = logging.getLogger()
    root.setLevel(level)

    # Limpiar handlers existentes para evitar duplicados en re-configuracion
    root.handlers.clear()

    if fmt == "jsonl":
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JSONLFormatter())
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )

    root.addHandler(handler)


def structured_log(
    event: str,
    **kwargs: Any,
) -> None:
    """
    Emite un evento estructurado al log.

    En modo JSONL: emite una linea JSON con el evento y todos los kwargs.
    En modo text: emite un logging.info con formato legible.

    Eventos estandar:
        - run_start: inicio de un run (run_id, dataset, strategy, ...)
        - embedding_batch: progreso de pre-embed (batch_n, total, elapsed_s)
        - query_result: resultado de una query (query_id, hit_at_5, mrr, ...)
        - run_complete: fin de un run (run_id, elapsed_s, queries_evaluated)
    """
    entry: Dict[str, Any] = {
        "event": event,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **kwargs,
    }

    if _LOG_FORMAT == "jsonl":
        # Emitir JSON puro a stderr (capturado por el handler JSONL)
        logger.info(json.dumps(entry, ensure_ascii=False, default=str))
    else:
        # Emitir formato legible
        detail = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info(f"[{event}] {detail}")


__all__ = [
    "configure_logging",
    "structured_log",
    "JSONLFormatter",
]
