#!/usr/bin/env python3
"""
Entry point para sandbox MTEB.

Uso:
    python -m sandbox_mteb.run              # Run con config del .env
    python -m sandbox_mteb.run --dry-run    # Solo valida config
    python -m sandbox_mteb.run --env /path  # .env alternativo
    python -m sandbox_mteb.run -v           # Logging verbose (DEBUG)

La parametrizacion del run (queries, corpus, DEV_MODE, estrategia, etc.)
se controla exclusivamente via .env. Ver env.example.
"""

import argparse
import logging
import sys
from pathlib import Path

# Asegurar que el proyecto raiz esta en sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sandbox_mteb.config import MTEBConfig
from sandbox_mteb.evaluator import MTEBEvaluator
from shared.report import RunExporter
from shared.structured_logging import configure_logging


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    # DT-3: configure_logging lee LOG_FORMAT del entorno ("text" o "jsonl")
    configure_logging(level=level)
    # Reducir ruido de libs externas
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MTEB RAG Evaluation Sandbox"
    )
    # Default .env: buscar primero en el directorio del sandbox
    sandbox_dir = Path(__file__).resolve().parent
    default_env = str(sandbox_dir / ".env")

    parser.add_argument(
        "--env",
        default=default_env,
        help=f"Ruta al archivo .env (default: {default_env})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo validar config y mostrar resumen",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Logging verbose (DEBUG)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("sandbox_mteb")

    # 1. Construir config
    env_path = Path(args.env)
    if not env_path.exists():
        logger.error(f"Archivo .env no encontrado: {env_path.resolve()}")
        logger.error("Copiar .env.example a .env y completar valores.")
        return 1

    config = MTEBConfig.from_env(str(env_path))

    # 2. Validar
    errors = config.validate()
    if errors:
        logger.error("Errores de configuracion:")
        for e in errors:
            logger.error(f"  - {e}")
        return 1

    # 3. Mostrar resumen
    print(config.summary())
    config.ensure_directories()

    # 4. Dry run?
    if args.dry_run:
        print("\n[DRY RUN] Config valida. No se ejecuta evaluacion.")
        return 0

    # 5. Ejecutar
    evaluator = MTEBEvaluator(config)
    run_result = evaluator.run()

    # 6. Exportar
    exporter = RunExporter(output_dir=config.storage.evaluation_results_dir)
    paths = exporter.export(run_result)

    print(f"\nResultados exportados:")
    for kind, path in paths.items():
        print(f"  {kind}: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
