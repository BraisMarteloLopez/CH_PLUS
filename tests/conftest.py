"""
Configuracion compartida de pytest para todos los tests.

FIX DTm-4: centraliza sys.path y mocking de modulos de infraestructura
que no estan disponibles en el entorno de test (boto3, langchain_*, chromadb).

Con pyproject.toml [tool.pytest.ini_options] pythonpath=["."], pytest
resuelve imports del proyecto automaticamente. Los mocks de modulos
externos se aplican aqui una sola vez en lugar de repetirse en cada archivo.

Comportamiento:
  - Si el paquete real esta instalado: NO se mockea (integracion funciona).
  - Si el paquete NO esta instalado: se inyecta MagicMock (unit tests siguen
    funcionando porque usan @patch o mocks locales, no los modulos reales).

Esto permite ejecutar `pytest tests/` en ambos entornos:
  - Restringido (sin NIM/MinIO): unit tests pasan, integracion se salta.
  - Desarrollo (con NIM/MinIO): todo pasa junto.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Asegurar que el directorio raiz del proyecto esta en sys.path.
# Esto permite importar shared.* y sandbox_mteb.* independientemente
# de si pyproject.toml tiene pythonpath=["."] configurado.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Modulos de infraestructura que pueden no estar disponibles.
_INFRA_MODULES = [
    "boto3",
    "botocore",
    "botocore.exceptions",
    "langchain_nvidia_ai_endpoints",
    "langchain_core",
    "langchain_core.messages",
    "langchain_core.documents",
    "langchain_core.embeddings",
    "langchain_chroma",
    "chromadb",
]

for mod in _INFRA_MODULES:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()
