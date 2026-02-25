"""
Modulo: Tantivy Index
Descripcion: Indice BM25 basado en Tantivy (motor Rust de busqueda full-text).
             Reemplaza rank_bm25 (Python puro, in-memory) con un indice
             invertido real en disco.

Ubicacion: shared/retrieval/tantivy_index.py

Ventajas sobre rank_bm25:
  - O(k) por query vs O(n) scan lineal
  - Stemmer Snowball integrado (en, es, de, fr, it, pt, ...)
  - Indice en disco, reutilizable entre ejecuciones
  - BM25 nativo con parametros k1, b configurables
  - Escalable a millones de documentos

API compatible con BM25Index para swap transparente en HybridRetriever.
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import tantivy
    HAS_TANTIVY = True
except ImportError:
    HAS_TANTIVY = False
    tantivy = None


# Mapeo de codigos de idioma a nombres de stemmer Tantivy
_TANTIVY_LANGUAGES = {
    "en": "en_stem",
    "es": "es_stem",
    "de": "de_stem",
    "fr": "fr_stem",
    "it": "it_stem",
    "pt": "pt_stem",
    "ru": "ru_stem",
    "nl": "nl_stem",
    "sv": "sv_stem",
    "fi": "fi_stem",
    "da": "da_stem",
    "hu": "hu_stem",
    "ro": "ro_stem",
    "tr": "tr_stem",
    "ar": "ar_stem",
}


class TantivyIndex:
    """
    Indice BM25 basado en Tantivy.

    API compatible con BM25Index:
        build_index(documents) -> int
        search(query, top_k) -> List[Tuple[doc_id, content, score]]
        clear() -> None
        size -> int
    """

    def __init__(
        self,
        index_dir: Optional[Path] = None,
        language: str = "en",
    ):
        """
        Crea un indice Tantivy.

        Args:
            index_dir: Directorio para el indice en disco. Si None, usa tmpdir.
            language: Codigo de idioma para stemmer (en, es, de, fr, ...).
        """
        if not HAS_TANTIVY:
            raise ImportError(
                "tantivy no instalado: pip install tantivy"
            )

        self._language = language
        self._tokenizer_name = _TANTIVY_LANGUAGES.get(language, "en_stem")
        self._owns_dir = index_dir is None
        self._index_dir = index_dir or Path(tempfile.mkdtemp(prefix="tantivy_"))
        self._index: Optional[tantivy.Index] = None
        self._doc_count = 0

        # Mapa doc_id -> content para devolver en resultados de busqueda
        self._doc_contents: Dict[str, str] = {}

        logger.debug(
            f"TantivyIndex: dir={self._index_dir}, "
            f"lang={language}, tokenizer={self._tokenizer_name}"
        )

    def build_index(self, documents: List[Dict[str, Any]]) -> int:
        """
        Indexa documentos.

        Args:
            documents: Lista de dicts con keys: doc_id, content, title (opcional).

        Returns:
            Numero de documentos indexados.
        """
        if not documents:
            return 0

        # Limpiar indice previo si existe
        self._cleanup_index()

        # Construir schema
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field(
            "doc_id", stored=True, tokenizer_name="raw"
        )
        schema_builder.add_text_field(
            "title", stored=True, tokenizer_name=self._tokenizer_name
        )
        schema_builder.add_text_field(
            "content", stored=True, tokenizer_name=self._tokenizer_name
        )
        schema = schema_builder.build()

        # Crear directorio si no existe
        self._index_dir.mkdir(parents=True, exist_ok=True)

        # Crear indice
        self._index = tantivy.Index(schema, path=str(self._index_dir))

        # Indexar documentos
        writer = self._index.writer(heap_size=50_000_000)  # 50MB heap

        for doc in documents:
            doc_id = doc.get("doc_id", "")
            content = doc.get("content", "")
            title = doc.get("title", "")

            self._doc_contents[doc_id] = content

            writer.add_document(tantivy.Document(
                doc_id=doc_id,
                title=title,
                content=content,
            ))

        writer.commit()
        self._index.reload()

        self._doc_count = len(documents)

        logger.debug(f"TantivyIndex: {self._doc_count} documentos indexados")
        return self._doc_count

    def search(
        self, query: str, top_k: int = 50
    ) -> List[Tuple[str, str, float]]:
        """
        Busqueda BM25.

        Args:
            query: Texto de busqueda.
            top_k: Numero maximo de resultados.

        Returns:
            Lista de (doc_id, content, score) ordenada por score descendente.
        """
        if self._index is None or self._doc_count == 0:
            return []

        if not query or not query.strip():
            return []

        try:
            # Sanitizar query: Tantivy parse_query falla con caracteres
            # especiales como apostrofes, comillas, parentesis, ?, etc.
            # Eliminamos todo lo que no sea alfanumerico o espacio.
            clean_query = re.sub(r'[^\w\s]', ' ', query)
            clean_query = ' '.join(clean_query.split())  # normalizar espacios

            if not clean_query:
                return []

            # Parsear query sobre title y content
            parsed_query = self._index.parse_query(
                clean_query, ["title", "content"]
            )

            searcher = self._index.searcher()
            search_result = searcher.search(parsed_query, top_k)

            results = []
            for score, doc_addr in search_result.hits:
                doc = searcher.doc(doc_addr)
                doc_id = doc.get_first("doc_id") or ""
                content = self._doc_contents.get(doc_id, "")
                results.append((doc_id, content, float(score)))

            return results

        except Exception as e:
            logger.warning(f"TantivyIndex search error: {e}")
            return []

    def clear(self) -> None:
        """Elimina el indice y limpia recursos."""
        self._cleanup_index()
        self._doc_contents.clear()
        self._doc_count = 0

    def _cleanup_index(self) -> None:
        """Limpia recursos del indice."""
        self._index = None
        if self._owns_dir and self._index_dir.exists():
            try:
                shutil.rmtree(self._index_dir)
                self._index_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Error limpiando indice Tantivy: {e}")

    @property
    def size(self) -> int:
        return self._doc_count

    def __del__(self) -> None:
        """Limpia tmpdir si es propio."""
        if self._owns_dir and self._index_dir.exists():
            try:
                shutil.rmtree(self._index_dir, ignore_errors=True)
            except Exception:
                pass


__all__ = ["TantivyIndex", "HAS_TANTIVY"]
