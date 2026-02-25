"""
Modulo: Vector Store
Descripcion: Wrapper ChromaDB para evaluacion RAG.

Ubicacion: shared/vector_store.py

FIX: delete_all_documents() ahora elimina la coleccion subyacente
en Chroma y la recrea, en lugar de solo reasignar el wrapper Python.
"""

import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

from shared.types import EmbeddingModelProtocol

logger = logging.getLogger(__name__)

# Desactivar telemetria de ChromaDB (posthog) ANTES de importar.
# Sin esto, chromadb intenta conectar a us.i.posthog.com al salir,
# y si la red no tiene acceso, el proceso se cuelga indefinidamente.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

try:
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    Chroma = None
    Document = None
    Embeddings = None
    chromadb = None


class ChromaVectorStore:
    """
    Wrapper de ChromaDB para evaluacion RAG.

    Uso:
        store = ChromaVectorStore(config, embedding_model)
        store.add_documents(documents)
        results = store.similarity_search_with_score("query", k=5)
        store.delete_all_documents()
    """

    def __init__(self, config: Dict[str, Any], embedding_model: EmbeddingModelProtocol):
        if not HAS_CHROMA:
            raise ImportError("pip install langchain-chroma chromadb")

        self.collection_name = config.get(
            "CHROMA_COLLECTION_NAME",
            f"eval_collection_{uuid.uuid4().hex[:8]}",
        )
        self.persist_directory = config.get("CHROMA_PERSIST_DIRECTORY")
        self.embedding_model = embedding_model
        self.batch_size = config.get("EMBEDDING_BATCH_SIZE", 0)

        # HNSW num_threads=1 reduce no-determinismo del grafo (elimina
        # variabilidad de threading). Sin embargo, ChromaDB 0.5-0.6 no soporta
        # hnsw:random_seed, por lo que la asignacion de niveles HNSW sigue
        # siendo no-determinista entre runs con distinto collection_name.
        # Ver DTm-13 en README.md.
        self._hnsw_num_threads = config.get("HNSW_NUM_THREADS", 1)
        self._collection_metadata = {
            "hnsw:num_threads": self._hnsw_num_threads,
        }

        # Crear cliente Chroma nativo para control directo de colecciones
        if self.persist_directory:
            self._client = chromadb.PersistentClient(
                path=self.persist_directory
            )
        else:
            self._client = chromadb.Client()

        chroma_kwargs = {
            "client": self._client,
            "collection_name": self.collection_name,
            "embedding_function": embedding_model,
            "collection_metadata": self._collection_metadata,
        }

        self._store = Chroma(**chroma_kwargs)
        self._document_count = 0
        logger.debug(
            f"ChromaVectorStore inicializado: {self.collection_name} "
            f"(hnsw_threads={self._hnsw_num_threads})"
        )

    def add_documents(self, documents: List[Any]) -> List[str]:
        if not documents:
            logger.warning("add_documents llamado con lista vacia")
            return []
        try:
            all_ids = []
            if self.batch_size > 0 and len(documents) > self.batch_size:
                total_batches = (
                    len(documents) + self.batch_size - 1
                ) // self.batch_size
                for i in range(0, len(documents), self.batch_size):
                    batch = documents[i : i + self.batch_size]
                    batch_num = (i // self.batch_size) + 1
                    logger.debug(
                        f"Indexando batch {batch_num}/{total_batches} "
                        f"({len(batch)} docs)"
                    )
                    ids = self._store.add_documents(batch)
                    all_ids.extend(ids)
            else:
                all_ids = self._store.add_documents(documents)

            self._document_count += len(documents)
            logger.debug(
                f"Anadidos {len(documents)} documentos. "
                f"Total: {self._document_count}"
            )
            return all_ids
        except Exception as e:
            logger.error(f"Error anadiendo documentos: {e}")
            raise

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Any, float]]:
        try:
            if filter:
                return self._store.similarity_search_with_score(  # type: ignore[no-any-return]
                    query, k=k, filter=filter
                )
            return self._store.similarity_search_with_score(query, k=k)  # type: ignore[no-any-return]
        except Exception as e:
            logger.error(f"Error en busqueda: {e}")
            return []

    def similarity_search_by_vector_with_score(
        self,
        vector: List[float],
        k: int = 5,
    ) -> List[Tuple[Any, float]]:
        """
        Busqueda por vector pre-computado. Evita llamada al embedding model.

        FIX DTm-2: usa self._client.get_collection() (API publica de chromadb)
        en lugar de self._store._collection (API interna de LangChain Chroma).
        """
        try:
            from langchain_core.documents import Document

            collection = self._client.get_collection(self.collection_name)
            results = collection.query(
                query_embeddings=[vector],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )

            output = []
            if results and results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    content = results["documents"][0][i] if results["documents"] else ""
                    distance = results["distances"][0][i] if results["distances"] else 0.0

                    doc = Document(page_content=content, metadata=metadata)
                    output.append((doc, float(distance)))

            return output

        except Exception as e:
            logger.error(f"Error en busqueda por vector: {e}")
            return []

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        return [
            doc
            for doc, _ in self.similarity_search_with_score(query, k, filter)
        ]

    def delete_all_documents(self) -> None:
        """
        Limpia la coleccion eliminandola y recreandola.

        FIX: La version anterior solo reasignaba el wrapper Python,
        sin eliminar los datos de la coleccion subyacente en Chroma.
        Ahora se usa el cliente nativo para delete + recreate.
        """
        try:
            # Eliminar coleccion via cliente nativo
            self._client.delete_collection(self.collection_name)

            # Recrear wrapper LangChain apuntando a coleccion nueva
            chroma_kwargs = {
                "client": self._client,
                "collection_name": self.collection_name,
                "embedding_function": self.embedding_model,
                "collection_metadata": self._collection_metadata,
            }
            self._store = Chroma(**chroma_kwargs)
            self._document_count = 0
            logger.debug(
                f"Coleccion {self.collection_name} eliminada y recreada"
            )
        except Exception as e:
            logger.error(f"Error eliminando coleccion: {e}")
            self._document_count = 0

    def get_document_count(self) -> int:
        return self._document_count

    def __repr__(self) -> str:
        return (
            f"ChromaVectorStore(collection='{self.collection_name}', "
            f"documents={self._document_count})"
        )


__all__ = ["ChromaVectorStore", "HAS_CHROMA"]
