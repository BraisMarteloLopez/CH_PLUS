"""
MinIO Dataset Loader para sandbox MTEB.

Carga datasets MTEB/BeIR pre-descargados desde MinIO (formato Parquet).

Estructura esperada en MinIO:
    s3://{bucket}/{prefix}/
    +-- manifest.json
    +-- hotpotqa/
    |   +-- queries.parquet
    |   +-- corpus.parquet
    |   +-- qrels.parquet
    |   +-- metadata.json
    +-- fever/
        +-- ...
"""

import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

from shared.types import (
    LoadedDataset,
    NormalizedQuery,
    NormalizedDocument,
    get_dataset_config,
)
from .config import MinIOStorageConfig

logger = logging.getLogger(__name__)


class MinIOLoader:
    """
    Carga datasets de evaluacion desde MinIO.

    A diferencia de la version original, no tiene from_settings().
    Se construye con parametros explicitos desde MTEBConfig.
    """

    def __init__(self, storage_config: MinIOStorageConfig):
        endpoint = storage_config.minio_endpoint
        if not endpoint.startswith("http"):
            endpoint = f"http://{endpoint}"

        self.endpoint = endpoint
        self.bucket = storage_config.minio_bucket
        self.prefix = storage_config.s3_datasets_prefix
        self.cache_dir = storage_config.datasets_cache_dir

        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=storage_config.minio_access_key,
            aws_secret_access_key=storage_config.minio_secret_key,
        )

        self._manifest: Optional[Dict] = None
        logger.info(f"MinIOLoader: {endpoint}/{self.bucket}/{self.prefix}")

    # -----------------------------------------------------------------
    # API PUBLICA
    # -----------------------------------------------------------------

    def check_connection(self) -> bool:
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return True
        except ClientError as e:
            logger.error(f"Error conexion MinIO: {e}")
            return False

    def list_available_datasets(self) -> List[str]:
        manifest = self._get_manifest()
        return manifest.get("datasets", [])

    def load_dataset(
        self,
        dataset_name: str,
        use_cache: bool = True,
    ) -> LoadedDataset:
        """
        Carga un dataset desde MinIO y lo normaliza.

        NO aplica limites de queries/corpus. El evaluador decide cuantos
        usar en su run. Esto elimina la mutacion in-place del original.
        """
        ds_config = get_dataset_config(dataset_name)

        # Intentar cache local primero
        if use_cache and self.cache_dir:
            cached = self._load_from_cache(dataset_name)
            if cached:
                logger.info(f"Dataset '{dataset_name}' cargado desde cache")
                return cached

        logger.info(f"Descargando dataset '{dataset_name}' desde MinIO...")

        result = LoadedDataset(
            name=dataset_name,
            dataset_type=ds_config["type"],
            primary_metric=ds_config["primary_metric"],
            secondary_metrics=ds_config.get("secondary_metrics", []),
        )

        try:
            queries_df = self._download_parquet(f"{dataset_name}/queries.parquet")
            corpus_df = self._download_parquet(f"{dataset_name}/corpus.parquet")
            qrels_df = self._download_parquet(f"{dataset_name}/qrels.parquet")

            self._populate_from_dataframes(result, queries_df, corpus_df, qrels_df)

            # Metadata
            result.metadata = self._download_json(f"{dataset_name}/metadata.json") or {}
            result.load_status = "success"

            logger.info(f"Dataset '{dataset_name}' cargado: {result.get_statistics()}")

            # Guardar en cache
            if use_cache and self.cache_dir:
                self._save_to_cache(result)

            return result

        except Exception as e:
            logger.error(f"Error cargando dataset '{dataset_name}': {e}")
            result.load_status = "error"
            result.error_message = str(e)
            return result

    # -----------------------------------------------------------------
    # PRIVATE: DataFrame -> LoadedDataset (FIX DTm-1)
    # -----------------------------------------------------------------

    @staticmethod
    def _populate_from_dataframes(
        result: LoadedDataset,
        queries_df,
        corpus_df,
        qrels_df,
    ) -> None:
        """
        Puebla un LoadedDataset a partir de DataFrames de queries, corpus y qrels.

        Extraido de load_dataset() y _load_from_cache() para eliminar
        duplicacion (~50 lineas identicas).
        """
        qrels: Dict[str, List[str]] = {}

        if queries_df is not None:
            result.total_queries = len(queries_df)
            for _, row in queries_df.iterrows():
                qid = str(row.get("query_id", ""))
                raw_answer = str(row.get("answer", "")) if row.get("answer") else ""
                raw_answer_type = str(row.get("answer_type", ""))
                if not raw_answer_type and raw_answer:
                    raw_answer_type = "text"
                question_type = str(row.get("question_type", "") or row.get("type", ""))

                result.queries.append(NormalizedQuery(
                    query_id=qid,
                    query_text=str(row.get("text", "")),
                    expected_answer=raw_answer or None,
                    answer_type=raw_answer_type or None,
                    metadata={
                        "question_type": question_type,
                        "level": str(row.get("level", "")),
                    },
                ))

        if corpus_df is not None:
            result.total_corpus = len(corpus_df)
            for _, row in corpus_df.iterrows():
                did = str(row.get("doc_id", ""))
                result.corpus[did] = NormalizedDocument(
                    doc_id=did,
                    title=str(row.get("title", "")),
                    content=str(row.get("text", "")),
                )

        if qrels_df is not None:
            for _, row in qrels_df.iterrows():
                qid = str(row.get("query_id", ""))
                did = str(row.get("doc_id", ""))
                qrels.setdefault(qid, []).append(did)

        for query in result.queries:
            query.relevant_doc_ids = qrels.get(query.query_id, [])

    # -----------------------------------------------------------------
    # PRIVATE: S3
    # -----------------------------------------------------------------

    def _get_manifest(self, force_refresh: bool = False) -> Dict:
        if self._manifest and not force_refresh:
            return self._manifest
        try:
            resp = self.client.get_object(
                Bucket=self.bucket,
                Key=f"{self.prefix}/manifest.json",
            )
            self._manifest = json.loads(resp["Body"].read().decode("utf-8"))
            return self._manifest
        except ClientError as e:
            logger.warning(f"No se encontro manifest: {e}")
            return {"datasets": [], "error": str(e)}

    def _download_parquet(self, key: str):
        import pandas as pd

        full_key = f"{self.prefix}/{key}"
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=full_key)
            data = resp["Body"].read()
            return pd.read_parquet(io.BytesIO(data))
        except ClientError as e:
            logger.warning(f"No se pudo descargar {full_key}: {e}")
            return None

    def _download_json(self, key: str) -> Optional[Dict]:
        full_key = f"{self.prefix}/{key}"
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=full_key)
            return json.loads(resp["Body"].read().decode("utf-8"))
        except ClientError:
            return None

    # -----------------------------------------------------------------
    # PRIVATE: CACHE LOCAL
    # -----------------------------------------------------------------

    def _load_from_cache(self, dataset_name: str) -> Optional[LoadedDataset]:
        import pandas as pd

        cache_dir = self.cache_dir / dataset_name
        if not cache_dir.exists():
            return None

        queries_path = cache_dir / "queries.parquet"
        corpus_path = cache_dir / "corpus.parquet"
        qrels_path = cache_dir / "qrels.parquet"

        if not all(p.exists() for p in [queries_path, corpus_path]):
            return None

        ds_config = get_dataset_config(dataset_name)
        result = LoadedDataset(
            name=dataset_name,
            dataset_type=ds_config["type"],
            primary_metric=ds_config["primary_metric"],
            secondary_metrics=ds_config.get("secondary_metrics", []),
        )

        queries_df = pd.read_parquet(queries_path)
        corpus_df = pd.read_parquet(corpus_path)
        qrels_df = pd.read_parquet(qrels_path) if qrels_path.exists() else None

        self._populate_from_dataframes(result, queries_df, corpus_df, qrels_df)

        result.load_status = "success"
        return result

    def _save_to_cache(self, dataset: LoadedDataset) -> None:
        import pandas as pd

        if not self.cache_dir:
            return

        cache_dir = self.cache_dir / dataset.name
        cache_dir.mkdir(parents=True, exist_ok=True)

        queries_data = [
            {
                "query_id": q.query_id,
                "text": q.query_text,
                "answer": q.expected_answer or "",
                "answer_type": q.answer_type or "",
                "question_type": q.metadata.get("question_type", ""),
                "level": q.metadata.get("level", ""),
            }
            for q in dataset.queries
        ]
        pd.DataFrame(queries_data).to_parquet(
            cache_dir / "queries.parquet", index=False
        )

        corpus_data = [
            {"doc_id": d.doc_id, "title": d.title or "", "text": d.content}
            for d in dataset.corpus.values()
        ]
        pd.DataFrame(corpus_data).to_parquet(
            cache_dir / "corpus.parquet", index=False
        )

        qrels_data = [
            {"query_id": q.query_id, "doc_id": did, "relevance": 1}
            for q in dataset.queries
            for did in q.relevant_doc_ids
        ]
        pd.DataFrame(qrels_data).to_parquet(
            cache_dir / "qrels.parquet", index=False
        )

        logger.debug(f"Dataset '{dataset.name}' guardado en cache local")


__all__ = ["MinIOLoader"]
