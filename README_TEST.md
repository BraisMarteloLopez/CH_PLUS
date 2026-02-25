# Tests — RAG_P

147 unit tests + 15 tests de integracion (162 total). Ejecutables con Python 3.10+.

```bash
pytest tests/                      # Todo junto (unit + integracion)
pytest tests/ -v                   # Verbose
pytest tests/ -m "not integration" # Solo unit
pytest tests/integration/ -v       # Solo integracion
pytest tests/test_dtm4_rrf.py     # Archivo especifico
```

## Infraestructura

| Componente | Descripcion |
|---|---|
| `tests/conftest.py` | Mocks **condicionales**: solo mockea `boto3`, `langchain_*`, `chromadb`, `botocore` si el paquete real no esta instalado (`try __import__`, fallback `MagicMock`). En entornos con dependencias reales, los modulos se preservan. |
| `tests/integration/conftest.py` | Carga `.env` real desde `sandbox_mteb/.env`. Si no existe o faltan variables, salta todos los tests de integracion automaticamente. Fixtures de sesion: `mteb_config`, `minio_loader`, `embedding_base_url`, `llm_base_url`. |
| `pyproject.toml` | `testpaths = ["tests"]`, `pythonpath = ["."]`. |
| CI/CD | Pendiente. `pytest tests/` ejecutable en cualquier entorno con Python 3.10+. |

## Cobertura por archivo

| Archivo | Que testea | Tests |
|---|---|---|
| `test_dt5_pre_rerank_traceability.py` | `_execute_retrieval()` con/sin reranker: `pre_rerank_candidate_ids` poblado/vacio, doc promovido trazable a posicion original | 3 |
| `test_dt6_01_faithfulness_sync.py` | Faithfulness sync: contexto largo pasa integro al judge | 1 |
| `test_dt6_02_faithfulness_async.py` | Faithfulness async: contexto largo pasa integro al judge | 1 |
| `test_dt6_03_context_utilization.py` | Context utilization sync/async: contexto largo integro, vacio retorna 0.0 | 3 |
| `test_dt6_context_truncation.py` | DT-6: faithfulness y context_utilization (sync/async) pasan contexto >4000 chars integro al judge. Contexto vacio retorna 0.0 sin invocar judge. | 5 |
| `test_dt7_05_06_rerank_status.py` | Rerank exitoso: `reranked_ok=True`, contador=0. Rerank fallido: `reranked_ok=False`, contador incrementa. | 2 |
| `test_dt7_07_no_reranker.py` | Sin reranker: `reranked_status=None`, `generation_doc_ids` vacio, `rerank_failures=None` en config_snapshot. | 1 |
| `test_dt7_08_csv_reranked.py` | Columna `reranked` en detail CSV con valores True/False/vacio. | 1 |
| `test_dt8_09_10_11_reranker_sort.py` | Sort descendente por `relevance_score`, scores identicos no fallan, score ausente = 0.0. | 3 |
| `test_dt9_extract_score_fallback.py` | `_extract_score_fallback()`: decimales 0-1, falsos positivos rechazados, escala 1-10 con prefijo, fracciones N/M, defaults, respuestas LLM reales. | 21 |
| `test_dtm4_build_run_aggregation.py` | `_build_run()` agregacion matematica (avg_hit@5, MRR, recall, complement, generation con zeros, failed excluidas) + contrato JSON `to_dict()`/`to_dict_full()` (schema keys, rounding, campos condicionales). | 18 |
| `test_dtm4_loader_populate.py` | `_populate_from_dataframes()`: queries/corpus/qrels, multiples qrels, DataFrames None/vacios, answer_type inferido, metadata question_type. | 9 |
| `test_dtm4_rrf.py` | `reciprocal_rank_fusion()`: rankings vacios, orden, doc en ambos/un ranking, pesos, top_n, formula RRF, 3 rankings, parametro k. | 11 |
| `test_dtm4_subset_selection.py` | `_select_subset_dev()`: gold docs en corpus, distractores, seed determinista, error si gold > corpus, gold ausente. | 9 |
| `test_dtm4_tantivy_edge_cases.py` | `TantivyIndex` via `@patch`: sanitizacion regex, guards (query vacia/whitespace/None), clear, build vacio, constructor language. | 17 |
| `test_dtm5_12_13_secondary_metric_errors.py` | Metricas secundarias fallidas: `MetricResult(value=0.0, error=...)` + warning. Todas fallan: todas con error. | 3 |
| `test_dtm17_generation_retrieval_metrics.py` | Metricas de retrieval efectivo (post-rerank): `generation_recall`, `generation_hit`, `reranker_rescue_count`, agregacion en `_build_run()`. | 15 |
| `test_format_context.py` | `_format_context()`: placeholder vacio, headers [Doc N], separador, truncacion, boundary exacto. | 9 |
| `test_metrics_reference_based.py` | TextNormalizer (accents, dashes, articles) + F1 (boundaries, overlap, duplicados, normalizacion) + EM (boundaries, normalizacion, sin normalize) + Accuracy (basic, extra text, valid_labels). | 15 |

## Diseno: que se testea y que no

**Principio:** testear toda la computacion pura. Mocks solo para aislar modulos de infra (NIM, MinIO, ChromaDB).

### Testeable sin infraestructura (implementado)

| Metodo | Tipo | Tests | Valor |
|---|---|---|---|
| `_build_run()` + `to_dict()` | Agregacion + serializacion | 18 | Paso final del pipeline. Errores corrompen JSON. |
| generation/retrieval metrics | Agregacion post-rerank | 15 | `generation_recall`, `generation_hit`, `reranker_rescue_count`. |
| `_select_subset_dev()` | Logica Python pura | 9 | Invariantes: gold docs en corpus, seed determinista. |
| `reciprocal_rank_fusion()` | Formula matematica | 11 | `score = weight / (k + rank)`. Verificable. |
| `_populate_from_dataframes()` | Parsing via MockDataFrame | 9 | Interfaz: solo `iterrows()` + `get()`. pandas no necesario. |
| `TantivyIndex` guards | `@patch` HAS_TANTIVY | 17 | Guards retornan antes de tocar Rust. |
| `_extract_score_fallback()` | Regex pura | 21 | 3 patrones: decimales, escala 1-10, fracciones. |
| `_format_context()` | String formatting | 9 | Truncacion por max_chars. |
| TextNormalizer + metricas | Computacion pura | 15 | F1, EM, Accuracy con normalizacion. |
| faithfulness/context_util | Mock judge | 10 | No trunca contexto >4000 chars. |
| Reranker sort + status | Mock retriever/reranker | 7 | Sort descendente, fallback detectado. |

### No testeable sin infraestructura (documentado)

| Metodo | Requiere | Razon de no-test |
|---|---|---|
| `_init_components()` | NIM endpoints | Wiring. Valor bajo. |
| `_index_documents()` | ChromaDB + embeddings | Mock seria testear mocks. |
| `_execute_retrieval()` | ChromaDB poblado | Logica testeable (RRF, rerank) ya cubierta. |
| `_batch_embed_queries()` | REST NIM | Retry con backoff es simple. |
| `_execute_generation_async()` | LLM service | Metricas individuales ya testeadas. |
| Pipeline `run()` completo | Todo lo anterior | Solo en produccion con NIM + MinIO + ChromaDB. |

### MockDataFrame en loader tests

`_populate_from_dataframes()` solo usa `df.iterrows()` y `row.get()` — interfaz `Mapping`. No usa NaN handling, dtypes, `.loc`, `.iloc`. La conversion Parquet→DataFrame ocurre en `_download_parquet()` / `_load_from_cache()`, una capa distinta. MockDataFrame implementa exactamente la interfaz consumida. pandas no instalado ni necesario.
