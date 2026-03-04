# RAG_P v3.2

Sistema de evaluacion RAG (Retrieval-Augmented Generation) para benchmarking de pipelines de recuperacion y generacion sobre datasets MTEB/BeIR (HotpotQA actualmente) con infraestructura NVIDIA NIM.

## Arquitectura

```
RAG_P/
├── shared/                          # Libreria compartida
│   ├── types.py                     # NormalizedQuery, LoadedDataset, EvaluationRun, Protocols
│   ├── metrics.py                   # F1, ExactMatch, Accuracy, Faithfulness (LLM-judge)
│   ├── llm.py                       # AsyncLLMService, load_embedding_model
│   ├── config_base.py               # InfraConfig, RerankerConfig, helpers _env_*
│   ├── report.py                    # RunExporter: JSON + CSV summary + CSV detail
│   ├── vector_store.py              # ChromaVectorStore
│   ├── structured_logging.py        # Logging JSONL estructurado
│   └── retrieval/
│       ├── __init__.py              # Factory get_retriever()
│       ├── core.py                  # BaseRetriever, SimpleVectorRetriever, RetrievalConfig
│       ├── hybrid_retriever.py      # BM25 + Vector + RRF
│       ├── hybrid_plus_retriever.py # BM25+Vector+RRF + NER cross-linking
│       ├── entity_linker.py         # NER (spaCy) + indice invertido + cross-refs
│       ├── reranker.py              # CrossEncoderReranker (NVIDIARerank)
│       └── tantivy_index.py         # BM25 via Tantivy (Rust, fallback rank-bm25)
│
├── sandbox_mteb/                    # Evaluacion MTEB/BeIR
│   ├── config.py                    # MTEBConfig: .env -> dataclass validada
│   ├── loader.py                    # MinIO/Parquet -> LoadedDataset
│   ├── evaluator.py                 # Pipeline: pre-embed + retrieval + gen async
│   ├── run.py                       # Entry point (--dry-run, -v)
│   └── env.example                  # Plantilla .env
│
├── tests/                           # 191 unit + 3 integration tests (pytest)
│   ├── conftest.py                  # Mocks condicionales (solo si paquete no instalado)
│   ├── test_*.py                    # 20 archivos — ver README_TEST.md
│   └── integration/                 # Tests contra NIM + MinIO reales
│       ├── conftest.py              # Carga .env real, fixtures de sesion
│       └── test_*.py                # Requieren infraestructura accesible
│
├── pyproject.toml                   # Config pytest
├── mypy.ini                         # Config mypy
└── requirements.txt
```

`download_datasets/` (ETL HuggingFace->Parquet->MinIO) y `data/` (cache, resultados) no estan en el repositorio.

## Estrategias de retrieval

| Estrategia | Indexacion | Busqueda | Reranker |
|---|---|---|---|
| `SIMPLE_VECTOR` | Embedding directo (NIM) | Cosine similarity (ChromaDB) | Opcional |
| `HYBRID_PLUS` | NER cross-refs + Embedding + BM25 | BM25 (Tantivy) + Vector + RRF | Opcional |

`HYBRID_PLUS`: durante indexacion, spaCy NER extrae entidades de cada documento, construye un indice invertido in-memory, y genera cross-references textuales entre documentos que comparten entidades. El texto indexado es: contenido original + cross-refs. Esto permite que BM25 capture terminos puente entre documentos, mejorando retrieval en bridge questions multi-hop. Sin dependencia de NIM para indexacion. Sin spaCy instalado, se comporta como BM25+Vector+RRF puro (sin cross-refs, warning en log).

### Entity cross-linking (HYBRID_PLUS)

Motivacion: en HotpotQA, ~80% de las preguntas son de tipo **bridge** (multi-hop). Ejemplo: "Where was the director of Sinister born?" requiere recuperar Doc A ("Sinister") para identificar al director (Scott Derrickson), y Doc B ("Scott Derrickson") para encontrar su lugar de nacimiento. El bi-encoder genera un vector mas cercano a "Sinister" que a "Scott Derrickson", por lo que Doc B frecuentemente no se recupera.

Mecanismo: si Doc A ("Sinister") contiene una cross-reference textual `Related: Scott Derrickson (shared: scott derrickson)`, BM25 captura el termino "Scott Derrickson" en Doc A, promoviendo Doc B via RRF aunque el vector de la query no lo alcance directamente.

```
Flujo de indexacion HYBRID_PLUS:

  [Docs originales]
        |
        v
  [spaCy NER: extrae PERSON, ORG, GPE, LOC, ...]
        |
        v
  [EntityLinker: indice invertido in-memory, IDF filter (>5% corpus)]
        |
        v
  [Cross-refs textuales por documento]
        |
        v
  [Doc original + Cross-refs] --> [Index BM25+Vector]
```

Componentes en `shared/retrieval/entity_linker.py`:

| Componente | Funcion |
|---|---|
| `normalize_entity()` | Lowercase + eliminar articulos + colapsar espacios + eliminar puntuacion (preserva guiones internos) |
| `EntityExtractor` | spaCy NER con filtrado de tipos relevantes, deduplicacion, longitud minima 2 chars |
| `EntityLinker` | Indice invertido + IDF filter + generacion de cross-refs (top-N docs con mas entidades compartidas) |
| `DocEntities` | Dataclass intermedio: entidades extraidas por documento |

Parametros configurables via `.env`:

| Variable | Default | Descripcion |
|---|---|---|
| `ENTITY_MAX_CROSS_REFS` | 3 | Top-N documentos referenciados por documento |
| `ENTITY_MIN_SHARED` | 1 | Minimo entidades compartidas para generar cross-ref |
| `ENTITY_MAX_DOC_FRACTION` | 0.05 | Umbral IDF: ignorar entidades en >5% del corpus |

Limitaciones aceptadas:
- Normalizacion basica: no resuelve aliases (US/United States) ni formas parciales (Derrickson/Scott Derrickson). spaCy tipicamente reporta la forma completa en texto Wikipedia.
- Modelo `en_core_web_sm` (15MB, solo NER). Suficiente para entidades nombradas en Wikipedia.

## Pipeline de evaluacion

```
.env -> MTEBConfig -> MinIO/cache(Parquet) -> LoadedDataset
     -> shuffle(seed) -> slice(max_corpus)
     -> [NER + cross-linking (spaCy) si HYBRID_PLUS]
     -> index(ChromaDB [+ Tantivy si HYBRID_PLUS])
     -> pre-embed queries (batch REST NIM)
     -> retrieve(local ChromaDB [+ BM25 + RRF si HYBRID_PLUS], sync)
     -> [rerank(cross-encoder) si habilitado]
     -> generate + metrics (async)
     -> EvaluationRun -> JSON + CSV
```

Queries se pre-embeben en batch via REST antes del loop de retrieval. La fase de retrieval usa vectores pre-computados con busqueda local, eliminando roundtrip REST por query. Fallback a embedding por query si el batch falla.

## Metricas

### Retrieval

Hit@K, MRR, Recall@K (K=1,3,5,10,20), NDCG@K sobre top `RETRIEVAL_K` documentos (pre-rerank).

### Generacion

Metrica primaria segun `answer_type` del query:

- `"label"` (yes/no): Accuracy (match normalizado)
- Otro (extractiva): F1 (token-overlap normalizado)

EM (Exact Match) siempre como secundaria. Faithfulness (LLM-judge) como secundaria si configurada.

### Separacion retrieval vs generacion

| Config | Metricas retrieval | Contexto generacion |
|---|---|---|
| Reranker OFF | top RETRIEVAL_K | top RETRIEVAL_K |
| Reranker ON | top RETRIEVAL_K (pre-rerank) | top RERANKER_TOP_N (post-rerank) |

### Retrieval efectivo (post-rerank)

Cuando el reranker esta activo, las metricas de retrieval (pre-rerank) pueden subestimar la calidad del contexto que recibe el LLM. El reranker opera sobre `PRE_FUSION_K` candidatos y puede promover docs de posiciones 21-150 al top de generacion.

Metricas adicionales sobre `generation_doc_ids` (post-rerank):

- `generation_recall`: fraccion de gold docs en el set de generacion.
- `generation_hit`: 1.0 si algun gold doc en el set de generacion, 0.0 si no.
- `reranker_rescue_count` (run-level): queries donde retrieval recall@K=0 pero generation_recall>0.

Sin reranker, estos campos no se emiten.

## Dataset: HotpotQA

Fuente: HotpotQA fullwiki (validation split) via HuggingFace.

| Propiedad | Valor |
|---|---|
| Queries | 7405 |
| Corpus | 66576 documentos (Wikipedia, deduplicados por titulo) |
| Qrels | 14810 (2.0 por query, solo supporting_facts) |
| Tipos de query | bridge (~80%), comparison (~20%) |
| Almacenamiento | MinIO (Parquet), `s3://lakehouse/datasets/evaluation/hotpotqa/` |
| Schema | v2.0: query_id, text, answer, answer_type, question_type, level |

**Limitacion del corpus.** 10 pasajes por query (2 gold + 8 distractores), deduplicados por titulo. Gold docs garantizados. Resultados **no comparables con benchmarks publicados** (corpus completo ~5.2M). Solo comparaciones relativas entre estrategias son validas. Corpus se baraja con seed fijo antes de `EVAL_MAX_CORPUS`.

**Retrieval multi-hop.** Cada query requiere 2 gold docs que tipicamente cubren espacios semanticos distintos (ej: una query bridge necesita un doc sobre la entidad mencionada y otro sobre la entidad preguntada). Un bi-encoder genera un unico vector por query, inherentemente mas cercano a uno de los dos temas. Esto explica el gap tipico entre recall@1 (~0.46) y recall@5 (~0.82) en corpus completo con SIMPLE_VECTOR: el primer gold doc se recupera rapido, el segundo requiere profundidad. HYBRID_PLUS (BM25 + Vector + RRF + NER cross-refs) mitiga esto capturando coincidencias lexicas que el bi-encoder pierde (nombres propios, fechas) y creando puentes explicitos entre documentos via entidades compartidas.

## Setup y uso

```bash
pip install -r requirements.txt
cp sandbox_mteb/env.example sandbox_mteb/.env
# Editar .env con endpoints NIM y MinIO
```

Para `HYBRID_PLUS` (entity cross-linking):

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

Sin spaCy instalado, `HYBRID_PLUS` funciona pero sin cross-refs (BM25+Vector+RRF puro, warning en log).

```bash
python -m sandbox_mteb.run                  # Run con .env
python -m sandbox_mteb.run --dry-run        # Solo validar config
python -m sandbox_mteb.run --env /path/.env # .env alternativo
python -m sandbox_mteb.run -v               # Verbose (DEBUG)
```

## Configuracion (.env)

Referencia completa en `sandbox_mteb/env.example`. Variables criticas:

```bash
# Embedding (NIM)
EMBEDDING_MODEL_NAME=nvidia/llama-3.2-nv-embedqa-1b-v2
EMBEDDING_BASE_URL=http://<nim-embedding-host>:8000/v1
EMBEDDING_MODEL_TYPE=asymmetric       # asymmetric para NIM (input_type query/passage)
EMBEDDING_BATCH_SIZE=5                # Reducir si NIM da errores de buffer

# LLM (NIM)
LLM_BASE_URL=http://<nim-llm-host>:8000/v1
LLM_MODEL_NAME=nvidia/nemotron-3-nano
NIM_MAX_CONCURRENT_REQUESTS=32        # ATENCION: no NIM_MAX_CONCURRENT
NIM_REQUEST_TIMEOUT=120               # ATENCION: no NIM_TIMEOUT

# Retrieval
RETRIEVAL_STRATEGY=SIMPLE_VECTOR      # SIMPLE_VECTOR | HYBRID_PLUS
RETRIEVAL_K=20                        # Docs para metricas retrieval
RETRIEVAL_PRE_FUSION_K=150            # Candidatos pre-RRF / pool reranker (HYBRID_PLUS)
RETRIEVAL_RRF_K=60                    # Parametro k de RRF (HYBRID_PLUS)
RETRIEVAL_BM25_WEIGHT=0.5             # Peso BM25 en RRF (HYBRID_PLUS)
RETRIEVAL_VECTOR_WEIGHT=0.5           # Peso vector en RRF (HYBRID_PLUS)

# Reranker (opcional)
RERANKER_ENABLED=false
RERANKER_BASE_URL=http://<nim-reranker-host>:9000/v1
RERANKER_MODEL_NAME=nvidia/llama-3.2-nv-rerankqa-1b-v2
RERANKER_TOP_N=5                      # Docs post-rerank para generacion

# Dataset
MTEB_DATASET_NAME=hotpotqa
EVAL_MAX_QUERIES=0                    # 0 = todas
EVAL_MAX_CORPUS=0                     # 0 = todo
GENERATION_ENABLED=true
CORPUS_SHUFFLE_SEED=42                # -1 = sin shuffle (NO recomendado)

# Modo desarrollo
DEV_MODE=false                        # Subset con gold docs garantizados
DEV_QUERIES=200
DEV_CORPUS_SIZE=4000

# Entity cross-linking (solo HYBRID_PLUS)
ENTITY_MAX_CROSS_REFS=3               # Top-N docs referenciados por documento
ENTITY_MIN_SHARED=1                   # Min entidades compartidas para cross-ref
ENTITY_MAX_DOC_FRACTION=0.05          # Umbral IDF (ignorar entidades en >5% corpus)

# MinIO
MINIO_ENDPOINT=http://<minio-host>:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minio123
MINIO_BUCKET_NAME=lakehouse
S3_DATASETS_PREFIX=datasets/evaluation
```

### DEV_MODE

`DEV_MODE=true`: subset de `DEV_QUERIES` queries + gold docs garantizados en corpus + distractores hasta `DEV_CORPUS_SIZE`. Ignora `EVAL_MAX_QUERIES`/`EVAL_MAX_CORPUS`. Metricas optimistas (ratio gold/distractores ~10% vs ~0.003% en corpus completo); solo validas para comparacion relativa.

## Salida

Tres archivos por run en `data/results/`:

- `<run_id>.json`: config + metricas por query + doc_ids recuperados
- `<run_id>_summary.csv`: 1 fila, metricas agregadas
- `<run_id>_detail.csv`: N filas, una por query (incluye columna `question_type` para desglose bridge/comparison)

## Tests

189 unit tests + 3 tests de integracion (192 total). Ejecutables con Python 3.10+.

```bash
pytest tests/                      # Todo junto (unit + integracion)
pytest tests/ -v                   # Verbose
pytest tests/ -m "not integration" # Solo unit
pytest tests/integration/ -v       # Solo integracion
```

Tests relevantes para la estrategia `HYBRID_PLUS`:

| Archivo | Tests | Que cubre |
|---|---|---|
| `test_entity_linker.py` | 31 | Normalizacion, extraccion NER (mock spaCy), indice invertido, IDF filter, cross-refs, pipeline completo, stats |
| `test_hybrid_plus_retriever.py` | 17 | Indexacion con/sin cross-refs, sin LLM enrichment, swap a contenido original, clear_index, factory (sin llm_service) |

**Mocking condicional:** `tests/conftest.py` solo mockea modulos de infraestructura (`boto3`, `langchain_*`, `chromadb`) si el paquete real no esta instalado. En entornos con NIM/MinIO, los modulos reales se preservan y los tests de integracion funcionan junto a los unit tests. En entornos restringidos, se mockean automaticamente y la integracion se salta. Tests de `entity_linker.py` mockean spaCy NER para no depender de que `en_core_web_sm` este descargado.

Detalle de cobertura por archivo, decisiones de mocking y diseno de tests de integracion: ver `README_TEST.md`.

## Deuda tecnica

### Resueltas

| ID | Fix |
|---|---|
| DT-3 | Logging JSONL estructurado (`structured_logging.py`). `LOG_FORMAT=jsonl\|text`. |
| DT-4 | Tipado estricto: `mypy.ini`, `EmbeddingModelProtocol`, `LLMJudgeProtocol`, `py.typed`. 0 errores mypy. |
| DT-5 | `pre_rerank_candidate_ids` en `QueryRetrievalDetail` para trazabilidad reranker. |
| DT-6 | Eliminado truncamiento `context[:4000]` en faithfulness. Judge recibe mismo contexto que LLM de generacion. |
| DT-7 | Reranker fallback detectado: warning + contador `_rerank_failures` + metadata `reranked` en JSON/CSV. |
| DT-8 | `sorted()` explicito por `relevance_score` descendente en reranker. |
| DT-9 | `_extract_score_fallback()` reescrita: 3 patrones explicitos, elimina falsos positivos de parciales. |
| DTm-1 | `_populate_from_dataframes()` extraido en `loader.py`. ~50 lineas duplicadas eliminadas. |
| DTm-2 | `vector_store.py` usa API publica ChromaDB nativo en vez de `_store._collection` (API interna LangChain). |
| DTm-3 | `run_sync()` compatible con event loops activos (Jupyter) via `ThreadPoolExecutor`. |
| DTm-4 | 191 unit + 3 integracion tests. pytest nativo, `conftest.py` centralizado, `pyproject.toml` configurado. |
| DTm-5 | Metricas secundarias fallidas: `MetricResult(value=0.0, error=...)` + warning (no desaparecen del JSON). |
| DTm-6 | Retry con backoff exponencial en `_batch_embed_queries()`. |
| DTm-7 | `_run_async` renombrado a `run_sync`. |
| DTm-8 | `failure_rate_at_k` renombrado a `complement_recall_at_k`. |
| DTm-9 | Docstring sesgo embedding en `get_full_text()`. |
| DTm-10 | `collection_name = f"eval_{run_id}"` (unico, determinista). |
| DTm-11 | `LLMMetrics.__copy__`/`__deepcopy__` crean Lock nuevo. |
| DTm-17 | Metricas de retrieval efectivo (post-rerank): `generation_recall`, `generation_hit`, `reranker_rescue_count`. 15 tests. |

### Abiertas

| ID | Descripcion | Impacto |
|---|---|---|
| DT-2 | Eliminado: enriquecimiento contextual LLM eliminado en favor de NER cross-linking directo. | Resuelto. |
| DTm-12 | Sesgo LLM-judge en faithfulness para respuestas cortas: score 0.0-0.2 incluso con F1=1.0. Confirmado en run 20260223_095004 (distribucion bimodal: 22/65 faith<=0.2, todas respuestas cortas correctas). F1 es metrica primaria y suficiente; faithfulness solo informativa. | Sesgo metrica. Baja prioridad. |
| DTm-13 | No-determinismo HNSW: ChromaDB 0.5-0.6 no soporta `hnsw:random_seed`. Recall@K varia +/-0.02 entre runs con diferente `collection_name`. Mitigacion: fijar seed cuando ChromaDB lo soporte. | Reproducibilidad. Baja prioridad. |
| DTm-14 | Duplicacion contenido memoria: `retrieved_contents` + `generation_contents` (~1.5GB con 7K queries). Memoria suficiente en entorno actual. | Memoria. Baja prioridad. |
| DTm-15 | ETL HotpotQA no asigna `answer_type="label"` a queries comparison (yes/no). El evaluador usa F1 en lugar de Accuracy para estas queries. Sin impacto numerico (tokens unicos: F1 y Accuracy equivalentes), pero `primary_metric_type` en CSV es incorrecto para analisis post-hoc. Corregir en ETL o detectar heuristicamente en el evaluador. | Clasificacion metrica. Baja prioridad. |
| DTm-16 | Nemotron-3-nano responde "yes" a preguntas extractivas (~10% de queries en run 20260223_095004). El system prompt "For yes/no questions, start with yes or no" causa sobregeneralizacion en modelos pequenos. Deprime avg F1 en ~0.10 puntos. Mitigaciones: (a) condicionar instruccion yes/no por `answer_type` del query, (b) eliminar instruccion y delegar clasificacion al evaluador post-hoc, (c) usar modelo mas capaz. Referencia: `GENERATION_PROMPTS` en `sandbox_mteb/config.py`. | Calidad generacion. Media prioridad. |
| DTm-18 | Entity normalization basica (lowercase + articulos). No resuelve aliases (US/United States) ni formas parciales (Derrickson/Scott Derrickson). Mejora futura: entity linking contra Wikidata o embedding similarity entre entidades. | Calidad cross-refs. Baja prioridad. |
| DTm-19 | Eliminado: `ContextualRetriever` y `ContextualRetrieverPlus` eliminados. `HybridPlusRetriever` es autocontenido (~170 lineas). | Resuelto. |
| DTm-20 | `question_type` en detail CSV requiere propagacion manual desde query metadata en `evaluator.py`. Considerar un mecanismo generico de metadata passthrough en `QueryEvaluationResult`. | Extensibilidad. Baja prioridad. |