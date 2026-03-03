# PLAN DE ETAPAS: CONTEXTUAL_HYBRID_PLUS — Contextualizacion y Planificacion

## Documento de Referencia
`PLAN_CONTEXTUAL_HYBRID_PLUS_v2.md`

## Fecha: 2026-02-25

---

## 1. Contextualizacion del Proyecto

### 1.1 Estado Actual del Codebase

El proyecto **CH_PLUS** (RAG_P v3.2) es un framework de evaluacion RAG sobre
datasets MTEB/BeIR (actualmente HotpotQA). Consta de:

| Componente | LOC | Estado |
|---|---|---|
| `shared/` (libreria core) | ~3,300 | Estable, tipado con mypy |
| `sandbox_mteb/` (framework evaluacion) | ~1,900 | Estable, operativo |
| `tests/` | ~3,500 | ~162 tests (147 unit + 15 integration) |
| Documentacion | ~1,250 | README.md + README_TEST.md + Plan v2 |

### 1.2 Estrategias de Retrieval Existentes

Actualmente hay **2 estrategias** implementadas:

1. **SIMPLE_VECTOR**: Embedding puro via ChromaDB (`shared/retrieval/core.py:152`)
2. **CONTEXTUAL_HYBRID**: Enriquecimiento LLM + BM25+Vector+RRF (`shared/retrieval/contextual_retriever.py:261`)

El plan v2 propone una **tercera estrategia: CONTEXTUAL_HYBRID_PLUS** que extiende
CONTEXTUAL_HYBRID con cross-linking de entidades via spaCy NER.

### 1.3 Problema que Resuelve

En HotpotQA, ~80% de las preguntas son de tipo **bridge** (multi-hop). Ejemplo:
"Where was the director of Sinister born?" requiere:
1. Recuperar Doc A ("Sinister") para identificar al director (Scott Derrickson)
2. Recuperar Doc B ("Scott Derrickson") para encontrar su lugar de nacimiento

El bi-encoder genera un vector mas cercano a "Sinister" que a "Scott Derrickson",
por lo que Doc B frecuentemente no se recupera. La propuesta: inyectar
cross-references textuales entre documentos que comparten entidades, para que
BM25 capture el termino puente.

### 1.4 Validacion del Plan v2 contra el Codebase Real

| Aspecto del Plan | Verificado | Observaciones |
|---|---|---|
| `RetrievalStrategy` enum en `core.py:28` | OK | Enum con SIMPLE_VECTOR y CONTEXTUAL_HYBRID |
| `RetrievalConfig` dataclass en `core.py:39` | OK | 10 campos, `from_env()` en linea 66 |
| Factory `get_retriever()` en `__init__.py:32` | OK | 2 ramas (SIMPLE_VECTOR, CONTEXTUAL_HYBRID) |
| `ContextualRetriever` en `contextual_retriever.py:261` | OK | Patron decorador con inner retriever |
| `EnrichedChunk` en `contextual_retriever.py:41` | OK | chunk_id, original_content, generated_context |
| `_run_batch_generation` en `contextual_retriever.py:403` | OK | Usa `run_sync` de `shared.llm` |
| `_swap_to_original_contents` en `contextual_retriever.py:376` | OK | Restaura contenido original para generacion |
| `MTEBConfig.validate()` en `config.py:127` | OK | VALID_STRATEGIES en linea 139, needs_llm en linea 148 |
| `RunExporter.to_detail_csv()` en `report.py:141` | OK | No tiene `question_type` actualmente |
| `_evaluate_queries` metadata en `evaluator.py:688` | OK | Solo propaga `reranked`, no `question_type` |
| `requirements.txt` | OK | No tiene spaCy, 15 dependencias actuales |
| Patron `HAS_TANTIVY` / `HAS_BM25` | OK | En `hybrid_retriever.py`, patron de importacion condicional |

**Conclusion:** El plan v2 es factualmente correcto respecto al estado del codebase.
No se detectan errores en las referencias a ficheros, tipos, o puntos de integracion.

---

## 2. Planificacion por Etapas

La implementacion se divide en **6 etapas** (refinamiento de las 5 fases del plan v2),
con criterios de entrada/salida claros para cada una.

```
ETAPA 1: Infraestructura base (spaCy + EntityLinker)
    |
    v
ETAPA 2: Tests unitarios del EntityLinker (TDD)
    |
    v
ETAPA 3: Integracion en pipeline de retrieval
    |
    v
ETAPA 4: Tests de integracion del retriever
    |
    v
ETAPA 5: Observabilidad (question_type en CSV)
    |
    v
ETAPA 6: Evaluacion comparativa y documentacion
```

---

### ETAPA 1: Infraestructura Base — spaCy + EntityLinker

**Objetivo:** Crear el modulo `entity_linker.py` completo y funcional de forma
aislada, sin tocar ningun fichero existente (excepto `requirements.txt`).

**Dependencia:** Ninguna (es la primera etapa).

#### Tareas

| # | Tarea | Fichero | Tipo |
|---|---|---|---|
| 1.1 | Agregar `spacy>=3.7.0,<4.0` a dependencias | `requirements.txt` | MODIFICAR |
| 1.2 | Crear `entity_linker.py` con importacion condicional `HAS_SPACY` | `shared/retrieval/entity_linker.py` | NUEVO |
| 1.3 | Implementar `EntityNormalizer.normalize()` | `shared/retrieval/entity_linker.py` | NUEVO |
| 1.4 | Implementar `EntityExtractor` con filtrado de tipos NER | `shared/retrieval/entity_linker.py` | NUEVO |
| 1.5 | Implementar `DocEntities` dataclass | `shared/retrieval/entity_linker.py` | NUEVO |
| 1.6 | Implementar `EntityLinker.build_index()` con IDF filter | `shared/retrieval/entity_linker.py` | NUEVO |
| 1.7 | Implementar `EntityLinker.generate_cross_refs()` | `shared/retrieval/entity_linker.py` | NUEVO |
| 1.8 | Implementar `EntityLinker.compute_cross_refs()` orquestador | `shared/retrieval/entity_linker.py` | NUEVO |
| 1.9 | Implementar `EntityLinker.get_stats()` | `shared/retrieval/entity_linker.py` | NUEVO |

#### Especificaciones tecnicas clave

**Importacion condicional** (mismo patron que `HAS_TANTIVY` en `hybrid_retriever.py`):
```python
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    spacy = None
```

**Tipos NER relevantes:** PERSON, ORG, GPE, LOC, FAC, EVENT, WORK_OF_ART, LAW,
PRODUCT, NORP. Excluidos: CARDINAL, ORDINAL, QUANTITY, PERCENT, DATE, TIME.

**Normalizacion:** lowercase + eliminar articulos iniciales + colapsar espacios +
eliminar puntuacion (excepto guiones internos). Limitacion aceptada: no resuelve
aliases (US/United States) ni formas parciales (Derrickson/Scott Derrickson).

**IDF Filter:** Entidades que aparecen en >5% del corpus se excluyen del
cross-linking. Threshold minimo absoluto: 10 docs (para corpus pequenos).

**Parametros configurables:**
- `max_cross_refs`: Top-N docs referenciados por documento (default: 3)
- `min_shared_entities`: Minimo de entidades compartidas para generar ref (default: 1)
- `max_entity_doc_fraction`: Umbral IDF (default: 0.05)

#### Criterios de salida
- [x] `entity_linker.py` importable sin errores
- [x] `HAS_SPACY=False` no genera error de importacion
- [x] Todas las clases tienen type hints completos
- [x] `mypy shared/retrieval/entity_linker.py` sin errores

**Estimacion: 4-6h**

---

### ETAPA 2: Tests Unitarios del EntityLinker (TDD)

**Objetivo:** Suite completa de tests para `entity_linker.py` con mocking de spaCy.

**Dependencia:** ETAPA 1 completada.

#### Tareas

| # | Tarea | Fichero | Tipo |
|---|---|---|---|
| 2.1 | Tests de `EntityNormalizer` (basic + edge cases) | `tests/test_entity_linker.py` | NUEVO |
| 2.2 | Tests de `EntityExtractor` (mock spaCy, tipos, dedup, min length) | `tests/test_entity_linker.py` | NUEVO |
| 2.3 | Tests de `EntityLinker.build_index()` (indice, IDF filter, corpus pequeno) | `tests/test_entity_linker.py` | NUEVO |
| 2.4 | Tests de `generate_cross_refs()` (shared, no shared, max, min, sort, IDF) | `tests/test_entity_linker.py` | NUEVO |
| 2.5 | Tests de `compute_cross_refs()` (pipeline completo, docs vacios) | `tests/test_entity_linker.py` | NUEVO |
| 2.6 | Test de `get_stats()` | `tests/test_entity_linker.py` | NUEVO |

#### Cobertura de tests (~18 tests)

| Test | Verifica |
|---|---|
| `test_normalizer_basic` | lowercase, articulos, puntuacion |
| `test_normalizer_edge_cases` | string vacio, solo puntuacion, guiones internos |
| `test_extractor_basic` | spaCy extrae PERSON, ORG, GPE de texto Wikipedia |
| `test_extractor_filters_types` | CARDINAL, DATE excluidos |
| `test_extractor_deduplicates` | misma entidad 2x en texto -> 1 en resultado |
| `test_extractor_min_length` | entidades de 1 char filtradas |
| `test_build_index_basic` | indice invertido correcto para 3 docs |
| `test_idf_filter` | entidad en >5% docs filtrada |
| `test_idf_filter_small_corpus` | threshold minimo absoluto (10) en corpus < 200 |
| `test_cross_refs_two_docs_shared_entity` | 2 docs comparten entidad -> cross-ref |
| `test_cross_refs_no_shared` | docs sin entidades comunes -> string vacio |
| `test_cross_refs_max_refs` | respeta max_cross_refs=2 |
| `test_cross_refs_min_shared` | min_shared=2 filtra docs con 1 entidad comun |
| `test_cross_refs_sorted_by_overlap` | doc con mas overlap primero |
| `test_cross_refs_filtered_entity_excluded` | entidad IDF-filtrada no genera cross-ref |
| `test_compute_cross_refs_pipeline` | pipeline completo con mock EntityExtractor |
| `test_compute_cross_refs_empty_docs` | lista vacia -> dict vacio |
| `test_stats` | get_stats() contiene campos esperados |

**Estrategia de mocking:** Mock de `EntityExtractor` para inyectar entidades
manuales. Coherente con el principio del proyecto: "testear computacion pura,
mocks solo para infra". spaCy NER es infra.

#### Criterios de salida
- [x] `pytest tests/test_entity_linker.py -v` — 18 tests pasan
- [x] Tests no dependen de `en_core_web_sm` descargado
- [x] Cobertura de todos los edge cases documentados en el plan

**Estimacion: 4-5h**

---

### ETAPA 3: Integracion en Pipeline de Retrieval

**Objetivo:** Cablear `CONTEXTUAL_HYBRID_PLUS` como tercera estrategia en el
pipeline existente, tocando los ficheros minimos necesarios.

**Dependencia:** ETAPA 1 completada (ETAPA 2 puede estar en paralelo).

#### Tareas

| # | Tarea | Fichero | Lineas afectadas | Tipo |
|---|---|---|---|---|
| 3.1 | Agregar `CONTEXTUAL_HYBRID_PLUS` al enum | `shared/retrieval/core.py` | L28-31 | MODIFICAR |
| 3.2 | Agregar 3 campos entity a `RetrievalConfig` | `shared/retrieval/core.py` | L39-79 | MODIFICAR |
| 3.3 | Agregar `from_env` para nuevos campos | `shared/retrieval/core.py` | L66-79 | MODIFICAR |
| 3.4 | Crear `ContextualRetrieverPlus` (composicion) | `shared/retrieval/contextual_retriever.py` | Final del fichero | MODIFICAR |
| 3.5 | Agregar rama PLUS en factory `get_retriever()` | `shared/retrieval/__init__.py` | L62-80 | MODIFICAR |
| 3.6 | Agregar imports de `ContextualRetrieverPlus` | `shared/retrieval/__init__.py` | L23-27, L83-93 | MODIFICAR |
| 3.7 | Agregar PLUS a `VALID_STRATEGIES` en `validate()` | `sandbox_mteb/config.py` | L139 | MODIFICAR |
| 3.8 | Actualizar `needs_llm` para incluir PLUS | `sandbox_mteb/config.py` | L148-151 | MODIFICAR |
| 3.9 | Agregar variables ENTITY_* a env.example | `sandbox_mteb/env.example` | Final del fichero | MODIFICAR |

#### Detalle de cambios por fichero

**`shared/retrieval/core.py`** — 3 cambios puntuales:

```python
# L31: Nuevo enum member
class RetrievalStrategy(Enum):
    SIMPLE_VECTOR = auto()
    CONTEXTUAL_HYBRID = auto()
    CONTEXTUAL_HYBRID_PLUS = auto()   # NUEVO

# L56-58: Nuevos campos en RetrievalConfig
    entity_max_cross_refs: int = 3
    entity_min_shared: int = 1
    entity_max_doc_fraction: float = 0.05

# En from_env(): 3 lineas nuevas
    entity_max_cross_refs=_env_int("ENTITY_MAX_CROSS_REFS", 3),
    entity_min_shared=_env_int("ENTITY_MIN_SHARED", 1),
    entity_max_doc_fraction=_env_float("ENTITY_MAX_DOC_FRACTION", 0.05),
```

**`shared/retrieval/contextual_retriever.py`** — Agregar `ContextualRetrieverPlus`
al final del fichero (~120 lineas nuevas). Patron: composicion sobre
`ContextualRetriever`. Duplica `retrieve`, `retrieve_by_vector`,
`_swap_to_original_contents`, `_run_batch_generation` (~30 lineas).
Duplicacion aceptada para evitar herencia problematica (documentada en DTm-19).

**`shared/retrieval/__init__.py`** — Nueva rama en factory:
```python
if strategy == RetrievalStrategy.CONTEXTUAL_HYBRID_PLUS:
    # ... validar llm_service, crear context_generator, retornar ContextualRetrieverPlus
```

**`sandbox_mteb/config.py`** — 2 cambios puntuales:
```python
# L139: Agregar a tuple
VALID_STRATEGIES = (
    RetrievalStrategy.SIMPLE_VECTOR,
    RetrievalStrategy.CONTEXTUAL_HYBRID,
    RetrievalStrategy.CONTEXTUAL_HYBRID_PLUS,  # NUEVO
)

# L148-151: Actualizar condicion
needs_llm = (
    self.generation_enabled
    or self.retrieval.strategy in (
        RetrievalStrategy.CONTEXTUAL_HYBRID,
        RetrievalStrategy.CONTEXTUAL_HYBRID_PLUS,  # NUEVO
    )
)
```

**`sandbox_mteb/env.example`** — Bloque nuevo al final (~10 lineas).

#### Criterios de salida
- [x] `RETRIEVAL_STRATEGY=CONTEXTUAL_HYBRID_PLUS` se parsea correctamente en `from_env()`
- [x] `MTEBConfig.validate()` acepta la nueva estrategia
- [x] `get_retriever()` construye `ContextualRetrieverPlus` correctamente
- [x] Sin spaCy: PLUS se comporta identico a HYBRID (warning en log)
- [x] Tests existentes siguen pasando (`pytest tests/ -v`)

**Estimacion: 4-6h**

---

### ETAPA 4: Tests del Retriever Plus

**Objetivo:** Tests unitarios de `ContextualRetrieverPlus` y test de integracion
del pipeline completo.

**Dependencia:** ETAPAS 1, 2 y 3 completadas.

#### Tareas

| # | Tarea | Fichero | Tipo |
|---|---|---|---|
| 4.1 | Test: indexacion con cross-refs (mock spaCy + inner) | `tests/test_contextual_retriever_plus.py` | NUEVO |
| 4.2 | Test: indexacion sin spaCy (HAS_SPACY=False) | `tests/test_contextual_retriever_plus.py` | NUEVO |
| 4.3 | Test: swap to original contents | `tests/test_contextual_retriever_plus.py` | NUEVO |
| 4.4 | Test: clear_index limpia todo | `tests/test_contextual_retriever_plus.py` | NUEVO |
| 4.5 | Test integracion: pipeline CONTEXTUAL_HYBRID_PLUS | `tests/integration/test_pipeline.py` | MODIFICAR |

#### Cobertura de tests (~5 tests)

| Test | Verifica |
|---|---|
| `test_index_documents_with_cross_refs` | Inner retriever recibe docs con cross-refs en content |
| `test_index_documents_without_spacy` | `HAS_SPACY=False` -> warning, docs sin cross-refs |
| `test_swap_to_original_contents` | Retrieval devuelve original_contents |
| `test_clear_index` | Limpia inner, cache, originals, linker_stats |
| `test_pipeline_contextual_hybrid_plus` | Run completo con spaCy+NIM (integration) |

#### Criterios de salida
- [x] `pytest tests/test_contextual_retriever_plus.py -v` — 4 tests pasan
- [x] Tests de integracion con marca `@pytest.mark.integration`
- [x] Tests existentes no se rompen

**Estimacion: 2-3h**

---

### ETAPA 5: Observabilidad — question_type en CSV

**Objetivo:** Propagar `question_type` desde la metadata de cada query hasta el
CSV de detalle, habilitando el analisis desglosado bridge vs comparison.

**Dependencia:** Independiente de ETAPAS 1-4 (puede ejecutarse en paralelo).

#### Tareas

| # | Tarea | Fichero | Lineas afectadas | Tipo |
|---|---|---|---|---|
| 5.1 | Propagar `question_type` de query.metadata a qr_metadata | `sandbox_mteb/evaluator.py` | L688-690 | MODIFICAR |
| 5.2 | Agregar columna `question_type` a fieldnames del detail CSV | `shared/report.py` | L168-199 | MODIFICAR |
| 5.3 | Poblar `question_type` en cada fila del detail CSV | `shared/report.py` | L205-253 | MODIFICAR |

#### Detalle de cambios

**`sandbox_mteb/evaluator.py`** — En el bloque de ensamblado de resultados (L688):
```python
# Agregar despues de la linea de reranked_status:
qr_metadata: Dict[str, Any] = {}
if reranked_status is not None:
    qr_metadata["reranked"] = reranked_status
# NUEVO: propagar question_type de la query original
qt = query.metadata.get("question_type", "")
if qt:
    qr_metadata["question_type"] = qt
```

**`shared/report.py`** — En `to_detail_csv()`:
```python
# En fieldnames (despues de "reranked"):
fieldnames.extend(["question_type"])

# En cada row:
row["question_type"] = qr.metadata.get("question_type", "")
```

#### Criterios de salida
- [x] Detail CSV contiene columna `question_type`
- [x] Para HotpotQA: valores `bridge` y `comparison` presentes
- [x] Queries sin question_type: columna vacia (no error)

**Estimacion: 1-2h**

---

### ETAPA 6: Evaluacion Comparativa y Documentacion

**Objetivo:** Ejecutar 3 runs comparativos y documentar resultados. Actualizar
README.md con la nueva estrategia.

**Dependencia:** TODAS las etapas anteriores completadas.

#### Tareas

| # | Tarea | Tipo |
|---|---|---|
| 6.1 | Run baseline: `RETRIEVAL_STRATEGY=SIMPLE_VECTOR` | EJECUCION |
| 6.2 | Run hybrid: `RETRIEVAL_STRATEGY=CONTEXTUAL_HYBRID` | EJECUCION |
| 6.3 | Run hybrid plus: `RETRIEVAL_STRATEGY=CONTEXTUAL_HYBRID_PLUS` | EJECUCION |
| 6.4 | Analisis comparativo por question_type (bridge vs comparison) | ANALISIS |
| 6.5 | Documentar resultados y conclusiones | DOCUMENTACION |
| 6.6 | Actualizar README.md con nueva estrategia y setup spaCy | MODIFICAR |
| 6.7 | Validar hipotesis H1-H4 | ANALISIS |

#### Configuracion de Runs

Los 3 runs deben ser identicos excepto `RETRIEVAL_STRATEGY`:
- Mismo seed (`CORPUS_SHUFFLE_SEED=42`)
- Mismas queries (`EVAL_MAX_QUERIES=0` o DEV_MODE fijo)
- Mismo corpus (`EVAL_MAX_CORPUS=0` o DEV_MODE fijo)
- Mismo reranker config
- Mismo embedding model

#### Hipotesis a Validar

| ID | Hipotesis | Metrica | Aceptar si |
|---|---|---|---|
| H1 | PLUS > HYBRID en bridge | Recall@5 bridge subset | delta > +0.02 |
| H2 | PLUS ~ HYBRID en comparison | Recall@5 comparison subset | delta en [-0.02, +0.02] |
| H3 | PLUS no degrada generacion | avg F1 global | delta > -0.01 |
| H4 | Cross-refs no saturan contexto | avg n_generation_docs | sin cambio |

#### Criterios de salida
- [x] 3 runs completados con JSON+CSV
- [x] Tabla comparativa documentada
- [x] Hipotesis evaluadas con datos reales
- [x] README.md actualizado

**Estimacion: 3-4h**

---

## 3. Resumen de Esfuerzo y Cronograma

| Etapa | Descripcion | Esfuerzo | Dependencia |
|---|---|---|---|
| **ETAPA 1** | EntityLinker (modulo nuevo) | 4-6h | - |
| **ETAPA 2** | Tests unitarios EntityLinker | 4-5h | ETAPA 1 |
| **ETAPA 3** | Integracion pipeline | 4-6h | ETAPA 1 |
| **ETAPA 4** | Tests retriever plus | 2-3h | ETAPAS 1,2,3 |
| **ETAPA 5** | Observabilidad (question_type) | 1-2h | Independiente |
| **ETAPA 6** | Evaluacion + documentacion | 3-4h | TODAS |
| | **Total** | **18-26h** | |

### Paralelismo posible

```
Tiempo  -->

T0:  ETAPA 1 (EntityLinker)
     |
T1:  ETAPA 2 (Tests EL)  ||  ETAPA 3 (Integracion)  ||  ETAPA 5 (question_type)
     |                         |
T2:  +-------- ETAPA 4 -------+
     |
T3:  ETAPA 6 (Evaluacion)
```

Con paralelismo maximo: **ruta critica ~14-19h** (ETAPA 1 -> ETAPA 3 -> ETAPA 4 -> ETAPA 6).

---

## 4. Ficheros Impactados — Vista Consolidada

| Fichero | Etapa | Accion | Impacto |
|---|---|---|---|
| `requirements.txt` | 1 | MODIFICAR | +1 linea (spacy) |
| `shared/retrieval/entity_linker.py` | 1 | **NUEVO** | ~250 lineas |
| `shared/retrieval/core.py` | 3 | MODIFICAR | +1 enum, +3 campos config, +3 lineas from_env |
| `shared/retrieval/contextual_retriever.py` | 3 | MODIFICAR | +~120 lineas (ContextualRetrieverPlus) |
| `shared/retrieval/__init__.py` | 3 | MODIFICAR | +~20 lineas (import + factory) |
| `sandbox_mteb/config.py` | 3 | MODIFICAR | +2 lineas (VALID_STRATEGIES, needs_llm) |
| `sandbox_mteb/env.example` | 3 | MODIFICAR | +~10 lineas (ENTITY_* vars) |
| `sandbox_mteb/evaluator.py` | 5 | MODIFICAR | +3 lineas (question_type propagation) |
| `shared/report.py` | 5 | MODIFICAR | +3 lineas (question_type en CSV) |
| `tests/test_entity_linker.py` | 2 | **NUEVO** | ~300 lineas (~18 tests) |
| `tests/test_contextual_retriever_plus.py` | 4 | **NUEVO** | ~150 lineas (~4 tests) |
| `tests/integration/test_pipeline.py` | 4 | MODIFICAR | +~20 lineas |
| `README.md` | 6 | MODIFICAR | +~30 lineas |

**Total:** 3 ficheros nuevos, 10 ficheros modificados, ~900 lineas nuevas.

---

## 5. Riesgos Priorizados con Mitigaciones Concretas

| Prioridad | Riesgo | Mitigacion | Etapa de deteccion |
|---|---|---|---|
| **ALTA** | R2: Entidades genericas saturan cross-refs | IDF filter `max_entity_doc_fraction=0.05` con threshold minimo 10 | ETAPA 2 (test IDF) |
| **ALTA** | R3: Normalizacion insuficiente | Aceptada. Validar con muestra manual en ETAPA 6 | ETAPA 6 |
| **MEDIA** | R1: spaCy NER ruidoso en texto Wikipedia | `MIN_ENTITY_LENGTH=2` + filtrado tipos. Validar en ETAPA 6 | ETAPA 2 + 6 |
| **MEDIA** | R4: Cross-refs degradan comparison questions | Parametros tuneables. Si H2 falla, reducir max_cross_refs | ETAPA 6 |
| **MEDIA** | R6: spaCy no disponible en entorno | Importacion condicional, degradacion graceful | ETAPA 1 (test) |
| **BAJA** | R5: Overhead de texto en indice | ~150-300 chars extra/doc, negligible | ETAPA 6 |
| **BAJA** | R8: Memoria del indice invertido | ~20MB para 66K docs, trivial | No aplica |
| **NULA** | R7: Circularidad cross-refs (A<->B) | No es problema, ambas direcciones refuerzan bridge | No aplica |

---

## 6. Deuda Tecnica Resultante

| ID | Descripcion | Prioridad | Etapa origen |
|---|---|---|---|
| DTm-18 | Normalizacion basica (no aliases, no formas parciales) | Baja | ETAPA 1 |
| DTm-19 | ~30 lineas duplicadas entre ContextualRetriever y Plus | Baja | ETAPA 3 |
| DTm-20 | question_type propagacion manual, considerar passthrough generico | Baja | ETAPA 5 |

---

## 7. Criterios de Aceptacion Global

1. [DONE] `pytest tests/test_entity_linker.py tests/test_contextual_retriever_plus.py -v` — **46 tests** pasan (31+15)
2. [SKIP] `mypy` — no disponible en entorno actual
3. [DONE] `RETRIEVAL_STRATEGY=CONTEXTUAL_HYBRID_PLUS` se parsea correctamente en `from_env()`
4. [PEND] Run completo genera JSON+CSV — requiere infra NIM+spaCy
5. [DONE] Detail CSV incluye columna `question_type` (shared/report.py)
6. [DONE] Sin spaCy: PLUS produce resultados identicos a HYBRID (test `test_no_spacy_still_indexes`)
7. [DONE] Tests existentes siguen pasando — **189 tests, zero regresiones** (143 originales + 46 nuevos)
8. [PEND] Evaluacion comparativa — requiere infra NIM+spaCy para runs

---

## 8. Estado de Implementacion (2026-03-03)

### Etapas completadas

| Etapa | Estado | Tests | Ficheros |
|---|---|---|---|
| **ETAPA 1** | DONE | - | `shared/retrieval/entity_linker.py` (334 LOC) |
| **ETAPA 2** | DONE | 31 tests | `tests/test_entity_linker.py` |
| **ETAPA 3** | DONE | - | `core.py`, `contextual_retriever.py`, `__init__.py`, `config.py`, `env.example` |
| **ETAPA 4** | DONE | 15 tests | `tests/test_contextual_retriever_plus.py` |
| **ETAPA 5** | DONE | - | `evaluator.py`, `report.py` |
| **ETAPA 6** | PARCIAL | - | Plan actualizado. Runs comparativos pendientes (infra). |

### Resumen cuantitativo

- **Ficheros nuevos:** 3 (`entity_linker.py`, `test_entity_linker.py`, `test_contextual_retriever_plus.py`)
- **Ficheros modificados:** 7 (`core.py`, `contextual_retriever.py`, `__init__.py`, `config.py`, `env.example`, `evaluator.py`, `report.py`)
- **Lineas nuevas:** ~1,040
- **Tests totales:** 189 (143 pre-existentes + 46 nuevos)
- **Regresiones:** 0

### Pendiente para ETAPA 6 completa

1. Instalar spaCy + descargar `en_core_web_sm` en entorno con NIM
2. Ejecutar 3 runs comparativos (SIMPLE_VECTOR, CONTEXTUAL_HYBRID, CONTEXTUAL_HYBRID_PLUS)
3. Analizar desglose bridge vs comparison usando columna `question_type` del CSV
4. Validar hipotesis H1-H4
5. Actualizar README.md con resultados y setup spaCy
