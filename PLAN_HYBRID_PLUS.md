# PLAN: Simplificacion a HYBRID_PLUS — Eliminacion de CONTEXTUAL_HYBRID

## Fecha: 2026-03-04

---

## 1. Motivacion

`CONTEXTUAL_HYBRID_PLUS` actual ejecuta dos pasos de enriquecimiento durante
indexacion:

1. **LLM enrichment** (~minutos): cada doc se envia a nemotron-3-nano para generar
   contexto semantico. Herencia directa de `CONTEXTUAL_HYBRID`.
2. **NER cross-linking** (~segundos): spaCy extrae entidades y genera cross-refs
   textuales entre documentos con entidades compartidas.

El paso 1 es el cuello de botella de indexacion y existe para mejorar BM25/vector
en general. El paso 2 ataca directamente el problema multi-hop (bridge questions).

**Pregunta:** si el cross-linking ya crea los puentes lexicos necesarios para
bridge questions, el LLM enrichment aporta lo suficiente para justificar su
coste (latencia, dependencia de NIM para indexacion)?

**Decision:** eliminar `CONTEXTUAL_HYBRID` y su LLM enrichment del proyecto.
La nueva estrategia `HYBRID_PLUS` combina BM25+Vector+RRF con NER cross-linking
sin dependencia de LLM para indexacion.

---

## 2. Arquitectura Objetivo

### Antes (3 estrategias)

```
SIMPLE_VECTOR:           Doc -> Embedding -> ChromaDB
CONTEXTUAL_HYBRID:       Doc -> LLM enrich -> Embedding+BM25 -> RRF
CONTEXTUAL_HYBRID_PLUS:  Doc -> LLM enrich -> NER cross-refs -> Embedding+BM25 -> RRF
```

### Despues (2 estrategias)

```
SIMPLE_VECTOR:   Doc -> Embedding -> ChromaDB
HYBRID_PLUS:     Doc -> NER cross-refs -> Embedding+BM25 -> RRF
```

### Flujo HYBRID_PLUS

```
FASE INDEXACION:

  [Docs originales]
        |
        v
  [spaCy NER: extrae entidades]
        |
        v
  [EntityLinker: indice invertido + IDF filter]
        |
        v
  [Cross-refs textuales por doc]
        |
        v
  [Doc original + Cross-refs] --> [HybridRetriever: Index BM25+Vector]


FASE RETRIEVAL (sin cambios):

  [Query] --> [BM25+Vector+RRF] --> [Top-K] --> [Reranker] --> [LLM]


FASE GENERACION:

  _swap_to_original_contents() devuelve texto original (sin cross-refs)
  al LLM de generacion.
```

### Ventajas

- **Eliminacion del cuello de botella:** NER tarda segundos, LLM enrichment tardaba
  minutos. Indexacion pasa de ~5min a ~30s para 8500 docs.
- **Sin dependencia de NIM para indexacion:** solo necesitas NIM para generacion
  y reranker. Indexacion funciona offline con solo spaCy.
- **Simplicidad:** eliminamos ~400 lineas de codigo (ContextualRetriever,
  LLMContextGenerator, EnrichedChunk, toda la logica de batch LLM).
- **Menos config:** eliminamos `RETRIEVAL_CONTEXT_MAX_TOKENS`,
  `RETRIEVAL_CONTEXT_BATCH_SIZE`. Menos variables en `.env`.

---

## 3. Codigo a ELIMINAR

| Fichero | Que se elimina | LOC aprox |
|---|---|---|
| `shared/retrieval/contextual_retriever.py` | `EnrichedChunk`, `LLMContextGenerator`, `ContextualRetriever` (todo excepto `ContextualRetrieverPlus` que se reescribe) | ~400 |
| `shared/retrieval/__init__.py` | Imports de `ContextualRetriever`, `LLMContextGenerator`, `EnrichedChunk`. Rama factory `CONTEXTUAL_HYBRID`. | ~25 |
| `shared/retrieval/core.py` | Enum `CONTEXTUAL_HYBRID`, `CONTEXTUAL_HYBRID_PLUS`. Campos `context_max_tokens`, `context_batch_size`. | ~10 |
| `sandbox_mteb/config.py` | `CONTEXTUAL_HYBRID` de VALID_STRATEGIES y `_contextual_strategies`. | ~5 |
| `sandbox_mteb/evaluator.py` | Logica `needs_llm` para estrategia de retrieval. | ~3 |
| `sandbox_mteb/env.example` | Seccion `CONTEXTUAL RETRIEVAL`, refs a CONTEXTUAL_HYBRID. | ~10 |
| `tests/test_contextual_retriever_plus.py` | Reescribir completo (ya no hay LLMContextGenerator). | rewrite |

**Total eliminado: ~450 lineas de codigo de produccion.**

---

## 4. Codigo NUEVO / MODIFICADO

### 4.1 `shared/retrieval/hybrid_plus_retriever.py` — NUEVO (~120 lineas)

Retriever limpio que combina HybridRetriever + EntityLinker. Sin herencia de
ContextualRetriever. Sin LLMContextGenerator. Sin EnrichedChunk.

```python
class HybridPlusRetriever(BaseRetriever):
    """
    Hybrid retrieval (BM25+Vector+RRF) con entity cross-linking (spaCy NER).

    Flujo index_documents:
      1. NER sobre contenido original de cada doc (si spaCy disponible)
      2. EntityLinker.build_index() -> cross-refs por doc
      3. Texto indexado = contenido original + cross-refs
      4. HybridRetriever indexa texto final

    Retrieval: delega a HybridRetriever, swap a original_contents.
    Sin LLM enrichment. Sin dependencia de NIM para indexacion.
    """

    def __init__(
        self,
        config: RetrievalConfig,
        embedding_model: EmbeddingModelProtocol,
        collection_name: Optional[str] = None,
        embedding_batch_size: int = 0,
        max_cross_refs: int = 3,
        min_shared_entities: int = 1,
        max_entity_doc_fraction: float = 0.05,
    ):
        # Inner: HybridRetriever (BM25+Vector+RRF)
        # Entity linker config
        # _original_contents: Dict[str, str] para swap durante generacion

    def index_documents(self, documents, collection_name=None) -> bool:
        # 1. NER + cross-linking (si HAS_SPACY)
        # 2. Construir docs: original + cross-refs
        # 3. Guardar _original_contents[doc_id] = original
        # 4. inner_retriever.index_documents(enriched_docs)

    def retrieve(self, query, top_k=None) -> RetrievalResult:
        # Delega a inner, swap a original_contents

    def retrieve_by_vector(self, query_text, query_vector, top_k=None) -> RetrievalResult:
        # Delega a inner, swap a original_contents

    def _swap_to_original_contents(self, result) -> RetrievalResult:
        # Reemplaza contents enriquecidos por originales
        # strategy_used = RetrievalStrategy.HYBRID_PLUS

    def clear_index(self) -> None:
        # Limpia inner + _original_contents + stats
```

### 4.2 `shared/retrieval/core.py` — MODIFICAR

```python
class RetrievalStrategy(Enum):
    SIMPLE_VECTOR = auto()
    HYBRID_PLUS = auto()           # antes: CONTEXTUAL_HYBRID + CONTEXTUAL_HYBRID_PLUS

# RetrievalConfig: eliminar context_max_tokens, context_batch_size
# from_env(): eliminar RETRIEVAL_CONTEXT_MAX_TOKENS, RETRIEVAL_CONTEXT_BATCH_SIZE
```

### 4.3 `shared/retrieval/contextual_retriever.py` — ELIMINAR

Fichero completo. 607 lineas eliminadas.

### 4.4 `shared/retrieval/__init__.py` — MODIFICAR

```python
# Eliminar imports de ContextualRetriever, LLMContextGenerator, EnrichedChunk
# Nuevo import:
from .hybrid_plus_retriever import HybridPlusRetriever

# Factory: 2 ramas (SIMPLE_VECTOR, HYBRID_PLUS)
# HYBRID_PLUS ya NO requiere llm_service
def get_retriever(...):
    if strategy == RetrievalStrategy.SIMPLE_VECTOR:
        return SimpleVectorRetriever(...)

    if strategy == RetrievalStrategy.HYBRID_PLUS:
        return HybridPlusRetriever(
            config=config,
            embedding_model=embedding_model,
            collection_name=collection_name,
            embedding_batch_size=embedding_batch_size,
            max_cross_refs=config.entity_max_cross_refs,
            min_shared_entities=config.entity_min_shared,
            max_entity_doc_fraction=config.entity_max_doc_fraction,
        )
```

**Cambio critico en factory:** `llm_service` ya no se necesita para construir el
retriever. Solo lo necesita el evaluador para generacion.

### 4.5 `shared/retrieval/hybrid_retriever.py` — MODIFICAR (1 linea)

```python
# L359: Cambiar strategy_used
strategy_used=RetrievalStrategy.HYBRID_PLUS,   # antes: CONTEXTUAL_HYBRID
```

Nota: HybridRetriever es el inner retriever de HybridPlusRetriever. Su
`strategy_used` se sobreescribe en `_swap_to_original_contents`, pero conviene
que sea correcto por coherencia.

### 4.6 `sandbox_mteb/config.py` — MODIFICAR

```python
VALID_STRATEGIES = (
    RetrievalStrategy.SIMPLE_VECTOR,
    RetrievalStrategy.HYBRID_PLUS,
)

# Eliminar _contextual_strategies y needs_llm para retrieval.
# LLM solo es requerido si generation_enabled.
needs_llm = self.generation_enabled
```

### 4.7 `sandbox_mteb/evaluator.py` — MODIFICAR

```python
# L329-333: Simplificar needs_llm
needs_llm = (
    self.config.generation_enabled
    # Ya no: or self.config.retrieval.strategy == RetrievalStrategy.CONTEXTUAL_HYBRID
)
```

### 4.8 `sandbox_mteb/env.example` — MODIFICAR

- Eliminar seccion `CONTEXTUAL RETRIEVAL` (L64-71)
- Actualizar estrategias validas: `SIMPLE_VECTOR, HYBRID_PLUS`
- Actualizar comentarios de RRF: `(solo HYBRID_PLUS)` en vez de `(solo CONTEXTUAL_HYBRID)`
- Actualizar seccion ENTITY: `(solo HYBRID_PLUS)` en vez de `(solo CONTEXTUAL_HYBRID_PLUS)`

### 4.9 `shared/retrieval/entity_linker.py` — MODIFICAR (1 linea)

```python
# L9: Actualizar comentario
# Uso: HYBRID_PLUS aplica cross-linking durante indexacion
```

### 4.10 `tests/test_contextual_retriever_plus.py` — REESCRIBIR como `tests/test_hybrid_plus_retriever.py`

Nuevos tests:

| Test | Verifica |
|---|---|
| `test_index_documents_with_cross_refs` | HybridRetriever recibe docs con cross-refs, sin LLM |
| `test_index_documents_without_spacy` | HAS_SPACY=False -> warning, docs sin cross-refs |
| `test_original_contents_stored` | _original_contents contiene contenido original |
| `test_retrieve_returns_original_content` | retrieve() devuelve original, no enriquecido |
| `test_retrieve_by_vector_returns_original` | idem via vector |
| `test_retrieve_sets_strategy` | strategy_used = HYBRID_PLUS |
| `test_clear_index` | Limpia inner + originals + stats |
| `test_factory_returns_hybrid_plus` | get_retriever con HYBRID_PLUS devuelve HybridPlusRetriever |
| `test_factory_does_not_require_llm` | **NUEVO**: factory no exige llm_service para HYBRID_PLUS |
| `test_empty_documents` | Lista vacia -> False |

### 4.11 `README.md` — MODIFICAR

- Eliminar toda referencia a CONTEXTUAL_HYBRID
- Actualizar tabla de estrategias (2 filas: SIMPLE_VECTOR, HYBRID_PLUS)
- Simplificar pipeline (sin paso LLM enrichment)
- Actualizar config .env (eliminar CONTEXT_* vars)
- Actualizar deuda tecnica (eliminar DTm-19, modificar DTm-20)

---

## 5. Etapas de Implementacion

```
ETAPA A: Crear HybridPlusRetriever (fichero nuevo aislado)
    |
    v
ETAPA B: Rewire del pipeline (enum, factory, config, evaluator)
    |
    v
ETAPA C: Eliminar contextual_retriever.py y codigo muerto
    |
    v
ETAPA D: Tests (reescribir + verificar no regresion)
    |
    v
ETAPA E: Documentacion (README, env.example, plan)
```

### ETAPA A: Crear HybridPlusRetriever

**Objetivo:** Fichero nuevo funcional, sin tocar ficheros existentes.

| # | Tarea | Fichero | Tipo |
|---|---|---|---|
| A.1 | Crear `HybridPlusRetriever` con NER cross-linking | `shared/retrieval/hybrid_plus_retriever.py` | NUEVO |

Criterios de salida:
- [ ] Fichero importable sin errores
- [ ] HAS_SPACY=False no genera error

### ETAPA B: Rewire del pipeline

**Objetivo:** Cablear HYBRID_PLUS como estrategia, eliminar CONTEXTUAL_HYBRID
del enum y factory.

| # | Tarea | Fichero | Tipo |
|---|---|---|---|
| B.1 | Reemplazar enum: eliminar CONTEXTUAL_HYBRID/PLUS, agregar HYBRID_PLUS | `shared/retrieval/core.py` | MODIFICAR |
| B.2 | Eliminar campos context_max_tokens, context_batch_size de RetrievalConfig | `shared/retrieval/core.py` | MODIFICAR |
| B.3 | Eliminar from_env de RETRIEVAL_CONTEXT_* | `shared/retrieval/core.py` | MODIFICAR |
| B.4 | Reescribir factory: 2 ramas (SIMPLE_VECTOR, HYBRID_PLUS) | `shared/retrieval/__init__.py` | MODIFICAR |
| B.5 | Eliminar imports de ContextualRetriever/LLMContextGenerator/EnrichedChunk | `shared/retrieval/__init__.py` | MODIFICAR |
| B.6 | Actualizar VALID_STRATEGIES y eliminar needs_llm para retrieval | `sandbox_mteb/config.py` | MODIFICAR |
| B.7 | Simplificar needs_llm en evaluator | `sandbox_mteb/evaluator.py` | MODIFICAR |
| B.8 | Actualizar strategy_used en HybridRetriever | `shared/retrieval/hybrid_retriever.py` | MODIFICAR |

Criterios de salida:
- [ ] `RETRIEVAL_STRATEGY=HYBRID_PLUS` se parsea en from_env()
- [ ] `RETRIEVAL_STRATEGY=CONTEXTUAL_HYBRID` produce KeyError (eliminado)
- [ ] Factory construye HybridPlusRetriever sin llm_service
- [ ] Config validate() acepta HYBRID_PLUS

### ETAPA C: Eliminar codigo muerto

**Objetivo:** Borrar fichero y limpiar imports residuales.

| # | Tarea | Fichero | Tipo |
|---|---|---|---|
| C.1 | Eliminar `contextual_retriever.py` completo | `shared/retrieval/contextual_retriever.py` | ELIMINAR |
| C.2 | Actualizar comentario en entity_linker.py | `shared/retrieval/entity_linker.py` | MODIFICAR |
| C.3 | Verificar que ningun fichero importa de contextual_retriever | todos | VERIFICAR |

Criterios de salida:
- [ ] `contextual_retriever.py` no existe
- [ ] Ningun import apunta a modulos eliminados
- [ ] `python -c "from shared.retrieval import get_retriever"` OK

### ETAPA D: Tests

**Objetivo:** Reescribir tests del retriever plus, verificar no regresion.

| # | Tarea | Fichero | Tipo |
|---|---|---|---|
| D.1 | Eliminar `test_contextual_retriever_plus.py` | `tests/test_contextual_retriever_plus.py` | ELIMINAR |
| D.2 | Crear `test_hybrid_plus_retriever.py` (~10 tests) | `tests/test_hybrid_plus_retriever.py` | NUEVO |
| D.3 | Verificar tests existentes pasan | `tests/` | VERIFICAR |

Criterios de salida:
- [ ] `pytest tests/test_hybrid_plus_retriever.py -v` — 10+ tests pasan
- [ ] `pytest tests/test_entity_linker.py -v` — 31 tests pasan (sin cambios)
- [ ] `pytest tests/ --ignore=tests/integration -v` — todos pasan sin regresion

### ETAPA E: Documentacion

**Objetivo:** Actualizar README, env.example, y plan con estado final.

| # | Tarea | Fichero | Tipo |
|---|---|---|---|
| E.1 | Actualizar env.example: eliminar CONTEXTUAL_*, actualizar comentarios | `sandbox_mteb/env.example` | MODIFICAR |
| E.2 | Actualizar README.md: 2 estrategias, sin LLM enrichment | `README.md` | MODIFICAR |
| E.3 | Actualizar requirements.txt comentarios | `requirements.txt` | MODIFICAR |

Criterios de salida:
- [ ] env.example no contiene "CONTEXTUAL"
- [ ] README.md documenta SIMPLE_VECTOR + HYBRID_PLUS
- [ ] Pipeline diagram sin paso LLM enrichment

---

## 6. Ficheros — Vista Consolidada

| Fichero | Accion | Cambio |
|---|---|---|
| `shared/retrieval/hybrid_plus_retriever.py` | **NUEVO** | ~120 lineas |
| `shared/retrieval/contextual_retriever.py` | **ELIMINAR** | -607 lineas |
| `shared/retrieval/core.py` | MODIFICAR | -2 enum, -2 campos config, -2 from_env |
| `shared/retrieval/__init__.py` | MODIFICAR | Reescribir factory y imports |
| `shared/retrieval/hybrid_retriever.py` | MODIFICAR | 1 linea (strategy_used) |
| `shared/retrieval/entity_linker.py` | MODIFICAR | 1 linea (comentario) |
| `sandbox_mteb/config.py` | MODIFICAR | Simplificar VALID_STRATEGIES y needs_llm |
| `sandbox_mteb/evaluator.py` | MODIFICAR | Simplificar needs_llm |
| `sandbox_mteb/env.example` | MODIFICAR | Eliminar CONTEXTUAL_*, actualizar |
| `tests/test_contextual_retriever_plus.py` | **ELIMINAR** | -400 lineas |
| `tests/test_hybrid_plus_retriever.py` | **NUEVO** | ~250 lineas |
| `README.md` | MODIFICAR | Reescribir estrategias |
| `requirements.txt` | MODIFICAR | Actualizar comentario |

**Balance neto: ~-650 lineas** (eliminar ~1000, agregar ~370).

---

## 7. Impacto en Tests Existentes

Tests de `entity_linker.py` (31): **sin cambios**. EntityLinker no conoce
la estrategia de retrieval.

Tests que referencian CONTEXTUAL_HYBRID: solo `test_contextual_retriever_plus.py`,
que se reemplaza por `test_hybrid_plus_retriever.py`.

Tests de integracion (`tests/integration/test_pipeline.py`): verificar si
referencian CONTEXTUAL_HYBRID. Si es asi, actualizar a HYBRID_PLUS.

Otros tests (metrics, report, evaluator, etc.): no referencian estrategias
contextuales directamente. Sin impacto.

---

## 8. Riesgos

| Riesgo | Probabilidad | Impacto | Mitigacion |
|---|---|---|---|
| Perdida de calidad en single-hop sin LLM enrichment | Media | Medio | Los datos diran: comparar HYBRID_PLUS vs antiguo CONTEXTUAL_HYBRID. Si hay regresion significativa, el LLM enrichment aportaba valor real. |
| HybridRetriever.strategy_used hardcodeado a CONTEXTUAL_HYBRID en otros sitios | Baja | Bajo | Grep exhaustivo hecho: solo L359 de hybrid_retriever.py |
| Tests de integracion rotos | Baja | Bajo | Verificar en ETAPA D |

---

## 9. Criterios de Aceptacion Global

1. `RETRIEVAL_STRATEGY=HYBRID_PLUS` funciona end-to-end (dry-run + run real)
2. `RETRIEVAL_STRATEGY=CONTEXTUAL_HYBRID` produce error claro (estrategia no existe)
3. No existe `contextual_retriever.py` en el repo
4. Factory no requiere `llm_service` para HYBRID_PLUS
5. `pytest tests/ --ignore=tests/integration` — todos pasan sin regresion
6. Sin spaCy: HYBRID_PLUS funciona (sin cross-refs, BM25+Vector+RRF puro)
7. README y env.example no contienen "CONTEXTUAL_HYBRID"
