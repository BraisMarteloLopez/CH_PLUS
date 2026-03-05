# Plan de Optimizacion HYBRID_PLUS

## 1. Diagnostico: Causas Raiz Identificadas

### 1.1 CRITICO - Comparativa Confundida por Corpus Diferente

La comparativa actual **no es valida**:

| Factor | SIMPLE_VECTOR | HYBRID_PLUS |
|--------|--------------|-------------|
| Corpus | 66K (100%) | 33K (50% random) |
| Queries | 600 | 125 |
| Gold docs garantizados | No (pero con 66K la cobertura es ~100%) | No (con 33K, ~28-35 queries sin gold docs) |

**Evidencia del run HYBRID_PLUS:**
- 38 queries (30.4%) tienen recall@20 = 0.0
- De esas 38, 35 tienen generation_recall = 0.0 (docs no encontrados ni en pool de 450 candidatos)
- **Esos 35 queries casi seguro no tienen sus gold docs en el subset de 33K**
- Con `CORPUS_SHUFFLE_SEED=42` y `max_corpus=33000` de 66576, se toma un 49.6% aleatorio

**Implicacion:** ~28% de los queries fallan por diseno experimental, no por el retriever.

### 1.2 CRITICO - Inline Enrichment Corrompe Embeddings Vectoriales

`hybrid_plus_retriever.py:154`:
```python
indexed_content = f"{content} {refs}"
# refs = "See also X regarding Y. See also Z regarding W."
```

- Los documentos se embeben con ~175 chars extra de cross-refs
- Los embeddings resultantes son **diferentes** a los que SIMPLE_VECTOR genera
- El componente vectorial del hibrido trabaja con embeddings contaminados
- SIMPLE_VECTOR con embeddings limpios = 87% recall; vector contaminado = degradado

**Este es probablemente el factor #1.** El inline enrichment ayuda a BM25 (bridge terms)
pero **destruye la busqueda vectorial** que es el pilar del sistema.

### 1.3 ALTO - RRF con Pesos Iguales Diluye la Senal Vectorial

Config actual: `bm25_weight=0.5, vector_weight=0.5`

- Vector search solo ya da 87% recall (SIMPLE_VECTOR benchmark)
- BM25 sobre texto enriquecido encuentra candidatos diferentes
- Pesos iguales permiten que BM25 desplace docs que vector rankeaba alto
- Para HotpotQA (preguntas semanticas multi-hop), vector deberia dominar

### 1.4 MODERADO - Graph Expansion Genera Pool Excesivo para Reranker

**Datos del run:**
- Pre-rerank candidates: promedio 446 docs (min 323, max 541)
- Esperado sin expansion: ~150 (RRF top_n cuando reranker enabled)
- Graph expansion anade ~300 vecinos por query
- Reranker busca aguja en pajar de 450 docs vs 20

**Pero:** Los graph-expanded docs se anaden DESPUES de los RRF results.
Las metricas de retrieval (Hit@5, Recall@5) usan `[:retrieval_k]` = primeros 20 del RRF.
Asi que graph expansion **no afecta las metricas de retrieval**, solo el pool del reranker.

### 1.5 EVIDENCIA - Curva de Recall Plana

```
Recall@1=0.276  @5=0.408  @10=0.420  @20=0.444
```

La curva casi no crece despues de @5. Esto indica:
- Los docs o estan en top 5, o no estan en el indice
- El ranking RRF no tiene docs relevantes "enterrados" en posiciones 6-20
- Confirma que el problema es a nivel de corpus (docs ausentes) + embeddings (vector degradado)

---

## 2. Plan de Optimizacion (por prioridad)

### Fix 0: Experimental - Corregir la Comparativa (PREREQUISITO)

**Antes de optimizar codigo, necesitamos una comparativa justa.**

Opcion A - Usar DEV_MODE:
```env
DEV_MODE=true
DEV_QUERIES=125
DEV_CORPUS_SIZE=33000
```
Esto garantiza gold docs en el corpus + rellena con distractores.

Opcion B - Usar corpus completo:
```env
EVAL_MAX_QUERIES=125
EVAL_MAX_CORPUS=0  # Todo el corpus (66K)
```

**Recomendacion:** Opcion A (DEV_MODE). Mismas 125 queries, mismos 33K docs,
pero gold docs garantizados. Comparativa limpia.

### Fix 1: Separar Embeddings del Enrichment (CRITICO)

**Cambio en `hybrid_plus_retriever.py`:**

El concepto: los cross-refs inline deben enriquecer SOLO el indice BM25,
no los embeddings vectoriales.

**Implementacion:** Modificar `HybridRetriever` para soportar indexacion dual:
- Vector index: recibe contenido ORIGINAL (embeddings limpios)
- BM25 index: recibe contenido ENRIQUECIDO (con cross-refs)

```python
# En HybridRetriever.index_documents():
# Actualmente indexa el MISMO texto en vector y BM25
# Cambiar a:
def index_documents(self, documents, collection_name=None,
                    bm25_documents=None):
    # documents -> vector index (original)
    # bm25_documents -> BM25 index (enriched, or same if None)
```

Y en `HybridPlusRetriever.index_documents()`:
```python
# Pasar docs originales para vector, enriquecidos para BM25
self._inner_retriever.index_documents(
    documents=original_docs,       # Vector: embeddings limpios
    collection_name=collection_name,
    bm25_documents=enriched_docs,  # BM25: con cross-refs
)
```

**Archivos a modificar:**
- `shared/retrieval/hybrid_retriever.py`: Aceptar `bm25_documents` opcional
- `shared/retrieval/hybrid_plus_retriever.py`: Pasar docs separados
- Tests existentes no se rompen (bm25_documents=None = comportamiento actual)

### Fix 2: Ajustar Pesos RRF (ALTO)

**Cambio en `.env`:**
```env
RETRIEVAL_VECTOR_WEIGHT=0.7
RETRIEVAL_BM25_WEIGHT=0.3
```

Justificacion: vector es el pilar (87% recall solo), BM25 es complementario.
Para HotpotQA (semantico), BM25 aporta menos que en dominios con terminologia tecnica.

**Alternativa mas conservadora:** 0.6/0.4

### Fix 3: Limitar Graph Expansion (MODERADO)

**Cambio en `hybrid_plus_retriever.py:_expand_with_graph()`:**

Actualmente: expande TODOS los vecinos de TODOS los docs recuperados.
Con 20 docs * 3 vecinos = hasta 60 nuevos docs, y estos a su vez tienen vecinos...
No, solo 1 nivel, pero 150 docs * 3 vecinos = 450.

**Propuesta:** Limitar expansion total a `max_graph_expansion` (configurable).

```python
# Limitar a N expansiones totales
MAX_GRAPH_EXPANSION = 30  # Configurable via RetrievalConfig

for doc_id in result.doc_ids:
    if len(expanded_ids) >= MAX_GRAPH_EXPANSION:
        break
    neighbors = self._cross_ref_graph.get(doc_id, [])
    for neighbor_id in neighbors:
        if len(expanded_ids) >= MAX_GRAPH_EXPANSION:
            break
        if neighbor_id not in existing_ids:
            # ... add neighbor
```

**Archivos a modificar:**
- `shared/retrieval/core.py`: Anadir `max_graph_expansion: int = 30` a RetrievalConfig
- `shared/retrieval/hybrid_plus_retriever.py`: Usar el limite
- `sandbox_mteb/env.example`: Documentar parametro

### Fix 4: Parametro RRF k (BAJO)

El `rrf_k=60` es estandar (paper original). Pero con vectores contaminados,
un k menor (e.g., 30) daria mas peso a los primeros ranks de cada ranker.
Esto solo tiene sentido DESPUES de aplicar Fix 1.

---

## 3. Orden de Implementacion

```
1. Fix 0 (experimental)  -> Run baseline justo
2. Fix 1 (dual indexing) -> Corregir el problema #1
3. Fix 2 (pesos RRF)     -> Optimizar fusion
4. Fix 3 (cap expansion) -> Limpiar pool reranker
5. Run comparativo con mismos params que baseline
```

Cada fix es aditivo e independiente. Se puede medir el impacto de cada uno.

---

## 4. Archivos Afectados

| Archivo | Fix | Cambio |
|---------|-----|--------|
| `shared/retrieval/hybrid_retriever.py` | 1 | Dual indexing (vector vs BM25) |
| `shared/retrieval/hybrid_plus_retriever.py` | 1,3 | Pasar docs separados + cap expansion |
| `shared/retrieval/core.py` | 3 | Nuevo param `max_graph_expansion` |
| `sandbox_mteb/env.example` | 2,3 | Documentar nuevos defaults |
| `tests/test_hybrid_plus_retriever.py` | 1,3 | Actualizar tests |
| `tests/test_dtm4_rrf.py` | - | No afectado |

## 5. Riesgos

- **Fix 1** cambia la interfaz de `HybridRetriever.index_documents()`. Backward-compatible
  via `bm25_documents=None` default.
- **Fix 2** es solo config, zero risk.
- **Fix 3** podria reducir recall en bridge questions si el cap es muy bajo.
  30 es conservador (3 vecinos * top 10 docs).
