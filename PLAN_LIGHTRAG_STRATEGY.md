# Plan: Estrategia LIGHT_RAG para CH_PLUS

## 0. Que es LightRAG y que NO es

**LightRAG** (HKUDS, EMNLP 2025, arXiv:2410.05779) es un framework que:

1. Usa un LLM para extraer **tripletas** (entidad → relacion → entidad) del corpus
2. Construye un **knowledge graph real** con nodos tipados y aristas semanticas
3. Hace **dual-level retrieval**: low-level (entidades especificas) + high-level (temas)
4. Combina **vector search + graph traversal** para retrieval

**NO es** lo que tenemos ahora (HYBRID_PLUS = spaCy NER + co-ocurrencia + RRF).
La diferencia es categorica: tripletas semanticas vs pattern matching de strings.

---

## 1. Arquitectura Propuesta: LIGHT_RAG

### 1.1 Pipeline de Indexacion

```
Documentos
    |
    v
[1. Chunking]  (ya existe, docs vienen pre-chunked de MTEB)
    |
    v
[2. LLM Triplet Extraction]  *** NUEVO ***
    |  Input:  texto de cada doc
    |  Output: [(entity1, relation, entity2), ...]
    |  Prompt: structured extraction con formato JSON
    |  LLM:    AsyncLLMService (NIM existente, nemotron-3-nano)
    |
    v
[3. Knowledge Graph Construction]  *** NUEVO ***
    |  Nodos: entidades (con tipo, descripcion)
    |  Aristas: relaciones semanticas (con texto)
    |  Deduplicacion: merge de entidades por nombre normalizado
    |  Storage: NetworkX (in-memory, suficiente para 33K-66K docs)
    |
    v
[4a. Vector Index]           [4b. Graph Index]
    |  ChromaDB (existente)      |  Invertido: entity -> doc_ids
    |  Embeddings limpios        |  Subgrafo por query keywords
    |  (original content)        |
    v                            v
          [Retrieval Pipeline]
```

### 1.2 Pipeline de Retrieval

```
Query
    |
    v
[1. Query Analysis]  *** NUEVO ***
    |  LLM extrae:
    |    - low_level_keywords: entidades especificas ("Shirley Temple", "Kiss and Tell")
    |    - high_level_keywords: temas abstractos ("acting career", "political role")
    |
    v
[2a. Vector Search]          [2b. Graph Traversal]  *** NUEVO ***
    |  top_k por similitud       |  Low-level: buscar entidades en grafo,
    |  coseno (existente)        |    recuperar docs conectados via aristas
    |                            |  High-level: buscar temas en comunidades
    |                            |    del grafo, recuperar docs relevantes
    v                            v
[3. Fusion]
    |  Merge de resultados vector + graph
    |  Scoring: vector_score + graph_relevance_score
    |  Deduplicacion por doc_id
    |
    v
[4. Reranker]  (existente, sin cambios)
    |
    v
[5. Generation]  (existente, sin cambios)
```

---

## 2. Sobre Mantener Busqueda Hibrida (BM25+Vector)

### Recomendacion: **NO mantener BM25 en LIGHT_RAG**

**Razon:** El grafo de conocimiento reemplaza y supera el rol de BM25.

| Funcion         | BM25 en HYBRID_PLUS | Grafo en LIGHT_RAG |
|----------------|--------------------|--------------------|
| Keyword match  | Busqueda lexica    | Entity lookup en indice invertido |
| Bridge terms   | Cross-refs inline  | Aristas semanticas entre entidades |
| Multi-hop      | No (1 hop max)     | Traversal N-hop por relaciones |
| Terminologia   | Stemming + TF-IDF  | Entity normalization + dedup |

BM25 anade lo que el grafo ya cubre, pero sin la estructura relacional.
Mantenerlo duplica la logica sin gain medible.

**Arquitectura LIGHT_RAG = Vector + Graph (sin BM25)**

Si en evaluacion el grafo no aporta lo esperado, BM25 se puede re-activar
facilmente (HybridRetriever ya soporta dual indexing). Pero no deberia
ser el default.

---

## 3. Componentes a Implementar

### 3.1 `shared/retrieval/knowledge_graph.py` (NUEVO)

```python
@dataclass
class KGEntity:
    name: str               # normalizado
    entity_type: str        # PERSON, ORG, etc.
    description: str        # del LLM
    source_doc_ids: Set[str]

@dataclass
class KGRelation:
    source: str             # entity name
    target: str             # entity name
    relation: str           # tipo de relacion
    description: str        # del LLM
    source_doc_id: str

class KnowledgeGraph:
    """Knowledge graph in-memory basado en NetworkX."""

    def __init__(self):
        self._graph: nx.Graph = nx.Graph()
        self._entity_to_docs: Dict[str, Set[str]] = {}
        self._doc_to_entities: Dict[str, Set[str]] = {}

    def add_triplets(self, doc_id: str, triplets: List[KGRelation]):
        """Anade tripletas al grafo, deduplicando entidades."""

    def query_entities(self, entity_names: List[str], max_hops: int = 2) -> List[str]:
        """Traversal: dado entidades, devuelve doc_ids conectados hasta N hops."""

    def query_themes(self, theme_keywords: List[str], top_k: int = 10) -> List[str]:
        """High-level: busca docs por temas via comunidades del grafo."""

    def get_subgraph_context(self, entity_names: List[str]) -> str:
        """Genera contexto textual del subgrafo para la query."""
```

### 3.2 `shared/retrieval/triplet_extractor.py` (NUEVO)

```python
class TripletExtractor:
    """Extrae tripletas (entity, relation, entity) usando LLM."""

    EXTRACTION_PROMPT = '''Extract entities and relationships from this text.
    Return JSON: {"entities": [{"name": "...", "type": "...", "description": "..."}],
                  "relations": [{"source": "...", "target": "...", "relation": "...", "description": "..."}]}
    Text: {text}'''

    def __init__(self, llm_service: AsyncLLMService):
        self._llm = llm_service

    async def extract(self, doc_id: str, text: str) -> List[KGRelation]:
        """Extrae tripletas de un doc via LLM."""

    async def extract_batch(self, documents: List[Dict]) -> Dict[str, List[KGRelation]]:
        """Extraccion paralela con semaphore (NIM concurrencia)."""

    async def extract_query_keywords(self, query: str) -> Tuple[List[str], List[str]]:
        """Extrae low-level entities y high-level themes de una query."""
```

### 3.3 `shared/retrieval/lightrag_retriever.py` (NUEVO)

```python
class LightRAGRetriever(BaseRetriever):
    """Retriever LightRAG: Vector + Knowledge Graph dual-level."""

    def __init__(self, config, embedding_model, llm_service, ...):
        self._vector_retriever = SimpleVectorRetriever(...)
        self._knowledge_graph = KnowledgeGraph()
        self._triplet_extractor = TripletExtractor(llm_service)

    def index_documents(self, documents, collection_name=None):
        """
        1. Index docs en ChromaDB (contenido original)
        2. Extraer tripletas via LLM (async batch)
        3. Construir knowledge graph
        """

    def retrieve(self, query, top_k=None):
        """
        1. Vector search -> top_k candidates
        2. Query analysis -> low_level + high_level keywords
        3. Graph traversal -> related doc_ids
        4. Merge & score -> final results
        """

    def retrieve_by_vector(self, query_text, query_vector, top_k=None):
        """Mismo flujo pero con vector pre-computado."""
```

### 3.4 Modificaciones a Archivos Existentes

| Archivo | Cambio |
|---------|--------|
| `shared/retrieval/core.py` | Anadir `LIGHT_RAG` a `RetrievalStrategy` enum |
| `shared/retrieval/__init__.py` | Registrar `LightRAGRetriever` en factory |
| `shared/retrieval/core.py` | Anadir config params: `kg_max_hops`, `kg_extraction_batch_size` |
| `sandbox_mteb/env.example` | Documentar nuevos params de LIGHT_RAG |
| `sandbox_mteb/config.py` | Pasar LLM service al factory para LIGHT_RAG |
| `sandbox_mteb/evaluator.py` | Pasar LLM service a get_retriever cuando strategy=LIGHT_RAG |

---

## 4. Decisiones de Diseno

### 4.1 LLM para Extraccion: nemotron-3-nano (NIM existente)

- Ya esta desplegado y accesible via `AsyncLLMService`
- Suficiente para extraccion de tripletas (no requiere razonamiento complejo)
- Limitacion: calidad de tripletas depende del modelo. Si las tripletas son ruidosas,
  el grafo no aportara. Mitigacion: prompt engineering + validacion de output JSON.

### 4.2 Graph Store: NetworkX (in-memory)

- 33K-66K docs con ~3-5 tripletas/doc = ~150K-330K nodos+aristas
- NetworkX maneja esto sin problema (<1GB RAM)
- No necesitamos persistencia (se reconstruye cada run de evaluacion)
- Alternativa futura: Neo4j si el corpus crece a millones

### 4.3 Indexacion: Coste de LLM Calls

**Estimacion para 33K docs:**
- ~33K llamadas al LLM para extraccion de tripletas
- Con 32 concurrentes y ~200ms/call: ~33000/32*0.2 = ~206 segundos (~3.5 min)
- Con batch optimization (chunks de 5 docs por prompt): ~6600 calls = ~41 segundos

**Mitigacion:**
- Cache de tripletas en disco (JSON) para re-runs sin re-extraer
- Batch chunking: enviar 3-5 docs por prompt con separadores
- Async con semaphore (ya implementado en AsyncLLMService)

### 4.4 Query Analysis: Coste por Query

- 1 LLM call extra por query para extraer keywords
- ~200ms adicionales por query
- Con 125 queries: ~25 segundos extra
- Pre-extractable en batch (como los embeddings)

---

## 5. Orden de Implementacion

```
Fase 1: Foundation (estimado: ~400 LOC)
  1.1  knowledge_graph.py: KGEntity, KGRelation, KnowledgeGraph
  1.2  triplet_extractor.py: TripletExtractor con prompt + parser JSON
  1.3  Tests unitarios para ambos

Fase 2: Retriever (estimado: ~300 LOC)
  2.1  lightrag_retriever.py: LightRAGRetriever
  2.2  Integracion con factory (core.py, __init__.py)
  2.3  Config params en RetrievalConfig
  2.4  Tests unitarios

Fase 3: Evaluacion (estimado: ~50 LOC)
  3.1  Wiring en evaluator.py (pasar LLM service)
  3.2  env.example con defaults
  3.3  Run comparativo: SIMPLE_VECTOR vs LIGHT_RAG (mismas queries, mismo corpus)

Fase 4: Optimizacion (post-evaluacion)
  4.1  Cache de tripletas (evitar re-extraccion entre runs)
  4.2  Batch extraction (multiples docs por prompt)
  4.3  Tuning de max_hops, scoring weights
```

---

## 6. Que Pasa con HYBRID_PLUS

**No eliminar.** HYBRID_PLUS sigue siendo una estrategia valida en el sandbox.
El sandbox existe para comparar estrategias — tener tres opciones es mejor que dos:

- `SIMPLE_VECTOR`: baseline (vector puro)
- `HYBRID_PLUS`: BM25+Vector+RRF+entity co-occurrence (actual)
- `LIGHT_RAG`: Vector+KnowledgeGraph (nuevo)

La comparativa triple mostrara empiricamente si el knowledge graph justifica
el coste adicional de las LLM calls vs el approach sin LLM de HYBRID_PLUS.

---

## 7. Riesgos y Mitigaciones

| Riesgo | Impacto | Mitigacion |
|--------|---------|------------|
| nemotron-3-nano extrae tripletas de baja calidad | Grafo ruidoso, peor que SIMPLE_VECTOR | Prompt engineering, validacion JSON, fallback a spaCy NER |
| Coste de indexacion (33K LLM calls) | Run lento (~5-10 min extra) | Cache en disco, batch extraction |
| NetworkX no escala | RAM si corpus > 500K | No aplica para MTEB eval (66K max) |
| Query analysis anade latencia | ~200ms/query extra | Pre-extract en batch como embeddings |
| Calidad del grafo depende del corpus | HotpotQA docs son cortos (~2 frases), pocas tripletas | Mitigar con multi-doc extraction (agrupar por titulo) |

---

## 8. Metricas de Exito

LIGHT_RAG se justifica si, sobre el **mismo corpus y queries (DEV_MODE)**:

| Metrica | SIMPLE_VECTOR (baseline) | Target LIGHT_RAG |
|---------|-------------------------|-------------------|
| Hit@5   | ~87% (corpus completo)  | >90% |
| Recall@5| ~80% (DEV_MODE)         | >85% |
| Bridge Q recall | ~75%            | >85% (aqui es donde el grafo debe brillar) |
| Comparison Q recall | ~90%        | >=90% (no regresar) |

Si LIGHT_RAG no supera en bridge questions, el knowledge graph no aporta
y SIMPLE_VECTOR + reranker sigue siendo la mejor estrategia.
