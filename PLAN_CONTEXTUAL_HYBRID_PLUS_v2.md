# PLAN DE TRABAJO: CONTEXTUAL_HYBRID_PLUS v2

## Revision

Este plan reemplaza `PLAN_CONTEXTUAL_HYBRID_PLUS.md` v1. Corrige errores
factuales sobre el codebase, reemplaza la extraccion de entidades via LLM
por NER deterministico (spaCy), detalla la normalizacion de entidades,
especifica puntos de integracion exactos con ficheros y tipos reales, y
proporciona estimaciones de esfuerzo realistas.

---

## 1. Objetivo

Mejorar la recuperacion multi-hop en HotpotQA (bridge questions) sin construir
un grafo explicito, reutilizando la infraestructura de enrichment existente.

**Hipotesis:** Si durante la indexacion extraemos entidades nombradas de cada
documento y generamos cross-references textuales entre documentos que comparten
entidades, el retriever plano (BM25+Vector+RRF) podra encontrar documentos
conectados por cadenas de entidades, mejorando Recall@K en bridge questions.

**Mecanismo esperado:** En una bridge question como "Where was the director of
Sinister born?", el bi-encoder genera un vector mas cercano a "Sinister" que a
"Scott Derrickson" (sesgo documentado en README.md). Si Doc A ("Sinister") contiene
una cross-reference textual "Related: Scott Derrickson (shared: Scott Derrickson,
Sinister)", BM25 captura el termino "Scott Derrickson" en Doc A, promoviendo Doc B
("Scott Derrickson") via RRF aunque el vector de la query no lo alcance directamente.

---

## 2. Arquitectura

```
CONTEXTUAL_HYBRID (actual):

  [Doc] --> [LLM: genera contexto] --> [Doc + Contexto] --> [Index BM25+Vector]


CONTEXTUAL_HYBRID_PLUS (propuesto):

  FASE BATCH (indexacion):

  [Doc] --> [LLM: genera contexto]          (sin cambios, prompt actual)
              |
              v
        [EnrichedChunk]
              |
   +----------+----------+
   |                      |
   v                      v
  [spaCy NER:           [contexto LLM]
   extrae entidades]
              |
              v
  [EntityLinker: indice invertido in-memory]
              |
              v
  [Cross-refs textuales por doc]
              |
              v
  [Doc + Contexto LLM + Cross-refs] --> [Index BM25+Vector]


  FASE QUERY (sin cambios):

  [Query] --> [BM25+Vector+RRF] --> [Top-K] --> [Reranker] --> [LLM]

  FASE GENERACION (sin cambios):

  _swap_to_original_contents() devuelve texto original (sin enrichment
  ni cross-refs) al LLM de generacion. Ya implementado en ContextualRetriever.
```

### Decision clave: spaCy en lugar de LLM para NER

El plan v1 proponia un prompt extendido al LLM para extraer entidades
simultaneamente con el contexto. Esto no es viable por tres razones:

1. **Modelo nano:** `nemotron-3-nano` (~1B params) no sigue formatos
   estructurados de forma fiable. El propio proyecto documenta este
   problema en DTm-16 con una instruccion mucho mas simple ("For yes/no
   questions, start with yes or no"). Un formato `ENTITIES: <name> (<type>), ...`
   tendria un parse failure rate inaceptable.

2. **Determinismo:** spaCy es determinista. El mismo documento siempre
   produce las mismas entidades. El LLM no, lo que complica reproducibilidad
   y debugging.

3. **Latencia:** spaCy procesa ~50K tokens/segundo en CPU. El LLM procesa
   ~50 tokens/segundo via NIM. Para 66K documentos, spaCy NER tarda
   segundos; una segunda llamada LLM duplicaria el tiempo de enrichment.

4. **Independencia de infra:** spaCy no requiere endpoint NIM. Funciona
   offline, en CI, y en entornos donde solo se ejecutan tests unitarios.

**Modelo:** `en_core_web_sm` (15MB, solo NER, suficiente para entidades
nombradas en Wikipedia). Importacion condicional con `HAS_SPACY` para no
romper entornos sin spaCy instalado.

---

## 3. Fases de Implementacion

### FASE 0: Dependencia spaCy

**Ficheros:**
- `requirements.txt`: Agregar `spacy>=3.7.0,<4.0` y comentario sobre modelo.
- `README.md`: Instrucciones de setup: `pip install spacy && python -m spacy download en_core_web_sm`.

**Importacion condicional:** Mismo patron que `HAS_TANTIVY`, `HAS_BM25`,
`HAS_NVIDIA_RERANK` ya existentes en el proyecto.

```python
# shared/retrieval/entity_linker.py
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    spacy = None
```

**Fallback sin spaCy:** `CONTEXTUAL_HYBRID_PLUS` se comporta identicamente a
`CONTEXTUAL_HYBRID` (sin cross-refs, solo contexto LLM). Warning en log.

**Estimacion: 0.5h.**

---

### FASE 1: EntityLinker (`shared/retrieval/entity_linker.py`)

Fichero nuevo. Contiene toda la logica de NER, indice invertido, filtrado
IDF, normalizacion, y generacion de cross-references.

#### 1.1 Tipos

```python
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

@dataclass
class DocEntities:
    """Entidades extraidas de un documento."""
    doc_id: str
    doc_title: str
    entities: List[str]  # nombres normalizados
    raw_entities: List[Tuple[str, str]]  # (nombre_original, tipo_NER)
```

Nota: no se extiende `EnrichedChunk`. `DocEntities` es un tipo interno del
entity linker que vive y muere dentro de `index_documents`. No se serializa
ni se pasa a ChromaDB. Esto evita acoplar el NER al resto del pipeline.

#### 1.2 Normalizacion de entidades

Problema central: "Scott Derrickson", "Derrickson", "SCOTT DERRICKSON" y
"scott derrickson" deben unificarse. "U.S.", "US", "United States" no.
Normalizacion perfecta es un problema abierto; buscamos una solucion
suficientemente buena para Wikipedia.

```python
class EntityNormalizer:
    """Normalizacion de entidades para matching en indice invertido."""

    @staticmethod
    def normalize(name: str) -> str:
        """Normaliza un nombre de entidad para uso como clave del indice."""
        # 1. Lowercase
        result = name.lower().strip()
        # 2. Eliminar articulos iniciales (the, a, an)
        for article in ("the ", "a ", "an "):
            if result.startswith(article):
                result = result[len(article):]
        # 3. Colapsar espacios
        result = " ".join(result.split())
        # 4. Eliminar puntuacion excepto guiones internos
        #    "u.s." -> "us", "spider-man" -> "spider-man"
        import re
        result = re.sub(r'(?<!\w)[.\',]|[.\',](?!\w)', '', result)
        result = re.sub(r'[^\w\s-]', '', result)
        return result.strip()
```

**Limitacion aceptada:** "Derrickson" suelto no matchea "scott derrickson".
Esto requeriria alias resolution (NER linking contra Wikidata), que esta
fuera del scope. El target son matches exactos y near-exactos de entidades
completas tal como spaCy las reporta. Para Wikipedia, donde los titulos son
entidades canónicas, esto es suficiente.

#### 1.3 Extraccion NER

```python
class EntityExtractor:
    """Extrae entidades nombradas de documentos usando spaCy."""

    # Tipos NER relevantes para cross-linking multi-hop.
    # Excluir CARDINAL, ORDINAL, QUANTITY, PERCENT (numericos sin identidad).
    # Excluir DATE, TIME (muy genericos, no conectan docs).
    RELEVANT_NER_TYPES: Set[str] = {
        "PERSON", "ORG", "GPE", "LOC", "FAC",
        "EVENT", "WORK_OF_ART", "LAW", "PRODUCT", "NORP",
    }

    # Longitud minima de entidad normalizada (filtrar ruido tipo "I", "He").
    MIN_ENTITY_LENGTH: int = 2

    def __init__(self, model_name: str = "en_core_web_sm"):
        if not HAS_SPACY:
            raise ImportError("spacy no instalado: pip install spacy")
        self._nlp = spacy.load(model_name, disable=["parser", "lemmatizer"])
        # Solo necesitamos NER, desactivar parser y lemmatizer para velocidad.

    def extract(self, text: str) -> List[Tuple[str, str]]:
        """Extrae entidades de un texto.

        Returns:
            Lista de (nombre_normalizado, tipo_NER).
            Deduplicada por nombre normalizado.
        """
        doc = self._nlp(text[:5000])  # Truncar: spaCy sm ya es lento >10K chars
        seen: Dict[str, str] = {}  # normalized -> tipo (primera ocurrencia)
        for ent in doc.ents:
            if ent.label_ not in self.RELEVANT_NER_TYPES:
                continue
            normalized = EntityNormalizer.normalize(ent.text)
            if len(normalized) < self.MIN_ENTITY_LENGTH:
                continue
            if normalized not in seen:
                seen[normalized] = ent.label_
        return list(seen.items())
```

#### 1.4 Indice invertido + IDF filter

```python
from collections import Counter, defaultdict
import math

class EntityLinker:
    """Indice invertido de entidades con IDF filter y generacion de cross-refs."""

    def __init__(
        self,
        max_cross_refs: int = 3,
        min_shared_entities: int = 1,
        max_entity_doc_fraction: float = 0.05,
    ):
        self.max_cross_refs = max_cross_refs
        self.min_shared_entities = min_shared_entities
        self.max_entity_doc_fraction = max_entity_doc_fraction

        # Indice invertido: entidad_normalizada -> set(doc_ids)
        self._entity_to_docs: Dict[str, Set[str]] = defaultdict(set)
        # Indice directo: doc_id -> set(entidades_normalizadas)
        self._doc_to_entities: Dict[str, Set[str]] = {}
        # Metadatos: doc_id -> titulo (para generar refs legibles)
        self._doc_titles: Dict[str, str] = {}

        self._total_docs: int = 0
        self._filtered_entities: Set[str] = set()  # IDF-filtered

    def build_index(self, doc_entities_list: List[DocEntities]) -> None:
        """Construye indice invertido a partir de entidades extraidas.

        Dos pasadas:
          1. Construir indice completo.
          2. Filtrar entidades con doc frequency > max_entity_doc_fraction.
        """
        self._total_docs = len(doc_entities_list)

        # Pasada 1: construir indice
        for de in doc_entities_list:
            self._doc_titles[de.doc_id] = de.doc_title
            entity_set = set(de.entities)
            self._doc_to_entities[de.doc_id] = entity_set
            for entity in entity_set:
                self._entity_to_docs[entity].add(de.doc_id)

        # Pasada 2: filtrar entidades demasiado frecuentes (IDF filter)
        if self._total_docs > 0:
            threshold = int(self._total_docs * self.max_entity_doc_fraction)
            threshold = max(threshold, 10)  # minimo absoluto para corpus pequeños
            for entity, doc_set in self._entity_to_docs.items():
                if len(doc_set) > threshold:
                    self._filtered_entities.add(entity)

        n_filtered = len(self._filtered_entities)
        n_total = len(self._entity_to_docs)
        logger.info(
            f"EntityLinker: {self._total_docs} docs, "
            f"{n_total} entidades unicas, "
            f"{n_filtered} filtradas por IDF (>{self.max_entity_doc_fraction:.0%})"
        )

    def generate_cross_refs(self, doc_id: str) -> str:
        """Genera texto de cross-references para un documento.

        Selecciona los top-N documentos con mayor numero de entidades
        compartidas (excluyendo las filtradas por IDF).

        Returns:
            Texto de cross-refs para anexar al documento antes de indexar.
            String vacio si no hay refs.
        """
        my_entities = self._doc_to_entities.get(doc_id, set())
        if not my_entities:
            return ""

        # Entidades validas (no filtradas)
        valid_entities = my_entities - self._filtered_entities

        if not valid_entities:
            return ""

        # Contar entidades compartidas con cada otro documento
        related: Counter = Counter()
        shared_map: Dict[str, List[str]] = defaultdict(list)

        for entity in valid_entities:
            for other_doc_id in self._entity_to_docs[entity]:
                if other_doc_id != doc_id:
                    related[other_doc_id] += 1
                    shared_map[other_doc_id].append(entity)

        if not related:
            return ""

        # Filtrar por min_shared_entities
        candidates = [
            (did, count) for did, count in related.most_common()
            if count >= self.min_shared_entities
        ]

        if not candidates:
            return ""

        # Top-N
        top_refs = candidates[:self.max_cross_refs]

        # Generar texto
        ref_lines = []
        for ref_doc_id, shared_count in top_refs:
            title = self._doc_titles.get(ref_doc_id, ref_doc_id)
            entities_preview = ", ".join(shared_map[ref_doc_id][:3])
            ref_lines.append(
                f"Related: {title} (shared: {entities_preview})"
            )

        return "\n".join(ref_lines)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_docs": self._total_docs,
            "total_entities": len(self._entity_to_docs),
            "filtered_entities": len(self._filtered_entities),
            "docs_with_entities": sum(
                1 for ents in self._doc_to_entities.values() if ents
            ),
            "avg_entities_per_doc": (
                sum(len(e) for e in self._doc_to_entities.values()) / self._total_docs
                if self._total_docs > 0 else 0.0
            ),
        }
```

#### 1.5 Metodo publico orquestador

Un unico metodo que toma la lista de documentos y la lista de
`EnrichedChunk` ya generados por el `ContextualRetriever` existente,
y devuelve una lista de cross-ref strings indexada por doc_id.

```python
def compute_cross_refs(
    self,
    documents: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Pipeline completo: NER -> build_index -> generate_cross_refs.

    Args:
        documents: Lista de dicts con keys: doc_id, content, title.
                   (misma interfaz que recibe index_documents)

    Returns:
        Dict doc_id -> cross_refs_text. Solo doc_ids con refs no vacias.
    """
    extractor = EntityExtractor()
    doc_entities_list = []

    t0 = time.perf_counter()
    for doc in documents:
        raw_entities = extractor.extract(doc.get("content", ""))
        doc_entities_list.append(DocEntities(
            doc_id=doc.get("doc_id", ""),
            doc_title=doc.get("title", ""),
            entities=[name for name, _type in raw_entities],
            raw_entities=raw_entities,
        ))
    ner_ms = (time.perf_counter() - t0) * 1000

    self.build_index(doc_entities_list)

    t1 = time.perf_counter()
    result = {}
    for de in doc_entities_list:
        refs = self.generate_cross_refs(de.doc_id)
        if refs:
            result[de.doc_id] = refs
    link_ms = (time.perf_counter() - t1) * 1000

    n_with_refs = len(result)
    logger.info(
        f"EntityLinker: NER {ner_ms:.0f}ms, "
        f"cross-linking {link_ms:.0f}ms, "
        f"{n_with_refs}/{len(documents)} docs con cross-refs"
    )

    return result
```

**Estimacion: 6-8h** (incluye implementacion completa, edge cases, logging).

---

### FASE 2: Integracion como Estrategia

#### 2.1 Nuevo enum en `shared/retrieval/core.py`

```python
class RetrievalStrategy(Enum):
    SIMPLE_VECTOR = auto()
    CONTEXTUAL_HYBRID = auto()
    CONTEXTUAL_HYBRID_PLUS = auto()   # <-- nuevo
```

Impacto downstream: `RetrievalConfig.from_env()` ya usa
`RetrievalStrategy[_env("RETRIEVAL_STRATEGY", "SIMPLE_VECTOR")]`, por lo que
el nuevo valor se parsea automaticamente desde `.env` sin cambios en ese metodo.

#### 2.2 Nuevo retriever: `ContextualRetrieverPlus` (`shared/retrieval/contextual_retriever.py`)

**Patron: composicion, no herencia.** El flujo de `ContextualRetriever.index_documents`
es un pipeline lineal de 3 pasos: `_run_batch_generation -> build enriched_docs -> inner_retriever.index_documents`. Insertar cross-linking entre los pasos 2 y 3
requiere reimplementar `index_documents` casi completo. La herencia no ahorra codigo.
Un wrapper por composicion hace la intencion explicita.

```python
class ContextualRetrieverPlus(BaseRetriever):
    """
    Extends ContextualRetriever with entity cross-linking.

    Composicion: delega enrichment a ContextualRetriever._run_batch_generation,
    luego aplica NER + cross-linking antes de pasar docs al inner retriever.

    Flujo index_documents:
      1. LLM enrichment batch (delegado a context_generator existente)
      2. spaCy NER sobre contenido original de cada chunk
      3. EntityLinker.build_index() -> cross-refs por doc
      4. Texto enriquecido = contexto LLM + contenido original + cross-refs
      5. Inner retriever (HybridRetriever) indexa texto final

    Retrieval y _swap_to_original_contents son identicos a ContextualRetriever.
    """

    def __init__(
        self,
        config: RetrievalConfig,
        embedding_model: EmbeddingModelProtocol,
        context_generator: LLMContextGenerator,
        collection_name: Optional[str] = None,
        embedding_batch_size: int = 0,
        max_cross_refs: int = 3,
        min_shared_entities: int = 1,
        max_entity_doc_fraction: float = 0.05,
    ):
        super().__init__(config)
        self.embedding_model = embedding_model
        self.context_generator = context_generator
        self._original_contents: Dict[str, str] = {}

        # Entity linker config
        self._max_cross_refs = max_cross_refs
        self._min_shared_entities = min_shared_entities
        self._max_entity_doc_fraction = max_entity_doc_fraction
        self._linker_stats: Dict[str, Any] = {}

        # Inner retriever (HybridRetriever: BM25+Vector+RRF)
        from .hybrid_retriever import HybridRetriever, HAS_BM25, HAS_TANTIVY
        if HAS_TANTIVY or HAS_BM25:
            self._inner_retriever = HybridRetriever(
                config, embedding_model, collection_name,
                embedding_batch_size=embedding_batch_size,
            )
        else:
            from .core import SimpleVectorRetriever
            logger.warning(
                "CONTEXTUAL_HYBRID_PLUS: ni tantivy ni rank-bm25 disponible, "
                "usando SimpleVector como inner retriever"
            )
            self._inner_retriever = SimpleVectorRetriever(
                config, embedding_model, collection_name,
                embedding_batch_size=embedding_batch_size,
            )

    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> bool:
        if not documents:
            return False

        start_time = time.perf_counter()

        # Paso 1: LLM enrichment (reutiliza flujo existente)
        enriched_chunks = self._run_batch_generation(documents)

        # Paso 2: NER + cross-linking (solo si spaCy disponible)
        cross_refs: Dict[str, str] = {}
        from .entity_linker import HAS_SPACY
        if HAS_SPACY:
            from .entity_linker import EntityLinker
            linker = EntityLinker(
                max_cross_refs=self._max_cross_refs,
                min_shared_entities=self._min_shared_entities,
                max_entity_doc_fraction=self._max_entity_doc_fraction,
            )
            cross_refs = linker.compute_cross_refs(documents)
            self._linker_stats = linker.get_stats()
        else:
            logger.warning(
                "CONTEXTUAL_HYBRID_PLUS: spaCy no disponible. "
                "Comportamiento identico a CONTEXTUAL_HYBRID."
            )

        # Paso 3: Construir docs finales
        enriched_docs = []
        for chunk in enriched_chunks:
            self._original_contents[chunk.chunk_id] = chunk.original_content

            # Texto para indexacion: contexto LLM + contenido + cross-refs
            text = chunk.get_enriched_text()
            refs = cross_refs.get(chunk.chunk_id, "")
            if refs:
                text = f"{text}\n\n{refs}"

            enriched_docs.append({
                "doc_id": chunk.chunk_id,
                "content": text,
                "title": chunk.full_document_title or "",
            })

        # Paso 4: Indexar en inner retriever
        result = self._inner_retriever.index_documents(
            enriched_docs, collection_name=collection_name
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._is_indexed = result

        logger.info(
            f"ContextualRetrieverPlus: indexacion {elapsed_ms:.0f}ms. "
            f"Context stats: {self.context_generator.get_stats()}, "
            f"Linker stats: {self._linker_stats}"
        )
        return result

    # retrieve, retrieve_by_vector, _swap_to_original_contents:
    # Identicos a ContextualRetriever. Implementar delegando al inner
    # retriever y swapping a original_contents.
    # (misma logica, copiar metodos de ContextualRetriever)

    def _run_batch_generation(
        self, documents: List[Dict[str, Any]]
    ) -> List[EnrichedChunk]:
        from shared.llm import run_sync
        batch_size = self.config.context_batch_size
        return run_sync(
            self.context_generator.generate_contexts_batch(
                documents, batch_size=batch_size
            )
        )

    def clear_index(self) -> None:
        self._inner_retriever.clear_index()
        self.context_generator.clear_cache()
        self._original_contents.clear()
        self._linker_stats = {}
        self._is_indexed = False
```

**Nota sobre duplicacion de metodos:** `retrieve`, `retrieve_by_vector`,
`_swap_to_original_contents` y `_run_batch_generation` se repiten respecto a
`ContextualRetriever`. Es una duplicacion aceptada (~30 lineas) para evitar
herencia problematica. Si en el futuro se anade una tercera variante, extraer
un mixin `_ContentSwapMixin` con `_swap_to_original_contents` y
`_run_batch_generation`.

#### 2.3 Factory `shared/retrieval/__init__.py`

Agregar la nueva rama en `get_retriever`:

```python
if strategy == RetrievalStrategy.CONTEXTUAL_HYBRID_PLUS:
    if llm_service is None:
        raise ValueError(
            "CONTEXTUAL_HYBRID_PLUS requiere llm_service para "
            "generar contextos de enriquecimiento durante la indexacion."
        )
    context_generator = LLMContextGenerator(
        llm_service=llm_service,
        max_tokens=config.context_max_tokens,
    )
    return ContextualRetrieverPlus(
        config=config,
        embedding_model=embedding_model,
        context_generator=context_generator,
        collection_name=collection_name,
        embedding_batch_size=embedding_batch_size,
        max_cross_refs=config.entity_max_cross_refs,
        min_shared_entities=config.entity_min_shared,
        max_entity_doc_fraction=config.entity_max_doc_fraction,
    )
```

#### 2.4 Config: nuevos parametros en `RetrievalConfig`

En `shared/retrieval/core.py`, agregar campos al dataclass:

```python
@dataclass
class RetrievalConfig:
    # ... campos existentes ...

    # Entity cross-linking (CONTEXTUAL_HYBRID_PLUS)
    entity_max_cross_refs: int = 3
    entity_min_shared: int = 1
    entity_max_doc_fraction: float = 0.05
```

Y en `from_env`:

```python
entity_max_cross_refs=_env_int("ENTITY_MAX_CROSS_REFS", 3),
entity_min_shared=_env_int("ENTITY_MIN_SHARED", 1),
entity_max_doc_fraction=_env_float("ENTITY_MAX_DOC_FRACTION", 0.05),
```

#### 2.5 Validacion en `sandbox_mteb/config.py`

En `MTEBConfig.validate()`:

```python
# Actualizar VALID_STRATEGIES
VALID_STRATEGIES = (
    RetrievalStrategy.SIMPLE_VECTOR,
    RetrievalStrategy.CONTEXTUAL_HYBRID,
    RetrievalStrategy.CONTEXTUAL_HYBRID_PLUS,
)

# Actualizar needs_llm
needs_llm = (
    self.generation_enabled
    or self.retrieval.strategy in (
        RetrievalStrategy.CONTEXTUAL_HYBRID,
        RetrievalStrategy.CONTEXTUAL_HYBRID_PLUS,
    )
)
```

#### 2.6 Variables .env

Agregar a `sandbox_mteb/env.example`:

```bash
# =========================================================================
# ENTITY CROSS-LINKING (solo CONTEXTUAL_HYBRID_PLUS)
# =========================================================================
# Max cross-references por documento (mas = mas texto indexado, mas recall).
ENTITY_MAX_CROSS_REFS=3

# Minimo de entidades compartidas para generar cross-ref.
ENTITY_MIN_SHARED=1

# Umbral IDF: ignorar entidades que aparecen en >5% del corpus.
# Filtra entidades genericas ("United States", "World War II").
ENTITY_MAX_DOC_FRACTION=0.05
```

**Estimacion: 4-6h** (integracion, wiring, validacion, env).

---

### FASE 3: Tests

#### 3.1 `tests/test_entity_linker.py` (unit tests)

Toda la logica de entity_linker.py es computable pura: NER mock + indice
invertido + cross-refs. No requiere infraestructura.

| Test | Que verifica |
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
| `test_cross_refs_two_docs_shared_entity` | 2 docs comparten "Scott Derrickson" -> cross-ref generada |
| `test_cross_refs_no_shared` | docs sin entidades comunes -> string vacio |
| `test_cross_refs_max_refs` | respeta max_cross_refs=2 |
| `test_cross_refs_min_shared` | min_shared=2 filtra docs con 1 entidad comun |
| `test_cross_refs_sorted_by_overlap` | doc con mas entidades comunes primero |
| `test_cross_refs_filtered_entity_excluded` | entidad IDF-filtrada no genera cross-ref |
| `test_compute_cross_refs_pipeline` | pipeline completo con mock de EntityExtractor |
| `test_compute_cross_refs_empty_docs` | lista vacia -> dict vacio |
| `test_stats` | get_stats() contiene campos esperados |

**Mocking de spaCy:** Para que los tests no dependan de que `en_core_web_sm`
este descargado, mockear `EntityExtractor` inyectando entidades manuales.
Esto es coherente con el principio del proyecto: "testear computacion pura,
mocks solo para infra". spaCy NER es infra.

Tests especificos de integracion con spaCy real (verificar que el modelo
cargado produce entidades esperadas en texto Wikipedia) van en
`tests/integration/`.

**Estimacion ~18 tests, 4-5h.**

#### 3.2 `tests/test_contextual_retriever_plus.py`

| Test | Que verifica |
|---|---|
| `test_index_documents_with_cross_refs` | Mock spaCy + mock inner retriever. Verifica que inner retriever recibe docs con cross-refs en content. |
| `test_index_documents_without_spacy` | `HAS_SPACY=False` -> warning en log, docs sin cross-refs (identico a CONTEXTUAL_HYBRID). |
| `test_swap_to_original_contents` | Retrieval devuelve original_contents, no texto enriquecido. |
| `test_clear_index` | Limpia inner retriever, cache, original_contents, linker_stats. |

**Estimacion ~4 tests, 2-3h.**

#### 3.3 Tests de integracion

Agregar a `tests/integration/test_pipeline.py` un caso con
`RETRIEVAL_STRATEGY=CONTEXTUAL_HYBRID_PLUS` si spaCy y NIM disponibles.

**Estimacion: 1h.**

**Estimacion total tests: 7-9h.**

---

### FASE 4: Evaluacion Comparativa

#### 4.1 Runs

Tres runs con configuracion identica excepto `RETRIEVAL_STRATEGY`:

```bash
# Baseline vector puro
RETRIEVAL_STRATEGY=SIMPLE_VECTOR

# Hybrid actual (contexto LLM + BM25+Vector+RRF)
RETRIEVAL_STRATEGY=CONTEXTUAL_HYBRID

# Hybrid + entity cross-linking
RETRIEVAL_STRATEGY=CONTEXTUAL_HYBRID_PLUS
```

Resto de parametros fijos: mismo seed, mismas queries, mismo corpus,
mismo reranker config. Necesario para que la comparacion sea valida.

#### 4.2 Metricas a comparar

**Retrieval (pre-rerank):** Hit@5, Hit@10, MRR, Recall@5, Recall@20, NDCG@10.

**Retrieval efectivo (post-rerank):** generation_recall, generation_hit,
reranker_rescue_count.

**Generacion:** F1 (primaria), EM (secundaria), Faithfulness (informativa).

#### 4.3 Desglose por question_type

Campo `question_type` disponible en `NormalizedQuery.metadata["question_type"]`
(cargado por `_populate_from_dataframes` en `loader.py`). Valores para HotpotQA:
`bridge` (~80%) y `comparison` (~20%).

**Analisis post-hoc** sobre el CSV de detalle:

```python
# Pseudocodigo para analisis
detail_df = pd.read_csv(f"{run_id}_detail.csv")
# Cruzar con queries para obtener question_type
# (no esta en el CSV actual -> requiere join con dataset o agregar columna)
```

**Accion requerida:** Agregar `question_type` al detail CSV en `report.py`.
Actualmente no se exporta. Es un campo de metadata de la query, no del
resultado. Linea de cambio en `RunExporter.to_detail_csv`:

```python
row["question_type"] = qr.metadata.get("question_type", "")
```

Para que este campo este disponible, `_evaluate_queries` en evaluator.py
debe propagarlo al metadata de `QueryEvaluationResult`. Actualmente solo
propaga `reranked`.

#### 4.4 Hipotesis

| Hipotesis | Metrica | Aceptar si |
|---|---|---|
| H1: PLUS > HYBRID en bridge | Recall@5 bridge subset | delta > +0.02 |
| H2: PLUS ~ HYBRID en comparison | Recall@5 comparison subset | delta en [-0.02, +0.02] |
| H3: PLUS no degrada generacion | avg F1 global | delta > -0.01 |
| H4: Cross-refs no saturan contexto | avg n_generation_docs | sin cambio |

Si H1 falla: las cross-refs no estan produciendo bridges utiles. Revisar
normalizacion, IDF threshold, o calidad del NER.

Si H2 falla (PLUS peor en comparison): las cross-refs anaden ruido. Mitigar
reduciendo `ENTITY_MAX_CROSS_REFS` o aumentando `ENTITY_MIN_SHARED`.

**Estimacion: 3-4h** (runs + analisis + documentar resultados).

---

## 4. Orden de Ejecucion y Dependencias

```
FASE 0 (spaCy dep)
    |
    v
FASE 1 (entity_linker.py)  -->  FASE 3.1 (tests entity_linker)
    |
    v
FASE 2 (integracion)  -->  FASE 3.2 (tests retriever_plus)
    |                           |
    v                           v
FASE 4 (evaluacion)         FASE 3.3 (tests integracion)
```

Fases 1 y 3.1 pueden desarrollarse en paralelo (TDD). Fase 2 depende de
que Fase 1 compile. Fase 4 depende de que todo este integrado.

---

## 5. Ficheros Modificados y Nuevos

| Fichero | Accion | Cambio |
|---|---|---|
| `shared/retrieval/core.py` | MODIFICAR | Agregar `CONTEXTUAL_HYBRID_PLUS` a `RetrievalStrategy`. Agregar 3 campos entity a `RetrievalConfig` + `from_env`. |
| `shared/retrieval/entity_linker.py` | NUEVO | `EntityNormalizer`, `EntityExtractor`, `DocEntities`, `EntityLinker`, `HAS_SPACY`. |
| `shared/retrieval/contextual_retriever.py` | MODIFICAR | Agregar `ContextualRetrieverPlus` al final del fichero. |
| `shared/retrieval/__init__.py` | MODIFICAR | Importar `ContextualRetrieverPlus`, agregar rama `CONTEXTUAL_HYBRID_PLUS` en `get_retriever`. |
| `sandbox_mteb/config.py` | MODIFICAR | Agregar `CONTEXTUAL_HYBRID_PLUS` a `VALID_STRATEGIES`. Actualizar `needs_llm`. |
| `sandbox_mteb/env.example` | MODIFICAR | Agregar bloque `ENTITY_*` variables. |
| `shared/report.py` | MODIFICAR | Agregar `question_type` a detail CSV. |
| `sandbox_mteb/evaluator.py` | MODIFICAR | Propagar `question_type` de query metadata a `QueryEvaluationResult.metadata`. |
| `requirements.txt` | MODIFICAR | Agregar `spacy>=3.7.0,<4.0`. |
| `README.md` | MODIFICAR | Documentar nueva estrategia, setup spaCy, seccion Entity Cross-Linking. |
| `tests/test_entity_linker.py` | NUEVO | ~18 unit tests. |
| `tests/test_contextual_retriever_plus.py` | NUEVO | ~4 unit tests. |
| `tests/integration/test_pipeline.py` | MODIFICAR | Agregar caso CONTEXTUAL_HYBRID_PLUS. |

---

## 6. Estimacion Total

| Fase | Esfuerzo |
|---|---|
| FASE 0: Dependencia spaCy | 0.5h |
| FASE 1: EntityLinker | 6-8h |
| FASE 2: Integracion | 4-6h |
| FASE 3: Tests | 7-9h |
| FASE 4: Evaluacion | 3-4h |
| Documentacion (README, env.example) | 1-2h |
| **Total** | **22-30h** |

---

## 7. Riesgos

| ID | Riesgo | Probabilidad | Impacto | Mitigacion |
|---|---|---|---|---|
| R1 | spaCy NER produce entidades ruidosas en texto Wikipedia (fragmentos de oraciones clasificados como entidades) | Media | Medio | `MIN_ENTITY_LENGTH=2`, filtrado de tipos NER irrelevantes, IDF filter para entidades frecuentes. Validar en Fase 4 con muestra manual. |
| R2 | Entidades genericas ("United States", "World War II") generan cross-refs masivas | Alta | Alto | IDF filter con `ENTITY_MAX_DOC_FRACTION=0.05`. Umbral configurable. |
| R3 | Normalizacion insuficiente: "Derrickson" y "Scott Derrickson" no matchean | Alta | Medio | Limitacion aceptada (documentar). spaCy tipicamente reporta la forma completa en texto Wikipedia. Para mejora futura: entity linking contra Wikidata. |
| R4 | Cross-refs anaden ruido en comparison questions, degradando precision | Media | Medio | `ENTITY_MAX_CROSS_REFS=3` y `ENTITY_MIN_SHARED=1` limitan texto extra. Si H2 falla en Fase 4, reducir refs o aumentar min_shared. |
| R5 | Overhead de texto indexado degrada latencia de retrieval | Baja | Bajo | ~150-300 chars extra por doc. BM25 Tantivy y ChromaDB escalan bien. No afecta latencia de query, solo tamano de indice (+15-20%). |
| R6 | spaCy `en_core_web_sm` no disponible en entorno de ejecucion | Media | Bajo | Importacion condicional `HAS_SPACY`. Sin spaCy, PLUS = HYBRID (degradacion graceful). Warning en log. |
| R7 | Circularidad en cross-refs (A ref B, B ref A) | Alta | Nulo | No es un problema. Ambas direcciones refuerzan el bridge. No amplifica scores (BM25/vector son independientes por doc). |
| R8 | Overhead de memoria del indice invertido para 66K docs | Baja | Bajo | ~66K docs * ~10 entidades/doc * ~30 bytes/entidad = ~20MB. Trivial. |

---

## 8. Criterios de Aceptacion

1. `pytest tests/test_entity_linker.py tests/test_contextual_retriever_plus.py -v` pasa (18+4 tests).
2. `mypy shared/retrieval/entity_linker.py` sin errores.
3. `RETRIEVAL_STRATEGY=CONTEXTUAL_HYBRID_PLUS python -m sandbox_mteb.run --dry-run` valida config sin errores.
4. Run completo con DEV_MODE genera JSON+CSV con `retrieval_strategy: CONTEXTUAL_HYBRID_PLUS`.
5. detail CSV incluye columna `question_type`.
6. Sin spaCy instalado, CONTEXTUAL_HYBRID_PLUS produce resultados identicos a CONTEXTUAL_HYBRID (test explicito).
7. Evaluacion comparativa documentada con tabla de resultados por question_type.

---

## 9. Deuda Tecnica Resultante

| ID | Descripcion | Prioridad |
|---|---|---|
| DTm-18 | Entity normalization basica (lowercase + articulos). No resuelve aliases (US/United States) ni formas parciales (Derrickson/Scott Derrickson). Mejora futura: entity linking contra Wikidata o embedding similarity entre entidades. | Baja |
| DTm-19 | `ContextualRetrieverPlus` duplica ~30 lineas de `ContextualRetriever` (retrieve, retrieve_by_vector, _swap_to_original_contents). Extraer mixin si se anade una tercera variante. | Baja |
| DTm-20 | `question_type` en detail CSV requiere propagacion manual desde query metadata. Considerar un mecanismo generico de metadata passthrough en `QueryEvaluationResult`. | Baja |
