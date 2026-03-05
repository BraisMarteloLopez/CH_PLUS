"""
Modulo: Entity Linker
Descripcion: NER deterministico (spaCy), indice invertido de entidades,
             filtrado IDF, y generacion de cross-references textuales
             entre documentos que comparten entidades.

Ubicacion: shared/retrieval/entity_linker.py

Uso: HYBRID_PLUS aplica cross-linking durante indexacion
para mejorar retrieval multi-hop en bridge questions.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# IMPORTACION CONDICIONAL — mismo patron que HAS_TANTIVY en hybrid_retriever.py
# =============================================================================

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    spacy = None


# =============================================================================
# TIPOS
# =============================================================================

@dataclass
class DocEntities:
    """Entidades extraidas de un documento."""
    doc_id: str
    doc_title: str
    entities: List[str]                   # nombres normalizados
    raw_entities: List[Tuple[str, str]]   # (nombre_original, tipo_NER)


# =============================================================================
# NORMALIZACION
# =============================================================================

# Patron precompilado para eliminar puntuacion excepto guiones internos.
_RE_INTERNAL_PUNCT = re.compile(r"(?<!\w)[.',]|[.',](?!\w)")
_RE_NON_ALNUM = re.compile(r"[^\w\s-]")

# Articulos iniciales en ingles.
_LEADING_ARTICLES = ("the ", "a ", "an ")


def normalize_entity(name: str) -> str:
    """Normaliza un nombre de entidad para uso como clave del indice invertido.

    Pasos:
      1. Lowercase + strip
      2. Eliminar articulos iniciales (the, a, an)
      3. Colapsar espacios multiples
      4. Eliminar puntuacion excepto guiones internos
         "u.s." -> "us", "spider-man" -> "spider-man"
    """
    result = name.lower().strip()
    if not result:
        return ""

    # Articulos iniciales
    for article in _LEADING_ARTICLES:
        if result.startswith(article):
            result = result[len(article):]
            break

    # Colapsar espacios
    result = " ".join(result.split())

    # Eliminar puntuacion (preservar guiones internos)
    result = _RE_INTERNAL_PUNCT.sub("", result)
    result = _RE_NON_ALNUM.sub("", result)

    return result.strip()


# =============================================================================
# EXTRACCION NER
# =============================================================================

class EntityExtractor:
    """Extrae entidades nombradas de documentos usando spaCy.

    Tipos NER relevantes para cross-linking multi-hop.
    Excluidos: CARDINAL, ORDINAL, QUANTITY, PERCENT (numericos sin identidad),
    DATE, TIME (muy genericos, no conectan documentos).
    """

    RELEVANT_NER_TYPES: Set[str] = {
        "PERSON", "ORG", "GPE", "LOC", "FAC",
        "EVENT", "WORK_OF_ART", "LAW", "PRODUCT", "NORP",
    }

    MIN_ENTITY_LENGTH: int = 2

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        if not HAS_SPACY:
            raise ImportError(
                "spacy no instalado. Instalar: pip install spacy && "
                "python -m spacy download en_core_web_sm"
            )
        # Solo necesitamos NER; desactivar parser y lemmatizer para velocidad.
        self._nlp = spacy.load(model_name, disable=["parser", "lemmatizer"])

    def extract(self, text: str) -> List[Tuple[str, str]]:
        """Extrae entidades de un texto.

        Args:
            text: Texto fuente (truncado internamente a 5000 chars).

        Returns:
            Lista de (nombre_normalizado, tipo_NER).
            Deduplicada por nombre normalizado (primera ocurrencia gana).
        """
        doc = self._nlp(text[:5000])
        seen: Dict[str, str] = {}
        for ent in doc.ents:
            if ent.label_ not in self.RELEVANT_NER_TYPES:
                continue
            normalized = normalize_entity(ent.text)
            if len(normalized) < self.MIN_ENTITY_LENGTH:
                continue
            if normalized not in seen:
                seen[normalized] = ent.label_
        return list(seen.items())


# =============================================================================
# ENTITY LINKER
# =============================================================================

class EntityLinker:
    """Indice invertido de entidades con IDF filter y generacion de cross-refs.

    Flujo:
      1. build_index(): construye indice invertido + aplica IDF filter
      2. generate_cross_refs(doc_id): genera texto de cross-refs para un doc
      3. compute_cross_refs(documents): orquesta pipeline completo NER->index->refs
    """

    def __init__(
        self,
        max_cross_refs: int = 3,
        min_shared_entities: int = 1,
        max_entity_doc_fraction: float = 0.05,
    ) -> None:
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
        self._filtered_entities: Set[str] = set()

    def build_index(self, doc_entities_list: List[DocEntities]) -> None:
        """Construye indice invertido a partir de entidades extraidas.

        Dos pasadas:
          1. Construir indice completo.
          2. Filtrar entidades con doc frequency > max_entity_doc_fraction.
             Threshold minimo absoluto: 10 docs (para corpus pequenos).
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
            threshold = max(threshold, 10)
            for entity, doc_set in self._entity_to_docs.items():
                if len(doc_set) > threshold:
                    self._filtered_entities.add(entity)

        n_filtered = len(self._filtered_entities)
        n_total = len(self._entity_to_docs)
        logger.info(
            f"EntityLinker: {self._total_docs} docs, "
            f"{n_total} entidades unicas, "
            f"{n_filtered} filtradas por IDF "
            f"(>{self.max_entity_doc_fraction:.0%})"
        )

    def generate_cross_refs(self, doc_id: str) -> str:
        """Genera texto de cross-references para un documento.

        Selecciona los top-N documentos con mayor numero de entidades
        compartidas (excluyendo las filtradas por IDF).

        Returns:
            Texto de cross-refs para anexar al documento antes de indexar.
            String vacio si no hay refs suficientes.
        """
        my_entities = self._doc_to_entities.get(doc_id, set())
        if not my_entities:
            return ""

        # Entidades validas (no filtradas por IDF)
        valid_entities = my_entities - self._filtered_entities
        if not valid_entities:
            return ""

        # Contar entidades compartidas con cada otro documento
        related: Counter[str] = Counter()
        shared_map: Dict[str, List[str]] = defaultdict(list)

        for entity in valid_entities:
            for other_doc_id in self._entity_to_docs[entity]:
                if other_doc_id != doc_id:
                    # Solo contar si la entidad no esta filtrada para el otro doc tambien
                    related[other_doc_id] += 1
                    shared_map[other_doc_id].append(entity)

        if not related:
            return ""

        # Filtrar por min_shared_entities y tomar top-N
        candidates = [
            (did, count)
            for did, count in related.most_common()
            if count >= self.min_shared_entities
        ]
        if not candidates:
            return ""

        top_refs = candidates[: self.max_cross_refs]

        # Generar texto de cross-references en lenguaje natural.
        # Formato optimizado para BM25 (bridge terms) y embeddings (semantica).
        ref_lines: List[str] = []
        for ref_doc_id, _shared_count in top_refs:
            title = self._doc_titles.get(ref_doc_id, ref_doc_id)
            entities_str = " and ".join(shared_map[ref_doc_id][:3])
            ref_lines.append(
                f"See also {title} regarding {entities_str}."
            )

        return " ".join(ref_lines)

    def get_cross_ref_graph(self, doc_id: str) -> List[str]:
        """Devuelve lista de doc_ids relacionados (para graph expansion).

        Misma logica que generate_cross_refs pero devuelve IDs en vez de texto.
        """
        my_entities = self._doc_to_entities.get(doc_id, set())
        if not my_entities:
            return []

        valid_entities = my_entities - self._filtered_entities
        if not valid_entities:
            return []

        related: Counter[str] = Counter()
        for entity in valid_entities:
            for other_doc_id in self._entity_to_docs[entity]:
                if other_doc_id != doc_id:
                    related[other_doc_id] += 1

        if not related:
            return []

        candidates = [
            did for did, count in related.most_common()
            if count >= self.min_shared_entities
        ]
        return candidates[: self.max_cross_refs]

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
        if not documents:
            return {}

        extractor = EntityExtractor()
        doc_entities_list: List[DocEntities] = []

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
        result: Dict[str, str] = {}
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

    def compute_cross_ref_graph(
        self,
        documents: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """Pipeline completo: NER -> build_index -> graph de cross-refs.

        Returns:
            Dict doc_id -> [related_doc_ids] ordenados por relevancia.
            Solo doc_ids con al menos un vecino.
        """
        if not documents:
            return {}

        extractor = EntityExtractor()
        doc_entities_list: List[DocEntities] = []

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
        graph: Dict[str, List[str]] = {}
        for de in doc_entities_list:
            neighbors = self.get_cross_ref_graph(de.doc_id)
            if neighbors:
                graph[de.doc_id] = neighbors
        link_ms = (time.perf_counter() - t1) * 1000

        n_with_refs = len(graph)
        logger.info(
            f"EntityLinker: NER {ner_ms:.0f}ms, "
            f"graph-linking {link_ms:.0f}ms, "
            f"{n_with_refs}/{len(documents)} docs con vecinos"
        )

        return graph

    def get_stats(self) -> Dict[str, Any]:
        """Estadisticas del indice para logging y diagnostico."""
        return {
            "total_docs": self._total_docs,
            "total_entities": len(self._entity_to_docs),
            "filtered_entities": len(self._filtered_entities),
            "docs_with_entities": sum(
                1 for ents in self._doc_to_entities.values() if ents
            ),
            "avg_entities_per_doc": (
                sum(len(e) for e in self._doc_to_entities.values())
                / self._total_docs
                if self._total_docs > 0
                else 0.0
            ),
        }


__all__ = [
    "HAS_SPACY",
    "DocEntities",
    "normalize_entity",
    "EntityExtractor",
    "EntityLinker",
]
