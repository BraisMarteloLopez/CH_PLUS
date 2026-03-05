"""
Modulo: Knowledge Graph
Descripcion: Grafo de conocimiento in-memory basado en NetworkX.
             Almacena entidades y relaciones extraidas por LLM,
             soporta traversal multi-hop y busqueda por entidades.

Ubicacion: shared/retrieval/knowledge_graph.py

Uso: LIGHT_RAG construye el grafo durante indexacion y lo consulta
durante retrieval para complementar busqueda vectorial.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# IMPORTACION CONDICIONAL — NetworkX
# =============================================================================

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None  # type: ignore[assignment]


# =============================================================================
# TIPOS
# =============================================================================

@dataclass
class KGEntity:
    """Entidad extraida por LLM."""
    name: str                       # nombre normalizado
    entity_type: str                # PERSON, ORG, CONCEPT, etc.
    description: str = ""           # descripcion del LLM
    source_doc_ids: Set[str] = field(default_factory=set)


@dataclass
class KGRelation:
    """Relacion (tripleta) entre dos entidades."""
    source: str                     # nombre entidad origen (normalizado)
    target: str                     # nombre entidad destino (normalizado)
    relation: str                   # tipo de relacion
    description: str = ""           # descripcion del LLM
    source_doc_id: str = ""         # doc de donde se extrajo


# =============================================================================
# KNOWLEDGE GRAPH
# =============================================================================

class KnowledgeGraph:
    """Knowledge graph in-memory basado en NetworkX.

    Operaciones principales:
      - add_triplets(): construir grafo incrementalmente
      - query_entities(): traversal por entidades especificas (low-level)
      - query_by_keywords(): busqueda por keywords en descripciones (high-level)
      - get_subgraph_context(): generar texto de contexto del subgrafo
    """

    def __init__(self) -> None:
        if not HAS_NETWORKX:
            raise ImportError(
                "networkx no instalado. Instalar: pip install networkx"
            )
        self._graph: nx.Graph = nx.Graph()
        # Indices invertidos
        self._entity_to_docs: Dict[str, Set[str]] = defaultdict(set)
        self._doc_to_entities: Dict[str, Set[str]] = defaultdict(set)
        # Entidades con metadata
        self._entities: Dict[str, KGEntity] = {}
        # Relaciones por doc
        self._doc_to_relations: Dict[str, List[KGRelation]] = defaultdict(list)

    @property
    def num_entities(self) -> int:
        return len(self._entities)

    @property
    def num_relations(self) -> int:
        return self._graph.number_of_edges()

    @property
    def num_docs(self) -> int:
        return len(self._doc_to_entities)

    def _normalize_name(self, name: str) -> str:
        """Normalizacion simple de nombres de entidad."""
        return name.lower().strip()

    def add_triplets(self, doc_id: str, triplets: List[KGRelation]) -> int:
        """Anade tripletas al grafo desde un documento.

        Deduplica entidades por nombre normalizado. Multiples docs pueden
        contribuir a la misma entidad (merge por union de source_doc_ids).

        Args:
            doc_id: ID del documento fuente.
            triplets: Lista de relaciones extraidas por LLM.

        Returns:
            Numero de tripletas anadidas efectivamente.
        """
        added = 0
        for triplet in triplets:
            src_name = self._normalize_name(triplet.source)
            tgt_name = self._normalize_name(triplet.target)

            if not src_name or not tgt_name:
                continue

            # Registrar/actualizar entidades
            for name in (src_name, tgt_name):
                if name not in self._entities:
                    self._entities[name] = KGEntity(
                        name=name,
                        entity_type="UNKNOWN",
                        source_doc_ids={doc_id},
                    )
                else:
                    self._entities[name].source_doc_ids.add(doc_id)
                self._entity_to_docs[name].add(doc_id)
                self._doc_to_entities[doc_id].add(name)

            # Anadir nodos y arista al grafo
            self._graph.add_node(src_name, entity_type=self._entities[src_name].entity_type)
            self._graph.add_node(tgt_name, entity_type=self._entities[tgt_name].entity_type)

            # Si ya existe arista, acumular relaciones
            if self._graph.has_edge(src_name, tgt_name):
                existing = self._graph[src_name][tgt_name].get("relations", [])
                existing.append({
                    "relation": triplet.relation,
                    "description": triplet.description,
                    "doc_id": doc_id,
                })
                self._graph[src_name][tgt_name]["relations"] = existing
            else:
                self._graph.add_edge(
                    src_name, tgt_name,
                    relations=[{
                        "relation": triplet.relation,
                        "description": triplet.description,
                        "doc_id": doc_id,
                    }],
                )

            self._doc_to_relations[doc_id].append(triplet)
            added += 1

        return added

    def add_entity_metadata(
        self, name: str, entity_type: str, description: str
    ) -> None:
        """Actualiza metadata de una entidad (tipo, descripcion del LLM)."""
        norm = self._normalize_name(name)
        if norm in self._entities:
            if entity_type and entity_type != "UNKNOWN":
                self._entities[norm].entity_type = entity_type
            if description:
                self._entities[norm].description = description
            # Actualizar atributo del nodo
            if self._graph.has_node(norm):
                self._graph.nodes[norm]["entity_type"] = self._entities[norm].entity_type

    def query_entities(
        self,
        entity_names: List[str],
        max_hops: int = 2,
        max_docs: int = 20,
    ) -> List[Tuple[str, float]]:
        """Low-level retrieval: busca doc_ids conectados a entidades dadas.

        Traversal BFS hasta max_hops desde cada entidad query.
        Scoring: docs mas cercanos (menos hops) reciben score mas alto.

        Args:
            entity_names: Nombres de entidades a buscar.
            max_hops: Profundidad maxima de traversal.
            max_docs: Numero maximo de doc_ids a devolver.

        Returns:
            Lista de (doc_id, score) ordenada por score descendente.
        """
        doc_scores: Counter = Counter()

        for name in entity_names:
            norm = self._normalize_name(name)
            if norm not in self._entities or not self._graph.has_node(norm):
                continue

            # BFS con distancia
            visited: Set[str] = set()
            queue: List[Tuple[str, int]] = [(norm, 0)]
            visited.add(norm)

            while queue:
                current, depth = queue.pop(0)
                if depth > max_hops:
                    continue

                # Score inversamente proporcional a la distancia
                hop_score = 1.0 / (1.0 + depth)

                # Documentos asociados a esta entidad
                for doc_id in self._entity_to_docs.get(current, set()):
                    doc_scores[doc_id] += hop_score

                # Expandir vecinos
                if depth < max_hops:
                    for neighbor in self._graph.neighbors(current):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, depth + 1))

        # Ordenar por score y limitar
        ranked = doc_scores.most_common(max_docs)
        return ranked

    def query_by_keywords(
        self,
        keywords: List[str],
        max_docs: int = 20,
    ) -> List[Tuple[str, float]]:
        """High-level retrieval: busca docs por keywords en nombres de entidad
        y descripciones de relaciones.

        Matching simple por substring en entidades y relaciones.

        Args:
            keywords: Temas/keywords de alto nivel.
            max_docs: Numero maximo de doc_ids a devolver.

        Returns:
            Lista de (doc_id, score) ordenada por score descendente.
        """
        doc_scores: Counter = Counter()
        keywords_lower = [k.lower().strip() for k in keywords if k.strip()]

        if not keywords_lower:
            return []

        # Buscar en nombres de entidad
        for entity_name, entity in self._entities.items():
            for kw in keywords_lower:
                if kw in entity_name or kw in entity.description.lower():
                    for doc_id in entity.source_doc_ids:
                        doc_scores[doc_id] += 1.0

        # Buscar en descripciones de relaciones
        for _src, _tgt, edge_data in self._graph.edges(data=True):
            for rel_info in edge_data.get("relations", []):
                rel_text = (
                    rel_info.get("relation", "") + " " +
                    rel_info.get("description", "")
                ).lower()
                for kw in keywords_lower:
                    if kw in rel_text:
                        doc_id = rel_info.get("doc_id", "")
                        if doc_id:
                            doc_scores[doc_id] += 0.5

        return doc_scores.most_common(max_docs)

    def get_subgraph_context(
        self,
        entity_names: List[str],
        max_hops: int = 1,
        max_relations: int = 20,
    ) -> str:
        """Genera texto de contexto del subgrafo para enriquecer la query.

        Recorre el subgrafo alrededor de las entidades dadas y genera
        texto con las relaciones encontradas.

        Returns:
            Texto descriptivo de relaciones relevantes.
        """
        lines: List[str] = []
        seen_edges: Set[Tuple[str, str]] = set()

        for name in entity_names:
            norm = self._normalize_name(name)
            if not self._graph.has_node(norm):
                continue

            # Subgrafo local
            ego = nx.ego_graph(self._graph, norm, radius=max_hops)
            for src, tgt, data in ego.edges(data=True):
                edge_key = (min(src, tgt), max(src, tgt))
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                for rel_info in data.get("relations", []):
                    rel = rel_info.get("relation", "related to")
                    lines.append(f"{src} {rel} {tgt}")

                    if len(lines) >= max_relations:
                        break
                if len(lines) >= max_relations:
                    break
            if len(lines) >= max_relations:
                break

        return ". ".join(lines) + "." if lines else ""

    def get_stats(self) -> Dict[str, Any]:
        """Estadisticas del grafo para logging."""
        return {
            "num_entities": self.num_entities,
            "num_relations": self.num_relations,
            "num_docs_with_entities": self.num_docs,
            "avg_entities_per_doc": (
                sum(len(e) for e in self._doc_to_entities.values()) / self.num_docs
                if self.num_docs > 0
                else 0.0
            ),
            "graph_connected_components": (
                nx.number_connected_components(self._graph)
                if self._graph.number_of_nodes() > 0
                else 0
            ),
        }


__all__ = [
    "HAS_NETWORKX",
    "KGEntity",
    "KGRelation",
    "KnowledgeGraph",
]
