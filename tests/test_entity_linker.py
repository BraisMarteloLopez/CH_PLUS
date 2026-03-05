"""
Tests unitarios para shared/retrieval/entity_linker.py

Cubre: normalize_entity, EntityExtractor (mock), EntityLinker
(build_index, IDF filter, generate_cross_refs, compute_cross_refs, get_stats).

spaCy NO requerido: EntityExtractor se mockea inyectando entidades manuales.
"""

from unittest.mock import patch, MagicMock

from shared.retrieval.entity_linker import (
    normalize_entity,
    EntityExtractor,
    EntityLinker,
    DocEntities,
)


# =========================================================================
# normalize_entity
# =========================================================================

class TestNormalizeEntityBasic:
    """Tests basicos de normalizacion."""

    def test_lowercase(self):
        assert normalize_entity("Scott Derrickson") == "scott derrickson"

    def test_leading_article_the(self):
        assert normalize_entity("The United States") == "united states"

    def test_leading_article_a(self):
        assert normalize_entity("A Beautiful Mind") == "beautiful mind"

    def test_leading_article_an(self):
        assert normalize_entity("An Example") == "example"

    def test_punctuation_dots(self):
        assert normalize_entity("U.S.") == "us"

    def test_internal_hyphen_preserved(self):
        assert normalize_entity("Spider-Man") == "spider-man"

    def test_collapse_spaces(self):
        assert normalize_entity("  Scott   Derrickson  ") == "scott derrickson"


class TestNormalizeEntityEdgeCases:
    """Edge cases de normalizacion."""

    def test_empty_string(self):
        assert normalize_entity("") == ""

    def test_only_spaces(self):
        assert normalize_entity("   ") == ""

    def test_only_punctuation(self):
        result = normalize_entity("...")
        assert result == ""

    def test_internal_hyphen_complex(self):
        assert normalize_entity("Jean-Claude Van Damme") == "jean-claude van damme"

    def test_article_not_removed_if_part_of_name(self):
        # "the" solo se elimina al inicio
        result = normalize_entity("Catherine the Great")
        assert "the" in result


# =========================================================================
# EntityExtractor (con mock de spaCy)
# =========================================================================

def _make_mock_ent(text: str, label: str):
    """Crea un mock de entidad spaCy."""
    ent = MagicMock()
    ent.text = text
    ent.label_ = label
    return ent


def _make_mock_extractor(entities_per_text):
    """Crea un EntityExtractor con spaCy mockeado.

    Args:
        entities_per_text: dict text_prefix -> list of (text, label) tuples
                          o list of (text, label) para devolver siempre las mismas.
    """
    mock_nlp = MagicMock()

    def nlp_side_effect(text):
        doc = MagicMock()
        if isinstance(entities_per_text, list):
            doc.ents = [_make_mock_ent(t, l) for t, l in entities_per_text]
        else:
            doc.ents = []
            for prefix, ents in entities_per_text.items():
                if text.startswith(prefix):
                    doc.ents = [_make_mock_ent(t, l) for t, l in ents]
                    break
        return doc

    mock_nlp.side_effect = nlp_side_effect

    with patch("shared.retrieval.entity_linker.HAS_SPACY", True), \
         patch("shared.retrieval.entity_linker.spacy") as mock_spacy:
        mock_spacy.load.return_value = mock_nlp
        extractor = EntityExtractor()
    extractor._nlp = mock_nlp
    return extractor


class TestEntityExtractorBasic:
    """Tests de extraccion NER con mock."""

    def test_extracts_relevant_types(self):
        extractor = _make_mock_extractor([
            ("Scott Derrickson", "PERSON"),
            ("Sacramento", "GPE"),
            ("Lionsgate", "ORG"),
        ])
        result = extractor.extract("Scott Derrickson was born in Sacramento...")
        names = [name for name, _type in result]
        assert "scott derrickson" in names
        assert "sacramento" in names
        assert "lionsgate" in names

    def test_filters_irrelevant_types(self):
        extractor = _make_mock_extractor([
            ("Scott Derrickson", "PERSON"),
            ("42", "CARDINAL"),
            ("2012", "DATE"),
            ("50%", "PERCENT"),
        ])
        result = extractor.extract("Some text")
        names = [name for name, _type in result]
        assert "scott derrickson" in names
        assert len(result) == 1  # solo PERSON

    def test_deduplicates_by_normalized(self):
        extractor = _make_mock_extractor([
            ("Scott Derrickson", "PERSON"),
            ("SCOTT DERRICKSON", "PERSON"),  # duplicado normalizado
        ])
        result = extractor.extract("Some text")
        assert len(result) == 1
        assert result[0][0] == "scott derrickson"

    def test_min_length_filter(self):
        extractor = _make_mock_extractor([
            ("I", "PERSON"),      # len 1 -> filtrado
            ("AI", "ORG"),        # len 2 -> pasa
            ("US", "GPE"),        # len 2 -> pasa (tras normalizar u.s.)
        ])
        result = extractor.extract("Some text")
        names = [name for name, _type in result]
        assert "i" not in names
        assert "ai" in names


# =========================================================================
# EntityLinker.build_index
# =========================================================================

class TestBuildIndex:

    def test_basic_index(self):
        linker = EntityLinker()
        docs = [
            DocEntities("d1", "Doc 1", ["alice", "bob"], []),
            DocEntities("d2", "Doc 2", ["bob", "charlie"], []),
            DocEntities("d3", "Doc 3", ["charlie", "diana"], []),
        ]
        linker.build_index(docs)

        assert linker._total_docs == 3
        assert "alice" in linker._entity_to_docs
        assert linker._entity_to_docs["bob"] == {"d1", "d2"}
        assert linker._doc_to_entities["d1"] == {"alice", "bob"}

    def test_idf_filter(self):
        """Entidad en >5% docs (con min absoluto 10) se filtra."""
        linker = EntityLinker(max_entity_doc_fraction=0.05)
        # 20 docs, "common" en todos -> 20/20 = 100% > 5%, threshold = max(1, 10) = 10
        docs = [
            DocEntities(f"d{i}", f"Doc {i}", ["common", f"unique_{i}"], [])
            for i in range(20)
        ]
        linker.build_index(docs)

        assert "common" in linker._filtered_entities
        # Cada unique_N aparece en 1 doc -> no filtrado
        assert "unique_0" not in linker._filtered_entities

    def test_idf_filter_small_corpus_min_threshold(self):
        """Para corpus < 200, threshold minimo absoluto es 10."""
        linker = EntityLinker(max_entity_doc_fraction=0.05)
        # 5 docs, "shared" en todos -> 5/5 = 100% pero threshold = max(0, 10) = 10
        # 5 docs < 10 threshold -> NO se filtra
        docs = [
            DocEntities(f"d{i}", f"Doc {i}", ["shared"], [])
            for i in range(5)
        ]
        linker.build_index(docs)

        assert "shared" not in linker._filtered_entities


# =========================================================================
# EntityLinker.generate_cross_refs
# =========================================================================

class TestGenerateCrossRefs:

    def test_two_docs_shared_entity(self):
        linker = EntityLinker()
        docs = [
            DocEntities("d1", "Sinister (film)", ["scott derrickson", "sinister"], []),
            DocEntities("d2", "Scott Derrickson", ["scott derrickson", "sacramento"], []),
        ]
        linker.build_index(docs)

        refs_d1 = linker.generate_cross_refs("d1")
        assert "See also Scott Derrickson" in refs_d1
        assert "scott derrickson" in refs_d1

        refs_d2 = linker.generate_cross_refs("d2")
        assert "See also Sinister (film)" in refs_d2

    def test_no_shared_entities(self):
        linker = EntityLinker()
        docs = [
            DocEntities("d1", "Doc 1", ["alice"], []),
            DocEntities("d2", "Doc 2", ["bob"], []),
        ]
        linker.build_index(docs)

        assert linker.generate_cross_refs("d1") == ""
        assert linker.generate_cross_refs("d2") == ""

    def test_max_cross_refs(self):
        linker = EntityLinker(max_cross_refs=2)
        # d1 shares entity with d2, d3, d4 -> max 2 refs
        docs = [
            DocEntities("d1", "Main", ["entity_a"], []),
            DocEntities("d2", "Ref 1", ["entity_a"], []),
            DocEntities("d3", "Ref 2", ["entity_a"], []),
            DocEntities("d4", "Ref 3", ["entity_a"], []),
        ]
        linker.build_index(docs)

        refs = linker.generate_cross_refs("d1")
        assert refs.count("See also") == 2

    def test_min_shared_entities(self):
        linker = EntityLinker(min_shared_entities=2)
        docs = [
            DocEntities("d1", "A", ["x", "y", "z"], []),
            DocEntities("d2", "B", ["x", "y"], []),        # shares 2 -> passes
            DocEntities("d3", "C", ["x"], []),              # shares 1 -> filtered
        ]
        linker.build_index(docs)

        refs = linker.generate_cross_refs("d1")
        assert "B" in refs
        assert "C" not in refs

    def test_sorted_by_overlap(self):
        linker = EntityLinker(max_cross_refs=3)
        docs = [
            DocEntities("d1", "Main", ["a", "b", "c"], []),
            DocEntities("d2", "Less overlap", ["a"], []),           # 1 shared
            DocEntities("d3", "More overlap", ["a", "b", "c"], []),  # 3 shared
        ]
        linker.build_index(docs)

        refs = linker.generate_cross_refs("d1")
        # d3 (3 shared) should appear before d2 (1 shared)
        pos_more = refs.index("More overlap")
        pos_less = refs.index("Less overlap")
        assert pos_more < pos_less

    def test_filtered_entity_excluded(self):
        """Entidad IDF-filtrada no genera cross-refs."""
        linker = EntityLinker(max_entity_doc_fraction=0.05)
        # "common" in all 20 docs -> filtered (threshold = max(1, 10) = 10, 20 > 10)
        docs = [
            DocEntities(f"d{i}", f"Doc {i}", ["common"], [])
            for i in range(20)
        ]
        linker.build_index(docs)

        # Todos solo tienen "common" (filtrada) -> sin refs
        assert linker.generate_cross_refs("d0") == ""

    def test_unknown_doc_id(self):
        linker = EntityLinker()
        linker.build_index([DocEntities("d1", "D1", ["x"], [])])
        assert linker.generate_cross_refs("nonexistent") == ""

    def test_doc_without_entities(self):
        linker = EntityLinker()
        docs = [
            DocEntities("d1", "D1", [], []),
            DocEntities("d2", "D2", ["x"], []),
        ]
        linker.build_index(docs)
        assert linker.generate_cross_refs("d1") == ""


# =========================================================================
# EntityLinker.compute_cross_refs
# =========================================================================

class TestComputeCrossRefs:

    def test_pipeline_with_mock_extractor(self):
        """Pipeline completo: inyecta entidades via mock, verifica refs."""
        mock_entities = {
            "d1": [("Scott Derrickson", "PERSON"), ("Sinister", "WORK_OF_ART")],
            "d2": [("Scott Derrickson", "PERSON"), ("Sacramento", "GPE")],
            "d3": [("Python", "ORG")],
        }

        mock_extractor_instance = MagicMock()
        mock_extractor_instance.extract.side_effect = lambda text: [
            (normalize_entity(name), typ)
            for name, typ in mock_entities.get(text, [])
        ]

        linker = EntityLinker()
        documents = [
            {"doc_id": "d1", "content": "d1", "title": "Sinister (film)"},
            {"doc_id": "d2", "content": "d2", "title": "Scott Derrickson"},
            {"doc_id": "d3", "content": "d3", "title": "Python Language"},
        ]

        with patch.object(EntityLinker, "compute_cross_refs", wraps=linker.compute_cross_refs):
            with patch("shared.retrieval.entity_linker.EntityExtractor", return_value=mock_extractor_instance):
                result = linker.compute_cross_refs(documents)

        assert "d1" in result
        assert "Scott Derrickson" in result["d1"]
        assert "d2" in result
        assert "Sinister" in result["d2"]
        # d3 has no shared entities
        assert "d3" not in result

    def test_empty_documents(self):
        linker = EntityLinker()
        result = linker.compute_cross_refs([])
        assert result == {}


# =========================================================================
# EntityLinker.get_stats
# =========================================================================

class TestGetStats:

    def test_stats_fields(self):
        linker = EntityLinker()
        docs = [
            DocEntities("d1", "D1", ["alice", "bob"], []),
            DocEntities("d2", "D2", ["bob"], []),
            DocEntities("d3", "D3", [], []),
        ]
        linker.build_index(docs)

        stats = linker.get_stats()
        assert stats["total_docs"] == 3
        assert stats["total_entities"] == 2  # alice, bob
        assert stats["filtered_entities"] == 0
        assert stats["docs_with_entities"] == 2  # d1, d2 (d3 empty)
        assert stats["avg_entities_per_doc"] == 1.0  # (2 + 1 + 0) / 3

    def test_stats_empty(self):
        linker = EntityLinker()
        linker.build_index([])
        stats = linker.get_stats()
        assert stats["total_docs"] == 0
        assert stats["avg_entities_per_doc"] == 0.0
