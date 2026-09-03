"""
Neo4j client for knowledge graph operations.
Handles entity/relationship MERGE, deletion, and querying on Neo4j AuraDB.
Optimized with UNWIND batching for cloud database speed.
"""

from __future__ import annotations

import logging
from neo4j import GraphDatabase, Driver

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j graph database client."""

    def __init__(self, uri: str, user: str, password: str):
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Neo4j driver created for %s", uri)

    def close(self) -> None:
        self._driver.close()

    def verify_connection(self) -> bool:
        """Test connectivity."""
        try:
            with self._driver.session() as session:
                session.run("RETURN 1")
            return True
        except Exception as e:
            logger.error("Neo4j connection failed: %s", str(e))
            return False

    def setup_indexes(self) -> None:
        """Create indexes for efficient lookups."""
        with self._driver.session() as session:
            # Index on entity name for fast MERGE
            session.run(
                "CREATE INDEX entity_name_idx IF NOT EXISTS "
                "FOR (e:Entity) ON (e.name)"
            )
            # Index on doc_ids for cascade deletes
            session.run(
                "CREATE INDEX entity_doc_idx IF NOT EXISTS "
                "FOR (e:Entity) ON (e.doc_ids)"
            )
            logger.info("Neo4j indexes created/verified")

    def merge_entities(self, entities: list[dict], doc_id: str) -> None:
        """
        Batched MERGE of entities into the graph in a single roundtrip.
        Each entity: {name, type, description, chunk_ids}
        """
        if not entities:
            return

        with self._driver.session() as session:
            session.run(
                """
                UNWIND $entities AS item
                MERGE (e:Entity {name: item.name})
                ON CREATE SET
                    e.type = item.type,
                    e.description = item.description,
                    e.chunk_ids = item.chunk_ids,
                    e.doc_ids = [$doc_id]
                ON MATCH SET
                    e.type = CASE WHEN size(coalesce(e.description, '')) < size(coalesce(item.description, ''))
                                  THEN item.type ELSE e.type END,
                    e.description = CASE WHEN size(coalesce(e.description, '')) < size(coalesce(item.description, ''))
                                         THEN item.description ELSE e.description END,
                    e.chunk_ids = [x IN e.chunk_ids + item.chunk_ids | x],
                    e.doc_ids = CASE WHEN NOT $doc_id IN e.doc_ids
                                     THEN e.doc_ids + [$doc_id]
                                     ELSE e.doc_ids END
                """,
                entities=entities,
                doc_id=doc_id,
            )

        logger.info("Batched merged %d entities for doc_id=%s", len(entities), doc_id)

    def create_relationships(self, relationships: list[dict], doc_id: str) -> None:
        """
        Batched creation of relationships between entities in a single roundtrip.
        Each rel: {source, relation, target, description, chunk_ids}
        """
        if not relationships:
            return

        with self._driver.session() as session:
            session.run(
                """
                UNWIND $relationships AS item
                MATCH (s:Entity {name: item.source})
                MATCH (t:Entity {name: item.target})
                MERGE (s)-[r:RELATES_TO {type: item.relation}]->(t)
                ON CREATE SET
                    r.description = item.description,
                    r.chunk_ids = item.chunk_ids,
                    r.doc_ids = [$doc_id]
                ON MATCH SET
                    r.description = CASE WHEN size(coalesce(r.description, '')) < size(coalesce(item.description, ''))
                                         THEN item.description ELSE r.description END,
                    r.chunk_ids = [x IN r.chunk_ids + item.chunk_ids | x],
                    r.doc_ids = CASE WHEN NOT $doc_id IN r.doc_ids
                                     THEN r.doc_ids + [$doc_id]
                                     ELSE r.doc_ids END
                """,
                relationships=relationships,
                doc_id=doc_id,
            )

        logger.info("Batched created %d relationships for doc_id=%s", len(relationships), doc_id)

    def delete_document_graph(self, doc_id: str) -> dict:
        """
        Delete all entities and relationships that ONLY belong to this document.
        Entities shared across documents just have this doc_id removed.
        """
        with self._driver.session() as session:
            # Remove doc_id from shared entities
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE $doc_id IN e.doc_ids AND size(e.doc_ids) > 1
                SET e.doc_ids = [d IN e.doc_ids WHERE d <> $doc_id]
                RETURN count(e) as updated
                """,
                doc_id=doc_id,
            )
            updated = result.single()["updated"]

            # Delete entities exclusive to this document
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.doc_ids = [$doc_id] OR (size(e.doc_ids) = 1 AND $doc_id IN e.doc_ids)
                DETACH DELETE e
                RETURN count(e) as deleted
                """,
                doc_id=doc_id,
            )
            deleted = result.single()["deleted"]

            # Delete document relationships
            session.run(
                """
                MATCH ()-[r:RELATES_TO]->()
                WHERE r.doc_ids = [$doc_id]
                DELETE r
                """,
                doc_id=doc_id,
            )

            logger.info(
                "Deleted graph for doc_id=%s: %d nodes deleted, %d shared nodes updated",
                doc_id, deleted, updated,
            )
            return {"deleted": deleted, "updated": updated}

    def find_entities_by_names(self, names: list[str]) -> list[dict]:
        """Find entities by name (case-insensitive partial match)."""
        if not names:
            return []

        with self._driver.session() as session:
            result = session.run(
                """
                UNWIND $names AS name
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower(name) OR toLower(name) CONTAINS toLower(e.name)
                RETURN DISTINCT e.name AS name, e.type AS type,
                       e.description AS description, e.chunk_ids AS chunk_ids
                LIMIT 25
                """,
                names=names,
            )
            return [dict(record) for record in result]

    def find_entities_by_chunk_ids(self, chunk_ids: list[str], limit: int = 30) -> list[dict]:
        """Find entities directly associated with a list of chunk IDs."""
        if not chunk_ids:
            return []

        with self._driver.session() as session:
            result = session.run(
                """
                UNWIND $chunk_ids AS cid
                MATCH (e:Entity)
                WHERE cid IN e.chunk_ids
                RETURN DISTINCT e.name AS name, coalesce(e.type, 'Entity') AS type, e.doc_ids AS doc_ids
                LIMIT $limit
                """,
                chunk_ids=chunk_ids,
                limit=limit,
            )
            return [dict(record) for record in result]

    def expand_from_entities(
        self,
        entity_names: list[str],
        max_hops: int = 2,
        max_facts: int = 45,
        per_seed_limit: int = 10,
    ) -> list[dict]:
        """
        Traverse the graph bidirectionally from seed entities with balanced per-seed quotas.
        Ensures multiple topics or documents are not starved by one highly-connected entity.
        """
        if not entity_names:
            return []

        with self._driver.session() as session:
            result = session.run(
                """
                UNWIND $names AS seed_name
                MATCH (start:Entity)
                WHERE toLower(start.name) CONTAINS toLower(seed_name) OR toLower(seed_name) CONTAINS toLower(start.name)
                CALL (start) {
                    MATCH path = (start)-[r:RELATES_TO*1..""" + str(max_hops) + """]-(end:Entity)
                    WITH relationships(path) AS rels
                    UNWIND rels AS rel
                    WITH DISTINCT rel, startNode(rel) AS s, endNode(rel) AS t
                    RETURN DISTINCT
                        s.name AS source_entity,
                        coalesce(s.type, 'Concept') AS source_type,
                        rel.type AS relation,
                        t.name AS target_entity,
                        coalesce(t.type, 'Entity') AS target_type,
                        coalesce(rel.description, '') AS description,
                        coalesce(rel.chunk_ids, []) AS source_chunk_ids
                    LIMIT $per_seed_limit
                }
                RETURN DISTINCT
                    source_entity,
                    source_type,
                    relation,
                    target_entity,
                    target_type,
                    description,
                    source_chunk_ids
                LIMIT $max_facts
                """,
                names=entity_names,
                max_facts=max_facts,
                per_seed_limit=per_seed_limit,
            )
            facts = [dict(record) for record in result]

            logger.info(
                "Balanced bidirectional graph expansion from %d seed entities → %d facts",
                len(entity_names), len(facts),
            )
            return facts

    def get_graph_summary(self) -> dict:
        """Get summary statistics of the graph."""
        with self._driver.session() as session:
            node_count = session.run("MATCH (e:Entity) RETURN count(e) as count").single()["count"]
            rel_count = session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) as count").single()["count"]
            return {"entity_count": node_count, "relationship_count": rel_count}

    def get_document_subgraph(self, doc_id: str, limit: int = 1000) -> dict:
        """Get the entity-relationship subgraph for a specific document with all connected nodes guaranteed."""
        with self._driver.session() as session:
            # 1. Fetch relationships belonging to this document
            edges_result = session.run(
                """
                MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
                WHERE $doc_id IN r.doc_ids
                RETURN s.name AS source, r.type AS relation, t.name AS target,
                       coalesce(r.description, '') AS description
                LIMIT $limit
                """,
                doc_id=doc_id,
                limit=limit,
            )
            edges = [dict(r) for r in edges_result]

            # 2. Fetch all nodes belonging to this document
            nodes_result = session.run(
                """
                MATCH (e:Entity)
                WHERE $doc_id IN e.doc_ids
                RETURN e.name AS name, e.type AS type, coalesce(e.description, '') AS description
                LIMIT $limit
                """,
                doc_id=doc_id,
                limit=limit,
            )
            nodes_dict = {r["name"]: dict(r) for r in nodes_result}

            # 3. Ensure any node linked by edges is included so no edge is ever dropped in frontend
            edge_node_names = set()
            for e in edges:
                edge_node_names.add(e["source"])
                edge_node_names.add(e["target"])

            missing_names = edge_node_names - set(nodes_dict.keys())
            if missing_names:
                missing_result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.name IN $missing_names
                    RETURN e.name AS name, e.type AS type, coalesce(e.description, '') AS description
                    """,
                    missing_names=list(missing_names),
                )
                for r in missing_result:
                    nodes_dict[r["name"]] = dict(r)

            return {"nodes": list(nodes_dict.values()), "edges": edges}

    def get_global_subgraph(self, limit: int = 1500) -> dict:
        """Get the entire knowledge graph across all documents with guaranteed node endpoints."""
        with self._driver.session() as session:
            edges_result = session.run(
                """
                MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
                RETURN s.name AS source, r.type AS relation, t.name AS target,
                       coalesce(r.description, '') AS description
                LIMIT $limit
                """,
                limit=limit,
            )
            edges = [dict(r) for r in edges_result]

            nodes_result = session.run(
                """
                MATCH (e:Entity)
                RETURN e.name AS name, e.type AS type, coalesce(e.description, '') AS description
                LIMIT $limit
                """,
                limit=limit,
            )
            nodes_dict = {r["name"]: dict(r) for r in nodes_result}

            edge_node_names = set()
            for e in edges:
                edge_node_names.add(e["source"])
                edge_node_names.add(e["target"])

            missing_names = edge_node_names - set(nodes_dict.keys())
            if missing_names:
                missing_result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.name IN $missing_names
                    RETURN e.name AS name, e.type AS type, coalesce(e.description, '') AS description
                    """,
                    missing_names=list(missing_names),
                )
                for r in missing_result:
                    nodes_dict[r["name"]] = dict(r)

            return {"nodes": list(nodes_dict.values()), "edges": edges}

    def get_answer_subgraph(self, entity_names: list[str], max_hops: int = 1) -> dict:
        """
        Get the subgraph around entities relevant to an answer.
        Used for graph visualization in the frontend.
        """
        if not entity_names:
            return {"nodes": [], "edges": []}

        with self._driver.session() as session:
            nodes_result = session.run(
                """
                UNWIND $names AS name
                MATCH (start:Entity)
                WHERE toLower(start.name) CONTAINS toLower(name) OR toLower(name) CONTAINS toLower(start.name)
                MATCH path = (start)-[r:RELATES_TO*0..""" + str(max_hops) + """]-(connected:Entity)
                WITH DISTINCT connected
                RETURN connected.name AS name, connected.type AS type,
                       coalesce(connected.description, '') AS description
                LIMIT 40
                """,
                names=entity_names,
            )
            nodes = [dict(r) for r in nodes_result]

            node_names = [n["name"] for n in nodes]
            edges_result = session.run(
                """
                MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
                WHERE s.name IN $node_names AND t.name IN $node_names
                RETURN s.name AS source, r.type AS relation, t.name AS target,
                       coalesce(r.description, '') AS description
                LIMIT 60
                """,
                node_names=node_names,
            )
            edges = [dict(r) for r in edges_result]

            return {"nodes": nodes, "edges": edges}
