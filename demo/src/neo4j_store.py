from __future__ import annotations
from typing import Any
from neo4j import GraphDatabase
from src.config import settings

class KGStore:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            connection_timeout=10,
        )

    def close(self):
        self.driver.close()

    def verify(self):
        with self.driver.session() as s:
            return s.run("RETURN 1 AS ok").single()["ok"] == 1

    def stats(self) -> dict[str, int]:
        with self.driver.session() as s:
            nodes = s.run("MATCH (n:Entity) RETURN count(n) AS n").single()["n"]
            edges = s.run("MATCH ()-[r:EDGE]->() RETURN count(r) AS n").single()["n"]
        return {"nodes": nodes, "edges": edges}

    def search_entities(self, q: str, limit: int = 15) -> list[dict[str, Any]]:
        q = q.strip()
        if not q:
            return []
        cypher = """
        MATCH (n:Entity)
        WHERE toLower(n.name) CONTAINS toLower($q) OR toLower(n.id) CONTAINS toLower($q)
        RETURN n.id AS id, n.name AS name, n.kind AS kind,
               CASE WHEN toLower(n.name) = toLower($q) THEN 0
                    WHEN toLower(n.name) STARTS WITH toLower($q) THEN 1 ELSE 2 END AS rank
        ORDER BY rank, n.name
        LIMIT $limit
        """
        with self.driver.session() as s:
            return [dict(r) for r in s.run(cypher, q=q, limit=limit)]

    def resolve_entity(self, q: str, preferred_kind: str | None = None) -> dict[str, Any] | None:
        """Resolve only a unique exact name/ID; do not silently map benchmark aliases.

        The UI uses search_entities() for interactive partial-name suggestions.
        Evaluation/data-preparation need stricter resolution so a fuzzy first match
        cannot silently pair a benchmark row with the wrong biomedical entity.
        Add a curated alias map upstream when the benchmark uses alternate names.
        """
        q = q.strip()
        if not q:
            return None
        cypher = """
        MATCH (n:Entity)
        WHERE toLower(n.name) = toLower($q) OR toLower(n.id) = toLower($q)
        RETURN n.id AS id, n.name AS name, n.kind AS kind
        ORDER BY n.id
        LIMIT 2
        """
        with self.driver.session() as session:
            exact = [dict(r) for r in session.run(cypher, q=q)]
        if preferred_kind:
            preferred = [row for row in exact if row["kind"].lower() == preferred_kind.lower()]
            if len(preferred) == 1:
                return preferred[0]
        return exact[0] if len(exact) == 1 else None

    def shortest_paths(self, source_id: str, target_id: str, max_hops: int = 4, limit: int = 30) -> list[dict[str, Any]]:
        if source_id == target_id:
            return []
        # max_hops/limit are int settings, not user-supplied Cypher fragments.
        cypher = f"""
        MATCH (s:Entity {{id: $source_id}}), (t:Entity {{id: $target_id}})
        MATCH p = allShortestPaths((s)-[:EDGE*..{int(max_hops)}]-(t))
        WITH p LIMIT $limit
        RETURN
          [n IN nodes(p) | {{id:n.id, name:n.name, kind:n.kind}}] AS path_nodes,
          [r IN relationships(p) | {{type:r.metaedge,
                                    source:startNode(r).id,
                                    target:endNode(r).id}}] AS path_edges,
          length(p) AS hops
        """
        with self.driver.session() as s:
            return [dict(r) for r in s.run(cypher, source_id=source_id, target_id=target_id, limit=int(limit))]

    def direct_relations(self, source_id: str, target_id: str) -> list[dict[str, Any]]:
        """Return every stored edge directly connecting the selected entities."""
        cypher = """
        MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id})
        MATCH (a)-[r:EDGE]-(b)
        WITH startNode(r) AS source, endNode(r) AS target, r
        RETURN source.id AS source, source.name AS source_name, source.kind AS source_kind,
               target.id AS target, target.name AS target_name, target.kind AS target_kind,
               r.metaedge AS type
        ORDER BY type
        """
        with self.driver.session() as session:
            return [dict(row) for row in session.run(cypher, source_id=source_id, target_id=target_id)]

    def close_path_neighborhood(self, source_id: str, target_id: str, max_hops: int = 2, limit: int = 80):
        paths = self.shortest_paths(source_id, target_id, max_hops=max_hops, limit=limit)
        return paths
