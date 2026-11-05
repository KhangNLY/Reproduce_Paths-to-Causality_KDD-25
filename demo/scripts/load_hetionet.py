from __future__ import annotations
import argparse, csv, gzip, os, sys
from pathlib import Path
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
NODES_FILE = DATA / "hetionet-v1.0-nodes.tsv"
EDGES_FILE = DATA / "hetionet-v1.0-edges.sif.gz"
URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "local-demo-change-me")
BATCH_SIZE = 3000

def batched(rows, size=BATCH_SIZE):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch

def read_nodes():
    with NODES_FILE.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
        needed = {"id", "name", "kind"}
        if not reader.fieldnames or not needed.issubset(set(reader.fieldnames)):
            raise RuntimeError(f"Unexpected nodes header: {reader.fieldnames}")
        for r in reader:
            yield {"id": r["id"], "name": r["name"], "kind": r["kind"]}

def read_edges():
    with gzip.open(EDGES_FILE, "rt", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        first = True
        for row in reader:
            if not row or len(row) < 3:
                continue
            if first:
                first = False
                head = [x.strip().lower() for x in row[:3]]
                if head[0] in {"source", "source_id"} and head[2] in {"target", "target_id"}:
                    continue
            yield {"source": row[0], "metaedge": row[1], "target": row[2]}

def main():
    parser = argparse.ArgumentParser(description="Load Hetionet v1.0 TSV/SIF into local Neo4j")
    parser.add_argument("--reset", action="store_true", help="Delete the existing Entity/EDGE graph before loading")
    args = parser.parse_args()
    if not NODES_FILE.exists() or not EDGES_FILE.exists():
        sys.exit("Missing raw data. Run: bash scripts/download_hetionet.sh")
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    try:
        with driver.session() as session:
            current = session.run("MATCH (n:Entity) RETURN count(n) AS n").single()["n"]
            if current and not args.reset:
                sys.exit(f"Neo4j already has {current} Entity nodes. Use --reset explicitly to reload the graph.")
            if args.reset:
                print("Deleting existing Entity graph…", flush=True)
                session.run("MATCH (n:Entity) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 1000 ROWS")
            session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
        print(f"Loading nodes from {NODES_FILE} …", flush=True)
        total_nodes = 0
        for batch in batched(read_nodes()):
            with driver.session() as session:
                session.run("UNWIND $rows AS row CREATE (:Entity {id:row.id, name:row.name, kind:row.kind})", rows=batch)
            total_nodes += len(batch)
            if total_nodes % 15000 < BATCH_SIZE:
                print(f"  nodes: {total_nodes:,}", flush=True)
        print(f"Loaded {total_nodes:,} nodes. Loading edges (this may take a while)…", flush=True)
        total_edges = 0
        for batch in batched(read_edges()):
            with driver.session() as session:
                session.run("""
                    UNWIND $rows AS row
                    MATCH (s:Entity {id: row.source})
                    MATCH (t:Entity {id: row.target})
                    CREATE (s)-[:EDGE {metaedge: row.metaedge}]->(t)
                """, rows=batch)
            total_edges += len(batch)
            if total_edges % 50000 < BATCH_SIZE:
                print(f"  edges: {total_edges:,}", flush=True)
        with driver.session() as session:
            stats = session.run("MATCH (n:Entity) WITH count(n) AS nodes MATCH ()-[r:EDGE]->() RETURN nodes, count(r) AS edges").single()
            print(f"DONE: {stats['nodes']:,} nodes, {stats['edges']:,} edges")
    finally:
        driver.close()

if __name__ == "__main__":
    main()
