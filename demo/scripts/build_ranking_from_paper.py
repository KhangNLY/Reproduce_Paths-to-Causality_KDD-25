"""Build ranking_train.csv from the paper's own Subgraph Relevancy Estimation labels.

This replaces the local-LLM pseudo-label path (prepare_ranker_data.py) with the
real Module-1 output shipped in the benchmark files: each metapath carries the
paper's `relevant` label derived from the 1 ± log-prob relevancy formula.

Entity handling notes (verified against the local Hetionet load):
- `resolve_entity` returns None for names matching two nodes (e.g. "breast
  cancer" is both Disease and Side Effect), so a curated alias/preferred-kind
  map resolves benchmark names to the Hetionet node the paper's metapaths use.
- The paper's metapaths include paths of ANY length 1..4 hops, while
  allShortestPaths only returns shortest ones; missing labels are simply
  skipped (unlabeled paths cannot train the ranker).
- SEMEVAL is excluded: in the paper it is evaluated against Wikidata, not
  Hetionet, so its metapaths do not exist in this KG.
"""
from __future__ import annotations
import argparse, csv, json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings
from src.features import FEATURE_NAMES, path_features, path_text
from src.neo4j_store import KGStore

# Benchmark name -> (Hetionet name or None, preferred kind or None)
ALIASES = {
    "hypercalcemia": ("Calcium intoxication", "Side Effect"),
    "methemoglobinemia": ("Methaemoglobinaemia", "Side Effect"),
    "granulocytopenia": ("Agranulocytosis", "Side Effect"),
    "encephalopathy syndrome": ("Encephalopathy", "Side Effect"),
    "5-fluorouracil": ("Fluorouracil", "Compound"),
    "5-aminosalicylic acid": ("Mesalazine", "Compound"),
    "tub b3 protein": ("TUBB3", "Gene"),
    "tubb3 protein": ("TUBB3", "Gene"),
    "muc18": ("MCAM", "Gene"),
    "akt": ("AKT1", "Gene"),
    "hif-1α": ("HIF1A", "Gene"),
    "breast cancer": (None, "Disease"),
    "ovarian cancers": (None, "Disease"),
    "ovarian cancer": (None, "Disease"),
    "prostate cancer": (None, "Disease"),
    "lung cancer": (None, "Disease"),
    "ototoxicity": ("Ototoxicity", "Side Effect"),
    "hepatitis c antibody positive": ("Hepatitis C antibody positive", "Side Effect"),
    "aspirin": ("Acetylsalicylic acid", "Compound"),
    "breast carcinomas": (None, "Disease"),
    "asthma": (None, "Disease"),
    "her-2/neu": ("ERBB2", "Gene"),
    "her2/neu": ("ERBB2", "Gene"),
}


def resolve(store: KGStore, name: str) -> dict | None:
    e = store.resolve_entity(name)
    if e:
        return e
    target, kind = ALIASES.get(name.strip().lower(), (None, None))
    if target:
        return store.resolve_entity(target, preferred_kind=kind)
    if kind:
        # Ambiguous name (multiple nodes share it): disambiguate by node kind.
        rows = store.search_entities(name, limit=15)
        for row in rows:
            if row["kind"].lower() == kind.lower():
                return {"id": row["id"], "name": row["name"], "kind": row["kind"]}
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=["ade", "comagc", "gene"])
    ap.add_argument("--split", default="train_full")
    ap.add_argument("--max-paths", type=int, default=30)
    ap.add_argument("--output", default="data/benchmark/ranking_train.csv")
    ap.add_argument("--graded", action="store_true",
                    help="Use the paper's continuous rel_score (min-max scaled to 0..10) as the "
                         "label instead of the binary `relevant` flag. XGBoost ranking needs "
                         "within-group label variation, and the paper itself reports NDCG.")
    args = ap.parse_args()

    store = KGStore()
    rows_out, unresolved = [], []
    stats = {"pairs": 0, "pair_ok": 0, "path_labeled": 0, "path_unlabeled": 0, "groups": set()}
    labeled_rows: list[dict] = []
    try:
        for ds in args.datasets:
            src = ROOT / "data" / "benchmark" / ds / f"{args.split}.jsonl"
            if not src.exists():
                raise SystemExit(f"Missing {src}")
            for line in src.open(encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                stats["pairs"] += 1
                mps = r.get("metapaths") or []
                if not mps:
                    continue
                # The paper's metapath endpoints are the entities its own entity
                # linking resolved (e.g. benchmark "MDH1" -> Hetionet "MDH1B"),
                # so resolve from the metapath, not from the benchmark name.
                first = mps[0]["stops"].split(" - ")[0].strip()
                last = mps[0]["stops"].split(" - ")[-1].strip()
                e1, e2 = resolve(store, first), resolve(store, last)
                if not e1 or not e2 or e1["id"] == e2["id"]:
                    unresolved.append((ds, r["qid"], first if not e1 else "-", last if not e2 else "-"))
                    continue
                label_by_key = {}
                for m in mps:
                    key = " - ".join(p.strip().lower() for p in m["stops"].split(" - "))
                    # rel_score is the paper's continuous 1±log-prob relevancy; `relevant`
                    # is just its sign (>0). Keep both: binary default, graded optional.
                    label_by_key[key] = (int(m["relevant"]), float(m.get("rel_score", 0.0)))
                paths = store.shortest_paths(e1["id"], e2["id"], settings.max_hops, args.max_paths)
                # allShortestPaths can return node-duplicated entries; dedupe.
                dedup = {}
                for p in paths:
                    dedup[tuple(n["name"].lower() for n in p["path_nodes"])] = p
                paths = list(dedup.values())
                if not paths:
                    continue
                stats["pair_ok"] += 1
                group = f"{ds}_{args.split}_{r['qid']}"
                group_had_row = False
                for p in paths:
                    key = " - ".join(n["name"].strip().lower() for n in p["path_nodes"])
                    hit = label_by_key.get(key)
                    if hit is None:
                        stats["path_unlabeled"] += 1
                        continue
                    lab_binary, lab_score = hit
                    feats = path_features(p, r.get("sentence", ""), e1["name"], e2["name"])
                    item = {
                        "group_id": group, "label": lab_binary, "rel_score": lab_score,
                        "entity_1": e1["name"], "entity_2": e2["name"],
                        "context": r.get("sentence", ""), "path_text": path_text(p),
                    }
                    item.update(dict(zip(FEATURE_NAMES, feats)))
                    rows_out.append(item)
                    stats["path_labeled"] += 1
                    group_had_row = True
                if group_had_row:
                    stats["groups"].add(group)
    finally:
        store.close()

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.graded:
        # Min-max scale rel_score to integer grades 0..10 so every group with
        # distinct scores gets within-group label variation.
        scores = [r["rel_score"] for r in rows_out]
        lo, hi = min(scores), max(scores)
        span = max(1e-9, hi - lo)
        for r in rows_out:
            r["label"] = round((r["rel_score"] - lo) / span * 10)
    cols = ["group_id", "label", "rel_score", "entity_1", "entity_2", "context", "path_text", *FEATURE_NAMES]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_out)
    stats["groups"] = len(stats["groups"])
    print(f"Wrote {len(rows_out)} rows -> {out}")
    print(f"stats: {stats}")
    if unresolved:
        print(f"unresolved pairs ({len(unresolved)}):")
        for u in unresolved[:20]:
            print("  ", u)
    print("Note: labels come from the paper's Subgraph Relevancy Estimation output (field `relevant`).")


if __name__ == "__main__":
    main()
