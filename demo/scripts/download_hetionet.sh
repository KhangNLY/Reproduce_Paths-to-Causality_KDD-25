#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/raw"
mkdir -p "$OUT"
NODES="$OUT/hetionet-v1.0-nodes.tsv"
EDGES="$OUT/hetionet-v1.0-edges.sif.gz"
NODES_TMP="$NODES.tmp"
EDGES_TMP="$EDGES.tmp"

# Nodes TSV is directly available on raw.githubusercontent.com.
NODES_URL="https://raw.githubusercontent.com/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-nodes.tsv"
# The edge file is tracked by Git LFS. The raw.githubusercontent.com URL can
# return a small LFS pointer text file instead of the actual gzip object.
EDGES_URL="https://media.githubusercontent.com/media/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz"
cleanup() { rm -f "$NODES_TMP" "$EDGES_TMP"; }
trap cleanup EXIT

curl -fL --retry 3 "$NODES_URL" -o "$NODES_TMP"
if ! head -n 1 "$NODES_TMP" | awk -F '\t' '{gsub(/\r$/, "", $3)} $1 == "id" && $2 == "name" && $3 == "kind" {ok=1} END {exit !ok}'; then
  echo "ERROR: Hetionet nodes TSV header check failed: $NODES_TMP" >&2
  exit 1
fi

curl -fL --retry 3 "$EDGES_URL" -o "$EDGES_TMP"
# Guard against Git LFS pointer responses, incomplete downloads and wrong URLs.
if ! gzip -t "$EDGES_TMP"; then
  echo "ERROR: downloaded edges file is not a valid gzip archive." >&2
  echo "The official GitHub path may have returned a Git LFS pointer; inspect $EDGES_TMP or use the official Hetionet release archive." >&2
  exit 1
fi

mv "$NODES_TMP" "$NODES"
mv "$EDGES_TMP" "$EDGES"
trap - EXIT
file "$NODES" "$EDGES"
printf '\nDownloaded and validated Hetionet files. Next: docker compose up -d --build && docker compose exec app python scripts/load_hetionet.py\n'
