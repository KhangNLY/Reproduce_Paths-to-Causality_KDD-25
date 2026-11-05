from __future__ import annotations
import re
from collections import Counter
from src.relation_labels import relation_verb, relation_verb_vi

FEATURE_NAMES = ["hops", "node_count", "unique_relation_count", "context_overlap", "endpoint_overlap", "short_path_score"]
STOPWORDS = {"the", "and", "or", "to", "of", "in", "a", "an", "is", "are", "was", "were", "for", "with", "from", "by", "on", "that", "this", "as", "be", "can", "may", "not", "has", "have", "it", "its"}
NODE_KIND_LABELS_VI = {
    "Gene": "gen",
    "Disease": "bệnh",
    "Anatomy": "cơ quan/mô",
    "Compound": "hợp chất",
    "Side Effect": "tác dụng phụ",
    "Biological Process": "quá trình sinh học",
    "Cellular Component": "thành phần tế bào",
    "Molecular Function": "chức năng phân tử",
    "Pathway": "đường dẫn sinh học",
}

def tokens(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z0-9]+", text or "") if len(t) > 2 and t.lower() not in STOPWORDS}

def path_text(path: dict) -> str:
    """Describe a retrieved path while preserving each KG edge's stored direction.

    Neo4j path retrieval is undirected, so a path can traverse an edge backwards.
    The edge's source/target fields contain the actual Hetionet direction.
    """
    nodes = path.get("path_nodes", [])
    edges = path.get("path_edges", [])
    if not nodes:
        return ""

    def describe(node: dict) -> str:
        return f"{node.get('name', '')} [{node.get('kind', '')}]"

    result = describe(nodes[0])
    for i, edge in enumerate(edges):
        if i + 1 >= len(nodes):
            break
        left, right = nodes[i], nodes[i + 1]
        left_id, right_id = str(left.get("id", "")), str(right.get("id", ""))
        source, target = str(edge.get("source", "")), str(edge.get("target", ""))
        relation = str(edge.get("type", "related to"))
        verb = relation_verb(relation)
        relation_label = f"{verb} ({relation})"
        if source == left_id and target == right_id:
            result += f" -[{relation_label}]-> {describe(right)}"
        elif source == right_id and target == left_id:
            result += f" <-[{relation_label}]- {describe(right)}"
        else:
            # Preserve the traversal but signal that orientation metadata was missing.
            result += f" -[{relation_label}?]- {describe(right)}"
    return result


def path_text_vi(path: dict) -> str:
    """Explain node kinds and stored relation direction in Vietnamese."""
    nodes = path.get("path_nodes", [])
    edges = path.get("path_edges", [])
    if not nodes:
        return ""

    def describe(node: dict) -> str:
        kind = node.get("kind", "Entity")
        kind_vi = NODE_KIND_LABELS_VI.get(kind, kind)
        return f"{node.get('name', '')} ({kind_vi}; {kind})"

    result = describe(nodes[0])
    for i, edge in enumerate(edges):
        if i + 1 >= len(nodes):
            break
        left, right = nodes[i], nodes[i + 1]
        left_id, right_id = str(left.get("id", "")), str(right.get("id", ""))
        source, target = str(edge.get("source", "")), str(edge.get("target", ""))
        relation = str(edge.get("type", "related to"))
        verb = relation_verb_vi(relation)
        if source == left_id and target == right_id:
            result += f" -[{verb} ({relation})]-> {describe(right)}"
        elif source == right_id and target == left_id:
            result += f" <-[{verb} ({relation})]- {describe(right)}"
        else:
            result += f" -[{verb} ({relation})?]- {describe(right)}"
    return result

def path_features(path: dict, context: str, entity_1: str, entity_2: str) -> list[float]:
    ptext = path_text(path)
    p_tokens = tokens(ptext)
    c_tokens = tokens(context)
    e_tokens = tokens(entity_1) | tokens(entity_2)
    overlap = len(p_tokens & c_tokens) / max(1, len(p_tokens | c_tokens))
    endpoint_overlap = len(p_tokens & e_tokens) / max(1, len(e_tokens))
    hops = int(path.get("hops", max(0, len(path.get("path_nodes", [])) - 1)))
    relations = {e.get("type", "") for e in path.get("path_edges", [])}
    return [float(hops), float(len(path.get("path_nodes", []))), float(len(relations)), float(overlap), float(endpoint_overlap), 1.0 / max(1, hops)]
