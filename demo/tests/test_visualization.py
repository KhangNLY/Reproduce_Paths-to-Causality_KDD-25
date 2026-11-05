import sys
import types
import unittest

from src.features import path_text
from src.visualization import build_graph_payload, render_graph_html


class PathTextTests(unittest.TestCase):
    def test_path_text_preserves_actual_edge_direction_when_traversed_backwards(self):
        path = {
            "path_nodes": [
                {"id": "A", "name": "Gene A", "kind": "Gene"},
                {"id": "B", "name": "Disease B", "kind": "Disease"},
            ],
            "path_edges": [
                {"source": "B", "target": "A", "type": "DaG"},
            ],
            "hops": 1,
        }
        text = path_text(path)
        self.assertIn("Gene A [Gene] <-[associates (DaG)]- Disease B [Disease]", text)

    def test_path_text_handles_native_direction(self):
        path = {
            "path_nodes": [
                {"id": "A", "name": "Gene A", "kind": "Gene"},
                {"id": "B", "name": "Gene B", "kind": "Gene"},
            ],
            "path_edges": [{"source": "A", "target": "B", "type": "GiG"}],
        }
        self.assertIn("Gene A [Gene] -[interacts (GiG)]-> Gene B [Gene]", path_text(path))


class GraphPayloadTests(unittest.TestCase):
    def setUp(self):
        self.e1 = {"id": "A", "name": "Gene A", "kind": "Gene"}
        self.e2 = {"id": "B", "name": "Disease B", "kind": "Disease"}
        self.path = {
            "path_nodes": [self.e2, {"id": "X", "name": "Intermediate X", "kind": "Gene"}, self.e1],
            # Path traverses B -> X -> A, but the second stored edge points A -> X.
            "path_edges": [
                {"source": "B", "target": "X", "type": "DaG"},
                {"source": "A", "target": "X", "type": "GiG"},
            ],
        }

    def test_preserves_stored_edge_direction_and_deduplicates(self):
        payload = build_graph_payload([self.path, self.path], self.e1, self.e2, "causal")
        self.assertEqual({n["id"] for n in payload["nodes"]}, {"A", "B", "X"})
        self.assertEqual(len(payload["edges"]), 2)
        by_relation = {e["type"]: e for e in payload["edges"]}
        self.assertEqual((by_relation["DaG"]["source"], by_relation["DaG"]["target"]), ("B", "X"))
        self.assertEqual((by_relation["GiG"]["source"], by_relation["GiG"]["target"]), ("A", "X"))
        self.assertEqual(payload["prediction_overlay"]["source"], "A")
        self.assertEqual(payload["prediction_overlay"]["target"], "B")

    def test_non_causal_prediction_is_not_drawn_as_an_edge(self):
        payload = build_graph_payload([], self.e1, self.e2, "non-causal")
        self.assertEqual(len(payload["nodes"]), 2)
        self.assertEqual(payload["edges"], [])
        self.assertIsNone(payload["prediction_overlay"])

    def test_empty_graph_still_contains_endpoints_without_edges(self):
        payload = build_graph_payload([], self.e1, self.e2, "non-causal")
        self.assertEqual([n["name"] for n in payload["nodes"]], ["Gene A", "Disease B"])
        self.assertEqual(payload["edges"], [])
        self.assertIsNone(payload["prediction_overlay"])

    def test_uncertain_prediction_is_not_drawn_as_an_edge(self):
        payload = build_graph_payload([self.path], self.e1, self.e2, "uncertain")
        self.assertIsNone(payload["prediction_overlay"])

    def test_pyvis_renderer_uses_native_kg_direction_and_undirected_overlay(self):
        calls = {"nodes": [], "edges": [], "options": None}

        class FakeNetwork:
            def __init__(self, **kwargs):
                calls["init"] = kwargs
            def add_node(self, node_id, **kwargs):
                calls["nodes"].append((node_id, kwargs))
            def add_edge(self, source, target, **kwargs):
                calls["edges"].append((source, target, kwargs))
            def set_options(self, options):
                calls["options"] = options
            def generate_html(self):
                return "<html><body>mock-graph</body></html>"

        fake_pyvis = types.ModuleType("pyvis")
        fake_network = types.ModuleType("pyvis.network")
        fake_network.Network = FakeNetwork
        fake_pyvis.network = fake_network
        old_pyvis = sys.modules.get("pyvis")
        old_network = sys.modules.get("pyvis.network")
        sys.modules["pyvis"] = fake_pyvis
        sys.modules["pyvis.network"] = fake_network
        try:
            payload = build_graph_payload([self.path], self.e1, self.e2, "causal")
            html = render_graph_html(payload)
        finally:
            if old_pyvis is None:
                sys.modules.pop("pyvis", None)
            else:
                sys.modules["pyvis"] = old_pyvis
            if old_network is None:
                sys.modules.pop("pyvis.network", None)
            else:
                sys.modules["pyvis.network"] = old_network

        self.assertIn("mock-graph", html)
        kg_edges = [edge for edge in calls["edges"] if not edge[2].get("dashes")]
        overlays = [edge for edge in calls["edges"] if edge[2].get("dashes")]
        self.assertIn(("B", "X"), [(a, b) for a, b, _ in kg_edges])
        self.assertIn(("A", "X"), [(a, b) for a, b, _ in kg_edges])
        self.assertTrue(any("liên quan với" in edge[2].get("label", "") for edge in kg_edges))
        self.assertTrue(any("Hetionet (VI): Disease liên quan với Gene" in edge[2].get("title", "") for edge in kg_edges))
        self.assertEqual(len(overlays), 1)
        self.assertEqual(overlays[0][0:2], ("A", "B"))
        self.assertEqual(overlays[0][2].get("arrows"), "")
        import json
        json.loads(calls["options"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
