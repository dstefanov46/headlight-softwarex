from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai-module" / "ai-module"))

from headlight.synthetic_v2 import (  # noqa: E402
    ENDPOINTS,
    NODE_ORDER,
    ROUTING_EDGES,
    SyntheticConfig,
    build_adjacency,
    expand_sparse_edges,
    generate_dataset,
    load_manifest,
    simulate_run,
)


class SyntheticV2Tests(unittest.TestCase):
    def config(self) -> SyntheticConfig:
        return SyntheticConfig(
            fitting_runs=2,
            calibration_runs=1,
            holdout_runs=1,
            steps_per_run=72,
            candidates_per_step=6,
            base_seed=20_000,
        )

    def test_simulation_is_deterministic_and_has_counterfactual_boundaries(self) -> None:
        config = self.config()
        first = simulate_run(
            seed=20_001,
            split="holdout",
            scenario="mixed",
            workload="flash-crowd",
            config=config,
        )
        second = simulate_run(
            seed=20_001,
            split="holdout",
            scenario="mixed",
            workload="flash-crowd",
            config=config,
        )
        for name in (
            "node_features",
            "edge_features",
            "candidate_replicas",
            "latency_target",
            "sla_target",
            "reliability_target",
            "boundary_mask",
        ):
            np.testing.assert_array_equal(first[name], second[name])
        self.assertEqual(
            first["candidate_replicas"].shape,
            (config.steps_per_run, config.candidates_per_step, len(NODE_ORDER)),
        )
        self.assertEqual(
            first["latency_target"].shape[-3:],
            (config.candidates_per_step, len(ENDPOINTS), len(config.horizons_seconds)),
        )
        valid = first["sample_indices"]
        self.assertGreater(int(first["sla_target"][valid].sum()), 0)
        self.assertGreater(int(first["boundary_mask"][valid].sum()), 0)
        self.assertTrue(np.all(first["reliability_target"][valid] >= 0.0))
        self.assertTrue(np.all(first["reliability_target"][valid] <= 1.0))
        self.assertTrue(np.all(np.isfinite(first["node_features"])))
        self.assertTrue(np.all(np.isfinite(first["edge_features"])))

    def test_sparse_edges_expand_to_source_destination_contract(self) -> None:
        config = self.config()
        run = simulate_run(
            seed=20_002,
            split="fitting",
            scenario="clean",
            workload="gradual-ramp",
            config=config,
        )
        sparse = run["edge_features"][None, :3]
        dense = expand_sparse_edges(sparse, run["edge_source"], run["edge_destination"])
        self.assertEqual(
            dense.shape,
            (1, 3, len(NODE_ORDER), len(NODE_ORDER), sparse.shape[-1]),
        )
        source = int(run["edge_source"][0])
        destination = int(run["edge_destination"][0])
        np.testing.assert_array_equal(dense[0, :, source, destination], sparse[0, :, 0])

    def test_adjacency_has_meaningful_graph_coverage(self) -> None:
        graph = build_adjacency()
        matrix = np.asarray(graph["matrix"])
        observed = np.asarray(graph["observed_mask"])
        np.testing.assert_allclose(matrix.sum(axis=1), np.ones(len(NODE_ORDER)), atol=1e-6)
        self.assertGreaterEqual(int(observed.sum() - len(NODE_ORDER)), len(ROUTING_EDGES))

    def test_manifest_keeps_runs_in_disjoint_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paper = root / "aggregate.csv"
            paper.write_text(
                "controller,workload,metric,independent_unit,n_seed_runs,mean,median,ci95_low,ci95_high,claim\n"
                "headlight,diurnal,successful_throughput_rps,seed_run,5,4.3,4.3,4,5,x\n"
                "headlight,diurnal,p95_latency_ms,seed_run,5,300,300,250,350,x\n"
                "headlight,diurnal,cpu_seconds,seed_run,5,5000,5000,4000,6000,x\n"
                "headlight,diurnal,pod_seconds,seed_run,5,200000,200000,180000,220000,x\n",
                encoding="utf-8",
            )
            output = root / "dataset"
            generated = generate_dataset(output, config=self.config(), paper_summary=paper)
            loaded = load_manifest(output / "manifest.json")
            self.assertEqual(generated["dataset_hash"], loaded["dataset_hash"])
            seeds: dict[str, set[int]] = {}
            for split in ("fitting", "calibration", "holdout"):
                seeds[split] = {int(row["seed"]) for row in loaded["runs"] if row["split"] == split}
            self.assertFalse(seeds["fitting"] & seeds["calibration"])
            self.assertFalse(seeds["fitting"] & seeds["holdout"])
            self.assertFalse(seeds["calibration"] & seeds["holdout"])
            self.assertEqual(loaded["run_count"], 4)
            json.loads((output / "model-config.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
