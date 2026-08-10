from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai-module" / "ai-module"))

try:
    import torch
    import train_model
    from headlight import training
    from headlight.models import HeadlightForecastModel, build_model
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None
    train_model = None
    training = None
    HeadlightForecastModel = None
    build_model = None


class ModelEntrypointTests(unittest.TestCase):
    def setUp(self) -> None:
        if torch is None:
            self.skipTest("PyTorch is not installed")

    def test_training_parser_defaults_to_headlight_forecast_model(self) -> None:
        parser = training.build_parser()
        args = parser.parse_args(
            [
                "--once",
                "--dataset",
                "dataset.csv",
                "--service-profile",
                "profile.json",
                "--adjacency",
                "adjacency.json",
                "--bundle-output",
                "bundle",
            ]
        )
        self.assertEqual(args.controller, "headlight")
        model = build_model(
            {
                "model_type": args.controller,
                "num_nodes": 2,
                "input_size": 14,
                "hidden_width": 16,
                "graph_blocks": 1,
                "variant": "full",
                "residual_bound": 0.3,
            },
            {
                "matrix": [[1.0, 0.0], [0.0, 1.0]],
                "observed_mask": [[1, 0], [0, 1]],
            },
            torch.device("cpu"),
        )
        self.assertIsInstance(model, HeadlightForecastModel)

    def test_train_model_entrypoint_delegates_to_headlight_training_main(self) -> None:
        with mock.patch("headlight.training.main", return_value=0) as training_main:
            self.assertEqual(train_model.main(), 0)
        training_main.assert_called_once_with()

    def test_runtime_reconstructs_model_from_bundle_model_config(self) -> None:
        from headlight import runtime

        fake_bundle = argparse.Namespace(
            model_config={
                "model_type": "headlight",
                "num_nodes": 1,
                "input_size": 14,
                "hidden_width": 16,
                "graph_blocks": 1,
                "variant": "full",
                "residual_bound": 0.3,
                "lookback": 12,
                "sampling_interval_seconds": 15,
            },
            adjacency={"matrix": [[1.0]], "observed_mask": [[1]]},
            profile=argparse.Namespace(nodes=[argparse.Namespace(name="svc")]),
            capacity={},
            bundle_id="bundle",
        )
        with (
            mock.patch("headlight.runtime.ModelBundle.load", return_value=fake_bundle),
            mock.patch("headlight.runtime.load_checkpoint", return_value={}),
            mock.patch("headlight.runtime.build_model") as runtime_build_model,
            mock.patch("headlight.runtime.PrometheusMetricCollector"),
            mock.patch("headlight.runtime.KubectlStateReader"),
            mock.patch("headlight.runtime.ForecastPlacementPolicy"),
            mock.patch("headlight.runtime.PlacementActuator"),
        ):
            model = mock.Mock()
            model.load_state_dict.return_value = None
            model.eval.return_value = None
            runtime_build_model.return_value = model
            runtime.ControllerRuntime.from_bundle(
                "bundle",
                runtime_profile=mock.Mock(),
                prometheus_url="http://prometheus",
                action_handler_url="http://actions",
                controller_name="headlight",
                device=torch.device("cpu"),
            )
        runtime_build_model.assert_called_once()
        self.assertEqual(runtime_build_model.call_args.args[0]["model_type"], "headlight")


if __name__ == "__main__":
    unittest.main()
