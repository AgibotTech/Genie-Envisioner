import ast
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_UTILS_PATH = REPO_ROOT / "utils" / "model_utils.py"


def load_function(function_name):
    source = MODEL_UTILS_PATH.read_text()
    module_ast = ast.parse(source)
    for node in module_ast.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            namespace = {}
            function_module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(function_module)
            exec(compile(function_module, str(MODEL_UTILS_PATH), "exec"), namespace)
            return namespace[function_name]
    return None


class RecorderPipeline:
    def __init__(self, *args):
        self.args = args


class GeSimRegressionTests(unittest.TestCase):
    def test_compute_total_latent_frames_uses_encoded_future_axis(self):
        compute_total_latent_frames = load_function("compute_total_latent_frames")
        self.assertIsNotNone(compute_total_latent_frames)

        class DummyTensor:
            shape = (2, 16, 3, 5, 7)
            ndim = len(shape)

        self.assertEqual(compute_total_latent_frames(4, DummyTensor()), 7)

    def test_build_gesim_pipeline_matches_public_signature(self):
        build_gesim_pipeline = load_function("build_gesim_pipeline")
        self.assertIsNotNone(build_gesim_pipeline)

        pipe = build_gesim_pipeline(
            RecorderPipeline,
            text_encoder="text",
            tokenizer="tokenizer",
            transformer="transformer",
            vae="vae",
            scheduler="scheduler",
        )

        self.assertEqual(pipe.args, ("text", "tokenizer", "transformer", "vae", "scheduler"))

    def test_ge_trainer_uses_public_pipeline_builder_and_latent_frame_helper(self):
        trainer_source = (REPO_ROOT / "runner" / "ge_trainer.py").read_text()

        self.assertIn("compute_total_latent_frames", trainer_source)
        self.assertIn("build_gesim_pipeline", trainer_source)

    def test_ge_inferencer_uses_public_pipeline_builder(self):
        inferencer_source = (REPO_ROOT / "runner" / "ge_inferencer.py").read_text()

        self.assertIn("build_gesim_pipeline", inferencer_source)


if __name__ == "__main__":
    unittest.main()
