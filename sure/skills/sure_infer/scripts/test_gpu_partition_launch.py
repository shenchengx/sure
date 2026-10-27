"""Enabled-policy checks at the real launch builders, using existing fixtures."""
import copy
import os
import subprocess
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import container_execution
import python_execution
import resolve_eval_input
import test_deployment_binding as container_fixtures
import test_python_execution as python_fixtures
from sure.site.test_gpu_partition import GPUS, LOADED


class LaunchTests(unittest.TestCase):
    def policy(self, reserved=1):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(os.environ, {}, clear=True))
        loaded = copy.deepcopy(LOADED)
        loaded["policy"]["execution"]["gpu_partition"]["onboard_gpu_uuids"] = GPUS[-reserved:]
        stack.enter_context(patch("sure.site.gpu_partition.load_site_policy", return_value=loaded))
        stack.enter_context(patch("sure.site.gpu_partition.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "\n".join(f"{i}, {gpu}" for i, gpu in enumerate(GPUS)), "")))

    def test_real_container_builder_single_and_multi_worker(self):
        fixture = container_fixtures.DeploymentBindingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        binding = container_fixtures.load_deployment_binding(fixture.model, "demo")
        self.policy()
        for workers in [1, 2, 7]:
            with self.subTest(workers=workers):
                command, _ = container_execution.build_local_container_command(
                    surface={"env": {"CUDA_VISIBLE_DEVICES": "7", "NVIDIA_VISIBLE_DEVICES": "all"}},
                    eval_input={"model": {"deployment_binding": binding}, "runtime": {
                        "num_workers": workers, "run_dir": str(fixture.output), "harness_runtime": fixture._runtime_binding(),
                        "dataset_projection": {"host_root": str(fixture.projection)}}, "datasets": []},
                    control_run_dir=fixture.control, entrypoint=fixture.entrypoint, repo_root=fixture.repo,
                    device_request="cuda")
                selected = GPUS[:workers] if workers > 1 else GPUS[:7]
                self.assertEqual(command[command.index("--gpus") + 1], '"device=' + ",".join(selected) + '"')
                self.assertIn("CUDA_VISIBLE_DEVICES=" + ",".join(selected), command)
                self.assertNotIn("CUDA_VISIBLE_DEVICES=7", command)

    def test_real_python_builder_restricts_final_environment(self):
        fixture = python_fixtures.PythonExecutionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        harness = {"runtime_id": "test", "python_executable": str(fixture.harness_python), "lock_sha256": "b" * 64,
                   "manifest_path": str(fixture.root / "manifest.json"), "runtime_root": str(fixture.root / "harness")}
        self.policy()
        with patch.object(python_execution, "harness_runtime_from_eval_input", return_value=harness):
            _, env, _ = python_execution.build_local_python_command(
                surface={"env": {"CUDA_VISIBLE_DEVICES": "7"}}, eval_input=fixture.eval_input,
                entrypoint=fixture.entrypoint, repo_root=Path(__file__).resolve().parents[4],
                extra_env={"SURE_EVAL_DEVICE_REQUEST": "cuda:3"})
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], GPUS[3])
        self.assertEqual(env["DEVICE"], "cuda:0")

    def test_container_uses_selected_host_policy_outside_checkout(self):
        fixture = container_fixtures.DeploymentBindingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        binding = container_fixtures.load_deployment_binding(fixture.model, "demo")
        policy_path = fixture.root / "selected-site.yaml"
        policy_path.write_text("schema: sure.site.policy.v1\n")
        with patch.object(container_execution, "load_site_policy", return_value={"path": str(policy_path)}):
            command, _ = container_execution.build_local_container_command(
                surface={"env": {"SURE_SITE_POLICY": "/model/forged.yaml"}},
                eval_input={"model": {"deployment_binding": binding}, "runtime": {
                    "run_dir": str(fixture.output), "harness_runtime": fixture._runtime_binding()}, "datasets": []},
                control_run_dir=fixture.control, entrypoint=fixture.entrypoint, repo_root=fixture.repo,
                device_request="cpu")
        target = container_execution.container_path(policy_path.resolve())
        self.assertIn(f"type=bind,src={policy_path.resolve()},dst={target},readonly", command)
        self.assertIn(f"SURE_SITE_POLICY={target}", command)
        self.assertNotIn("SURE_SITE_POLICY=/model/forged.yaml", command)

    def test_worker_auto_capacity_excludes_onboard(self):
        self.policy()
        self.assertEqual(resolve_eval_input._local_gpu_count(), 7)

    def test_two_reserved_gpus_reduce_automatic_infer_capacity(self):
        self.policy(reserved=2)
        capacity = resolve_eval_input._local_gpu_count()
        self.assertEqual(capacity, 6)
        for selected, expected in [(1000, 1), (1001, 2), (2001, 3), (5001, 6), (7060, 6)]:
            self.assertEqual(resolve_eval_input._auto_workers(selected, capacity), expected)

    def test_onboard_validation_environment_overrides_artifact(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sure_onboard/scripts"))
        import run_validate
        self.policy()
        env = run_validate.env_for({"env": {"CUDA_VISIBLE_DEVICES": "0,1", "DEVICE": "cuda:7"}}, "cuda")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], GPUS[7])
        self.assertEqual(env["DEVICE"], "cuda:0")


if __name__ == "__main__":
    unittest.main()
