"""Regression checks for stage visibility, launch contexts and Docker argv."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sure.runtime.gpu_partition import GPU_CONTEXT, GpuPartitionError, enforce_gpu_context, read_context, restrict_docker_command
from sure.site.gpu_partition import partition_command, partition_environment
from sure.site.loader import SitePolicyError, validate_site_policy

ROOT = Path(__file__).resolve().parents[2]
GPUS = [f"GPU-{i:08x}-0000-0000-0000-000000000000" for i in range(8)]
POLICY = {
    "schema": "sure.site.policy.v1", "site_id": "test", "policy_version": 1,
    "storage": {"approved_models_roots": ["/tmp/models"], "forbidden_output_roots": ["/tmp/models"], "runtime_root": "/tmp/runtime"},
    "datasets": {"allowed_source_roots": {"test": "/tmp/data"}},
    "execution": {"surfaces": ["local"], "gpu_partition": {"onboard_gpu_uuid": GPUS[7]}},
}
LOADED = {"policy": POLICY, "path": "/tmp/test-policy.yaml", "sha256": "test"}


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.policy = patch("sure.site.gpu_partition.load_site_policy", return_value=LOADED)
        self.policy.start()
        self.addCleanup(self.policy.stop)
        self.probe = patch("sure.site.gpu_partition.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "\n".join(f"{i}, {uuid}" for i, uuid in enumerate(GPUS)), ""))
        self.mock_probe = self.probe.start()
        self.addCleanup(self.probe.stop)

    def test_onboard_only_reserved_uuid(self):
        env = partition_environment({}, "onboard")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], GPUS[7])
        self.assertEqual(env["DEVICE"], "cuda:0")

    def test_every_other_stage_excludes_reserved_uuid(self):
        for stage in ["infer", "approve", "trans", "agent_eval", "eval"]:
            with self.subTest(stage=stage):
                env = partition_environment({}, stage)
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"].split(","), GPUS[:7])
                self.assertEqual(read_context(env)["stage"], stage)

    def test_index_reordering_keeps_uuid(self):
        self.mock_probe.return_value.stdout = "\n".join(f"{i}, {uuid}" for i, uuid in enumerate(reversed(GPUS)))
        self.assertEqual(partition_environment({}, "onboard")["CUDA_VISIBLE_DEVICES"], GPUS[7])

    def test_missing_uuid_is_error(self):
        self.mock_probe.return_value.stdout = f"0, {GPUS[0]}"
        with self.assertRaisesRegex(GpuPartitionError, "absent"):
            partition_environment({}, "infer")

    def test_probe_failure_does_not_disable_policy(self):
        self.mock_probe.side_effect = OSError("no driver")
        with self.assertRaisesRegex(GpuPartitionError, "enumerate"):
            partition_environment({}, "onboard")

    def test_cpu_does_not_need_driver(self):
        self.mock_probe.side_effect = AssertionError("CPU must not probe GPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,7"
        env = partition_environment({"CUDA_VISIBLE_DEVICES": "all"}, "onboard", device="cpu")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["NVIDIA_VISIBLE_DEVICES"], "void")

    def test_inherited_cpu_context_stays_cpu(self):
        parent = partition_environment({}, "infer", device="cpu")
        with patch.dict(os.environ, parent):
            child = partition_environment({}, "eval")
            self.assertEqual(child["DEVICE"], "cpu")

    def test_unsupported_device_fails_when_enabled(self):
        for device in ["mps", "cuda:garbage"]:
            with self.subTest(device=device), self.assertRaisesRegex(GpuPartitionError, "unsupported device"):
                partition_environment({}, "infer", device=device)

    def test_parent_visibility_is_respected(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "2,7"
        self.assertEqual(partition_environment({}, "infer")["CUDA_VISIBLE_DEVICES"], GPUS[2])
        self.assertEqual(partition_environment({}, "onboard")["CUDA_VISIBLE_DEVICES"], GPUS[7])

    def test_hidden_or_excluded_onboard_is_error(self):
        for visible in ["", "-1", "0,1"]:
            with self.subTest(visible=visible):
                os.environ["CUDA_VISIBLE_DEVICES"] = visible
                with self.assertRaises(GpuPartitionError):
                    partition_environment({}, "onboard")

    def test_uuid_index_aliases_are_rejected(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = f"0,{GPUS[0]}"
        with self.assertRaisesRegex(GpuPartitionError, "duplicate"):
            partition_environment({}, "infer")

    def test_explicit_host_device_is_mapped_to_local_zero(self):
        env = partition_environment({}, "infer", device="cuda:3")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], GPUS[3])
        self.assertEqual(env["DEVICE"], "cuda:0")

    def test_explicit_reserved_device_is_rejected(self):
        with self.assertRaisesRegex(GpuPartitionError, "outside"):
            partition_environment({}, "infer", device="cuda:7")

    def test_workers_respect_capacity(self):
        env = partition_environment({}, "infer", workers=4)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"].split(","), GPUS[:4])
        self.assertEqual(len(read_context(partition_environment({}, "infer", workers=7))["devices"]), 7)
        with self.assertRaisesRegex(GpuPartitionError, "one of 1, 2, 4, or 7"):
            partition_environment({}, "infer", workers=8)

    def test_workers_use_only_supported_partition_sizes(self):
        for workers in [3, 5, 6, 8]:
            with self.subTest(workers=workers), self.assertRaisesRegex(GpuPartitionError, "one of 1, 2, 4, or 7"):
                partition_environment({}, "infer", workers=workers)

    def test_model_environment_cannot_select_or_forge_devices(self):
        fake = partition_environment({}, "onboard")
        env = partition_environment({**fake, "CUDA_VISIBLE_DEVICES": "7", "NVIDIA_VISIBLE_DEVICES": "all", "DEVICE": "cuda:7"}, "infer")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"].split(","), GPUS[:7])
        self.assertEqual(read_context(env)["stage"], "infer")

    def test_inherited_approve_context_survives_shared_helper(self):
        parent = partition_environment({}, "approve")
        with patch.dict(os.environ, parent):
            child = enforce_gpu_context({"CUDA_VISIBLE_DEVICES": GPUS[7], "DEVICE": "cuda:7"})
        self.assertEqual(read_context(child)["stage"], "approve")
        self.assertNotIn(GPUS[7], child["CUDA_VISIBLE_DEVICES"])

    def test_worker_narrows_context_and_model_cannot_widen_it(self):
        parent = partition_environment({}, "infer", workers=4)
        with patch.dict(os.environ, {**parent, "CUDA_VISIBLE_DEVICES": GPUS[2]}):
            child = enforce_gpu_context({"CUDA_VISIBLE_DEVICES": ",".join(GPUS)})
        self.assertEqual(child["CUDA_VISIBLE_DEVICES"], GPUS[2])
        self.assertEqual(child["DEVICE"], "cuda:0")

    def test_child_cannot_switch_into_onboard(self):
        with patch.dict(os.environ, partition_environment({}, "approve")):
            with self.assertRaisesRegex(GpuPartitionError, "change onboard"):
                partition_environment({}, "onboard")

    def test_unknown_stage_is_error(self):
        with self.assertRaisesRegex(GpuPartitionError, "unknown"):
            partition_environment({}, "unknown")

    def test_disabled_policy_preserves_existing_command(self):
        with patch("sure.site.gpu_partition.load_site_policy", return_value=None):
            command = ["docker", "run", "--gpus", "all", "image"]
            actual, env = partition_command(command, {"DEVICE": "original"}, "infer")
            self.assertEqual(actual, command)
            self.assertEqual(env, {"DEVICE": "original"})

    def test_malformed_context_fails_closed(self):
        with self.assertRaises(GpuPartitionError):
            enforce_gpu_context({}, source={GPU_CONTEXT: "{}"})

    def test_docker_all_becomes_uuid_set(self):
        command, _ = partition_command(["docker", "run", "--rm", "--gpus", "all", "image", "--device", "application-argument"], {}, "infer")
        self.assertNotIn("all", command)
        self.assertEqual(command[command.index("--gpus")+1], '"device=' + ",".join(GPUS[:7]) + '"')
        self.assertEqual(command[-3:], ["image", "--device", "application-argument"])

    def test_docker_explicit_reserved_card_is_rejected(self):
        with self.assertRaises(GpuPartitionError):
            partition_command(["docker", "run", "--gpus", "device=7", "image"], {}, "infer")

    def test_configured_docker_wrapper_is_also_restricted(self):
        for key in ["SURE_DOCKER_BIN", "SURE_ONBOARD_DOCKER_BIN", "SURE_EVAL_DOCKER_BIN"]:
            with self.subTest(key=key), patch.dict(os.environ, {key: "/opt/site/container-cli"}):
                command, _ = partition_command(["/opt/site/container-cli", "run", "--gpus", "all", "image"], {}, "onboard")
                self.assertEqual(command[command.index("--gpus") + 1], f'"device={GPUS[7]}"')

    def test_docker_count_becomes_exact_uuids(self):
        command, _ = partition_command(["docker", "run", "--gpus=2", "image"], {}, "infer")
        self.assertEqual(command[command.index("--gpus")+1], f'"device={GPUS[0]},{GPUS[1]}"')

    def test_docker_declared_env_cannot_override(self):
        command, _ = partition_command(["docker", "run", "-e", "CUDA_VISIBLE_DEVICES=7", "--env=NVIDIA_VISIBLE_DEVICES=all", "--gpus", "all", "image"], {}, "onboard")
        self.assertNotIn("CUDA_VISIBLE_DEVICES=7", command)
        self.assertIn(f"CUDA_VISIBLE_DEVICES={GPUS[7]}", command)
        self.assertIn(f"NVIDIA_VISIBLE_DEVICES={GPUS[7]}", command)

    def test_docker_cpu_hides_image_default_devices(self):
        command, _ = partition_command(["docker", "run", "image"], {}, "infer", device="cpu")
        self.assertNotIn("--gpus", command)
        self.assertIn("NVIDIA_VISIBLE_DEVICES=void", command)
        self.assertIn("CUDA_VISIBLE_DEVICES=", command)

    def test_unsupported_or_ambiguous_docker_options_fail(self):
        for options in [["--privileged"], ["--device", "/dev/nvidia0"], ["--env-file", "env"], ["-v", "/dev:/dev"], ["--gpus", "all", "--gpus", "all"], ["--not-a-real-option"]]:
            with self.subTest(options=options), self.assertRaises(GpuPartitionError):
                partition_command(["docker", "run", *options, "image"], {}, "infer")

    def test_shell_commands_are_rejected_when_enabled(self):
        with self.assertRaisesRegex(GpuPartitionError, "argv"):
            partition_command("docker run --gpus all image", {}, "onboard")

    def test_sequential_stages_do_not_mutate_parent_environment(self):
        before = dict(os.environ)
        partition_environment({}, "onboard")
        env = partition_environment({}, "infer")
        self.assertEqual(dict(os.environ), before)
        self.assertNotIn(GPUS[7], env["CUDA_VISIBLE_DEVICES"])


class PolicyTests(unittest.TestCase):
    def test_valid_config_and_absent_config(self):
        self.assertEqual(validate_site_policy(POLICY)["execution"]["gpu_partition"], {"onboard_gpu_uuid": GPUS[7]})
        value = copy.deepcopy(POLICY)
        del value["execution"]["gpu_partition"]
        self.assertNotIn("gpu_partition", validate_site_policy(value)["execution"])

    def test_invalid_configuration(self):
        for field in [None, {}, [], {"onboard_gpu_uuid": "7"}, {"onboard_gpu_uuid": [GPUS[7]]}, {"onboard_gpu_uuid": GPUS[7], "enabled": True}]:
            with self.subTest(field=field):
                value = copy.deepcopy(POLICY)
                value["execution"]["gpu_partition"] = field
                with self.assertRaises(SitePolicyError):
                    validate_site_policy(value)


class SubprocessTests(unittest.TestCase):
    def test_model_helper_is_stdlib_only_and_enforces_parent(self):
        context = {"stage": "approve", "onboard_uuid": GPUS[7], "devices": GPUS[:2]}
        code = "import json; from sure.runtime.gpu_partition import enforce_gpu_context; print(json.dumps(enforce_gpu_context({'CUDA_VISIBLE_DEVICES':'7'})))"
        env = {**os.environ, GPU_CONTEXT: json.dumps(context), "CUDA_VISIBLE_DEVICES": GPUS[1]}
        # -S proves that no site-packages (including PyYAML) is required.
        completed = subprocess.run([sys.executable, "-S", "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
        actual = json.loads(completed.stdout)
        self.assertEqual(actual["CUDA_VISIBLE_DEVICES"], GPUS[1])

    def test_onboard_cli_runs_real_child_with_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            policy = temporary / "site.json"
            policy.write_text(json.dumps(POLICY))
            probe = temporary / "nvidia-smi"
            probe.write_text("#!/bin/sh\nprintf '%s\\n' '0, " + GPUS[0] + "' '7, " + GPUS[7] + "'\n")
            probe.chmod(0o755)
            env = {key: value for key, value in os.environ.items() if key not in {GPU_CONTEXT, "CUDA_VISIBLE_DEVICES"}}
            env.update(SURE_SITE_POLICY=str(policy), PATH=str(temporary) + os.pathsep + os.environ.get("PATH", ""))
            child = "import os,json; print(json.dumps({k:os.environ[k] for k in ['CUDA_VISIBLE_DEVICES','DEVICE']}))"
            result = subprocess.run([sys.executable, str(ROOT / "sure/skills/sure_onboard/scripts/run_gpu_command.py"), "--", sys.executable, "-c", child], env=env, cwd=temporary, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout), {"CUDA_VISIBLE_DEVICES": GPUS[7], "DEVICE": "cuda:0"})


if __name__ == "__main__":
    unittest.main()
