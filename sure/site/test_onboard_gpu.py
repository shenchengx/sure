"""Exercise real process locks, waiting and cancellation without model weights."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from sure.runtime.gpu_partition import GPU_CONTEXT
from sure.runtime.onboard_gpu import wrap_onboard_command

GPUS = ["GPU-ffffffff-ffff-ffff-ffff-000000000001", "GPU-ffffffff-ffff-ffff-ffff-000000000002"]


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = {**os.environ, GPU_CONTEXT: json.dumps({"stage": "onboard", "devices": GPUS, "onboard_uuids": GPUS})}
        self.env.pop("CUDA_VISIBLE_DEVICES", None)

    def start(self, name, *, seconds=1, timeout=None, devices=None):
        env = dict(self.env)
        if devices is not None:
            env[GPU_CONTEXT] = json.dumps({"stage": "onboard", "devices": devices, "onboard_uuids": GPUS})
        command = [sys.executable, "-c", "import time; time.sleep(" + str(seconds) + ")"]
        command = wrap_onboard_command(command, env, timeout=timeout, log_path=self.root / f"{name}.jsonl")
        proc = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(self.cleanup, proc)
        return proc

    def cleanup(self, proc):
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)

    def events(self, name):
        path = self.root / f"{name}.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def wait_for(self, name, event):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            matches = [x for x in self.events(name) if x["event"] == event]
            if matches:
                return matches[0]
            time.sleep(0.02)
        self.fail(f"{name}: no {event}: {self.events(name)}")

    def test_two_cards_parallel_third_waits_and_reuses_released_card(self):
        a = self.start("a", seconds=1.2)
        aa = self.wait_for("a", "acquired")
        b = self.start("b", seconds=1.2)
        bb = self.wait_for("b", "acquired")
        self.assertNotEqual(aa["uuid"], bb["uuid"])
        c = self.start("c", seconds=0.1)
        self.wait_for("c", "waiting")
        self.assertFalse(any(x["event"] == "acquired" for x in self.events("c")))
        for proc in [a, b, c]:
            self.assertEqual(proc.wait(timeout=10), 0)
        cc = self.wait_for("c", "acquired")
        releases = [x for name in ["a", "b"] for x in self.events(name) if x["event"] == "released" and x["uuid"] == cc["uuid"]]
        self.assertLessEqual(releases[0]["time"], cc["time"])

    def test_wait_does_not_consume_execution_timeout(self):
        a = self.start("a", seconds=0.8, devices=GPUS[:1])
        self.wait_for("a", "acquired")
        b = self.start("b", seconds=0.1, timeout=0.4, devices=GPUS[:1])
        self.assertEqual(a.wait(timeout=10), 0)
        self.assertEqual(b.wait(timeout=10), 0)
        self.assertGreater(self.wait_for("b", "acquired")["waited_seconds"], 0.4)

    def test_timeout_releases_card(self):
        a = self.start("a", seconds=60, timeout=0.2, devices=GPUS[:1])
        self.assertEqual(a.wait(timeout=10), 124)
        self.wait_for("a", "released")
        b = self.start("b", seconds=0.01, devices=GPUS[:1])
        self.assertEqual(b.wait(timeout=10), 0)

    def test_cancel_waiter_does_not_release_owners_card(self):
        a = self.start("a", seconds=1, devices=GPUS[:1])
        self.wait_for("a", "acquired")
        b = self.start("b", devices=GPUS[:1])
        self.wait_for("b", "waiting")
        b.terminate()
        self.assertEqual(b.wait(timeout=10), 128 + signal.SIGTERM)
        self.assertFalse(any(x["event"] == "acquired" for x in self.events("b")))
        self.assertIsNone(a.poll())
        self.assertEqual(a.wait(timeout=10), 0)

    def test_cancel_running_owner_releases(self):
        a = self.start("a", seconds=60, devices=GPUS[:1])
        self.wait_for("a", "acquired")
        a.terminate()
        self.assertEqual(a.wait(timeout=10), 128 + signal.SIGTERM)
        b = self.start("b", seconds=0.01, devices=GPUS[:1])
        self.assertEqual(b.wait(timeout=10), 0)

    def test_abandoned_receipt_is_not_silently_reallocated(self):
        root = Path("/tmp") / f"sure-onboard-gpu-{os.getuid()}"
        root.mkdir(mode=0o700, exist_ok=True)
        owner = root / f"{GPUS[0]}.owner.json"
        owner.write_text('{"supervisor_pid": -1}')
        self.addCleanup(owner.unlink, missing_ok=True)
        proc = self.start("abandoned", seconds=0.01, devices=GPUS[:1])
        self.assertEqual(proc.wait(timeout=10), 2)
        self.assertFalse(any(x["event"] == "acquired" for x in self.events("abandoned")))

    def test_killed_parent_stops_workload_and_releases_card(self):
        log = self.root / "orphan.jsonl"
        runner = self.root / "parent.py"
        runner.write_text(
            "import os, subprocess, sys\n"
            "from sure.runtime.onboard_gpu import wrap_onboard_command\n"
            "command = [sys.executable, '-c', 'import time; time.sleep(60)']\n"
            "subprocess.run(wrap_onboard_command(command, os.environ, log_path=sys.argv[1]), capture_output=True)\n"
        )
        env = {**self.env, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
        parent = subprocess.Popen([sys.executable, str(runner), str(log)], env=env,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(self.cleanup, parent)
        self.wait_for("orphan", "acquired")
        parent.kill()
        parent.wait(timeout=10)
        self.wait_for("orphan", "released")
        next_task = self.start("next", seconds=0.01)
        self.assertEqual(next_task.wait(timeout=10), 0)

    def test_docker_auto_removed_container_with_wrapper_zero_exit(self):
        docker = self.root / "docker"
        docker.write_text(
            "#!" + sys.executable + "\n"
            "import pathlib, sys\n"
            "if sys.argv[1] == 'run':\n"
            "    pathlib.Path(sys.argv[sys.argv.index('--cidfile')+1]).write_text('a'*64)\n"
            "elif sys.argv[1] == 'inspect':\n"
            "    print('[]'); print('Error: No such object: ' + sys.argv[2], file=sys.stderr)\n"
            "else: raise RuntimeError('already removed container must not be removed twice')\n"
        )
        docker.chmod(0o755)
        command = [str(docker), "run", "--gpus", '\"device=' + ",".join(GPUS) + '\"', "image"]
        proc = subprocess.run(wrap_onboard_command(command, self.env, log_path=self.root / "docker.jsonl"),
                              env=self.env, capture_output=True, text=True, timeout=10)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.events("docker")[-1]["event"], "released")

    def test_cpu_other_stages_and_nested_leases_are_not_wrapped(self):
        command = [sys.executable, "-c", "pass"]
        for context in [None, {"stage": "onboard", "devices": [], "onboard_uuids": GPUS},
                        {"stage": "infer", "devices": ["GPU-ffffffff-ffff-ffff-ffff-000000000003"], "onboard_uuids": GPUS},
                        {"stage": "onboard", "devices": GPUS[:1], "onboard_uuids": GPUS, "onboard_lease": {"uuid": GPUS[0]}}]:
            env = {GPU_CONTEXT: json.dumps(context)} if context else {}
            self.assertIs(wrap_onboard_command(command, env), command)


if __name__ == "__main__":
    unittest.main()
