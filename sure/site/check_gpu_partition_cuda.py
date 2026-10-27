"""Opt-in real Docker/CUDA smoke; never downloads an image or loads a model.

Run with Harness Python: --image <cached torch image> --onboard-uuid <full UUID>.
Uses a temporary site policy, leaving the site's effective configuration intact.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sure.site.gpu_partition import partition_command
from sure.runtime.gpu_partition import GPU_CONTEXT, read_context

PROBE = """
import json, os, time
from sure.runtime.gpu_partition import enforce_gpu_context
# Model-declared env cannot undo the parent restriction.
os.environ.update(enforce_gpu_context({'CUDA_VISIBLE_DEVICES': 'all'}))
import torch
count = torch.cuda.device_count()
uuids = ['GPU-' + str(torch.cuda.get_device_properties(i).uuid).removeprefix('GPU-') for i in range(count)]
value = float(torch.ones((64, 64), device='cuda:0' if count else 'cpu').sum())
time.sleep(2)
print(json.dumps({'count': count, 'uuids': uuids, 'value': value,
                 'visible': os.environ['CUDA_VISIBLE_DEVICES'], 'device': os.environ['DEVICE']}))
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--onboard-uuid", required=True)
    args = parser.parse_args()
    if GPU_CONTEXT in os.environ or "CUDA_VISIBLE_DEVICES" in os.environ:
        parser.error("run in an unrestricted host shell (no inherited GPU context/CUDA_VISIBLE_DEVICES)")
    subprocess.run(["docker", "image", "inspect", args.image], stdout=subprocess.DEVNULL, check=True)
    policy = {
        "schema": "sure.site.policy.v1", "site_id": "gpu-smoke", "policy_version": 1,
        "storage": {"approved_models_roots": ["/tmp/models"], "forbidden_output_roots": ["/tmp/models"], "runtime_root": "/tmp/runtime"},
        "datasets": {"allowed_source_roots": {"test": "/tmp/data"}},
        "execution": {"surfaces": ["local"], "gpu_partition": {"onboard_gpu_uuids": [args.onboard_uuid]}},
    }

    def run(case: tuple[str, str, int]) -> dict:
        stage, device, workers = case
        command = ["docker", "run", "--rm", "--pull", "never", "--network", "none",
                   "--mount", f"type=bind,src={ROOT},dst=/repo,readonly", "--workdir", "/repo",
                   "--entrypoint", "python", args.image, "-c", PROBE]
        command, env = partition_command(command, os.environ, stage, device=device, workers=workers)
        expected = read_context(env)["devices"]
        completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120, check=True)
        actual = json.loads(completed.stdout.strip().splitlines()[-1])
        if actual["uuids"] != expected or actual["count"] != len(expected) or actual["value"] != 4096.0:
            raise AssertionError({"case": case, "expected": expected, "actual": actual})
        return {"stage": stage, "request": device, "workers": workers, **actual}

    with tempfile.TemporaryDirectory(prefix="sure-gpu-cuda-") as directory:
        path = Path(directory) / "site.json"
        path.write_text(json.dumps(policy))
        previous = os.environ.get("SURE_SITE_POLICY")
        os.environ["SURE_SITE_POLICY"] = str(path)
        try:
            # These two real CUDA processes overlap; no model weights or large tensors.
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(run, [("onboard", "auto", 1), ("infer", "auto", 1)]))
            for stage in ["trans", "approve", "agent_eval", "eval"]:
                results.append(run((stage, "auto", 1)))
            results.extend([run(("infer", "auto", 2)), run(("onboard", "cpu", 1))])
            print(json.dumps({"passed": len(results), "results": results}, indent=2))
        finally:
            if previous is None:
                os.environ.pop("SURE_SITE_POLICY", None)
            else:
                os.environ["SURE_SITE_POLICY"] = previous


if __name__ == "__main__":
    main()
