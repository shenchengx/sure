#!/usr/bin/env python3
"""Run ad-hoc onboard load evidence through the site's GPU boundary."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from sure.runtime.gpu_partition import GPU_CONTEXT, GpuPartitionError
from sure.site.gpu_partition import partition_command
from sure.runtime.onboard_gpu import wrap_onboard_command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--cwd")
    parser.add_argument("--lease-log", help="append GPU allocation events to this JSONL file")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("supply an argv command after --")
    try:
        command, env = partition_command(command, os.environ, "onboard", device=args.device)
        if GPU_CONTEXT in env:
            print(json.dumps({"gpu_partition": json.loads(env[GPU_CONTEXT])}, sort_keys=True), file=sys.stderr)
        command = wrap_onboard_command(command, env, log_path=args.lease_log)
        return subprocess.run(command, cwd=args.cwd, env=env, check=False).returncode
    except (GpuPartitionError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
