"""Node-local, same-user onboard GPU leases and supervised command execution.

A lease covers one GPU execution segment. Waiting is cancellable and does not
consume the command's execution timeout. No daemon or scheduler service needed.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

try:
    import fcntl
except ImportError:  # GPU pooling is supported on Linux/POSIX hosts only.
    fcntl = None

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sure.runtime.gpu_partition import (
    GPU_CONTEXT, GpuPartitionError, enforce_gpu_context, is_docker_executable,
    read_context, restrict_docker_command,
)


def wrap_onboard_command(command, env, *, timeout=None, log_path=None):
    """Wrap only an unleased onboard GPU command; preserve all other paths."""
    context = read_context(env)
    if not context or context["stage"] != "onboard" or not context["devices"] or context.get("onboard_lease"):
        return command
    if not isinstance(command, list):
        raise GpuPartitionError("GPU_PARTITION: pooled onboard execution requires argv")
    result = [sys.executable, str(Path(__file__).resolve()), "--parent-pid", str(os.getpid())]
    if timeout is not None:
        result += ["--timeout", str(timeout)]
    if log_path is not None:
        result += ["--lease-log", str(log_path)]
    return [*result, "--", *command]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--lease-log")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    context = read_context(os.environ)
    if not context or context["stage"] != "onboard" or not context["devices"] or not command:
        parser.error("an onboard GPU context and command are required")
    if fcntl is None:
        parser.error("onboard GPU leases require POSIX flock")
    requested_at = time.time()
    cancelled = 0

    def cancel(signum, _frame):
        nonlocal cancelled
        cancelled = signum

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)

    def stopped():
        return cancelled or os.getppid() != args.parent_pid

    def event(status, **fields):
        record = {"event": status, "time": time.time(), "pid": os.getpid(), **fields}
        line = json.dumps(record, sort_keys=True)
        if args.lease_log:
            with open(args.lease_log, "a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        try:
            print("GPU_LEASE " + line, file=sys.stderr, flush=True)
        except BrokenPipeError:
            # A killed caller closes its capture pipe. Persisting the release
            # receipt must not depend on that caller still reading stderr.
            pass

    # One namespace per physical host and Unix account, shared by all checkouts
    # and policy files. Never unlink lock files: doing so splits flock identity.
    root = Path("/tmp") / f"sure-onboard-gpu-{os.getuid()}"
    root.mkdir(mode=0o700, exist_ok=True)
    devices = context["devices"]
    docker = is_docker_executable(command[0])
    if docker:
        # The host already normalized Docker GPU options. Preserve any explicit
        # narrower request, then let the allocator choose from just that set.
        flag = command.index("--gpus")
        devices = command[flag + 1].strip('"').removeprefix("device=").split(",")
        if not set(devices).issubset(context["devices"]):
            raise GpuPartitionError("GPU_PARTITION: Docker allocation crosses the onboard pool")
        command[flag + 1] = "all"
    event("waiting", candidates=devices)
    lock = None
    selected = None
    heartbeat = time.monotonic()
    while not stopped() and lock is None:
        stale = []
        for device in devices:
            handle = (root / f"{device}.lock").open("a+")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            if (root / f"{device}.owner.json").exists():
                # A killed supervisor or failed cleanup must not hand out a
                # possibly occupied card. Keep the receipt for manual recovery.
                stale.append(device)
                handle.close()
                continue
            lock, selected = handle, device
            break
        if len(stale) == len(devices):
            raise GpuPartitionError("GPU_PARTITION: abandoned onboard leases require cleanup: " + ",".join(stale))
        if lock is None:
            if time.monotonic() - heartbeat >= 30:
                event("waiting", candidates=devices)
                heartbeat = time.monotonic()
            time.sleep(0.2)
    if lock is None:
        event("cancelled", phase="waiting")
        return 128 + (cancelled or signal.SIGTERM)
    owner = root / f"{selected}.owner.json"
    owner.write_text(json.dumps({"supervisor_pid": os.getpid(), "requested_at": requested_at}))
    proc = None
    cidfile = root / f"{uuid.uuid4().hex}.cid"
    code = 1
    try:
        acquired_at = time.time()
        context = {**context, "devices": [selected], "onboard_lease": {
            "uuid": selected, "supervisor_pid": os.getpid(), "requested_at": requested_at,
            "acquired_at": acquired_at, "lock_path": str(root / f"{selected}.lock"),
        }}
        env = enforce_gpu_context(os.environ, source={GPU_CONTEXT: json.dumps(context)})
        if docker:
            command = restrict_docker_command(command, env)
            command[2:2] = ["--cidfile", str(cidfile)]
        event("acquired", uuid=selected, waited_seconds=acquired_at-requested_at, command=command)
        # The workload inherits the lease fd as an additional safeguard if its
        # supervisor is killed. The supervisor handles cancellation and Docker
        # daemon cleanup before releasing the lease on normal/timeout paths.
        proc = subprocess.Popen(command, env=env, start_new_session=True, pass_fds=(lock.fileno(),))
        started = time.monotonic()
        while proc.poll() is None:
            if stopped():
                code = 128 + (cancelled or signal.SIGTERM)
                break
            if args.timeout is not None and time.monotonic() - started >= args.timeout:
                event("timeout", uuid=selected, timeout_seconds=args.timeout)
                code = 124
                break
            time.sleep(0.1)
        else:
            code = proc.returncode
        return code
    finally:
        if proc is not None:
            # Also remove descendants left behind by a command that exited.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if docker and cidfile.exists():
                cid = cidfile.read_text().strip()
                if cid:
                    inspection = subprocess.run([command[0], "inspect", cid], capture_output=True, text=True, timeout=30)
                    # Some site Docker wrappers return zero even when inspect
                    # reports a missing object; use the daemon diagnostic.
                    if not any(text in inspection.stdout + inspection.stderr for text in ["No such object", "No such container"]):
                        subprocess.run([command[0], "rm", "-f", cid], capture_output=True, text=True, timeout=30)
                        inspection = subprocess.run([command[0], "inspect", cid], capture_output=True, text=True, timeout=30)
                    detail = inspection.stdout + inspection.stderr
                    if not any(text in detail for text in ["No such object", "No such container"]):
                        raise GpuPartitionError("GPU_PARTITION: cannot confirm onboard container cleanup: " + detail)
        cidfile.unlink(missing_ok=True)
        owner.unlink()
        event("released", uuid=selected, exit_code=code)
        lock.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GpuPartitionError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
