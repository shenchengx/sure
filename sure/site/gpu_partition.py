"""Host-side stage GPU policy resolution. No allocation or scheduling."""
from __future__ import annotations

import json
import os
import subprocess
from typing import Mapping

from sure.runtime.gpu_partition import (
    GPU_CONTEXT, STAGES, GpuPartitionError, enforce_gpu_context, narrow_devices,
    read_context, restrict_docker_command, is_docker_executable,
)
from sure.site.loader import load_site_policy


def partition_environment(
    env: Mapping[str, str], stage: str, *, device: str = "auto", workers: int = 1,
) -> dict[str, str]:
    device = device.lower()
    if type(workers) is not int or workers < 1:
        raise GpuPartitionError("GPU_PARTITION: workers must be a positive integer")
    inherited = read_context(os.environ)
    if inherited is not None:
        if stage != inherited["stage"]:
            if stage == "onboard" or inherited["stage"] == "onboard":
                raise GpuPartitionError("GPU_PARTITION: cannot change onboard stage inside an execution")
            inherited = {**inherited, "stage": stage}
        result = enforce_gpu_context(env, device=device, source={**os.environ, GPU_CONTEXT: json.dumps(inherited)})
    else:
        loaded = load_site_policy()
        policy = loaded["policy"]["execution"].get("gpu_partition") if loaded else None
        if policy is None:
            result = dict(env)
            result.pop(GPU_CONTEXT, None)
            return result
        if device not in {"auto", "cuda", "cpu"} and not (device.startswith("cuda:") and device[5:].isdigit()):
            raise GpuPartitionError(f"GPU_PARTITION: unsupported device {device!r}")
        if stage not in STAGES:
            raise GpuPartitionError(f"GPU_PARTITION: unknown stage {stage!r}")
        onboard = policy["onboard_gpu_uuids"]
        indices = {}
        if device != "cpu":
            try:
                probe = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True, timeout=15, check=True)
                indices = dict(line.strip().split(", ", 1) for line in probe.stdout.splitlines() if line.strip())
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                raise GpuPartitionError(f"GPU_PARTITION: cannot enumerate node GPUs: {exc}") from exc
            if not set(onboard).issubset(indices.values()):
                raise GpuPartitionError(f"GPU_PARTITION: configured onboard UUID is absent: {onboard}")
        devices = list(indices.values())
        if device != "cpu" and "CUDA_VISIBLE_DEVICES" in os.environ:
            devices = narrow_devices(os.environ["CUDA_VISIBLE_DEVICES"], devices, indices)
        devices = [item for item in devices if (item in onboard) == (stage == "onboard")]
        requested = device.lower()
        if requested.startswith("cuda:"):
            devices = narrow_devices(requested.split(":", 1)[1], devices, indices)
        if requested != "cpu" and not devices:
            raise GpuPartitionError(f"GPU_PARTITION: no permitted GPU for stage {stage}")
        context = {"stage": stage, "onboard_uuids": onboard, "devices": devices, "host_indices": indices, "policy_path": loaded["path"], "policy_sha256": loaded["sha256"]}
        result = enforce_gpu_context(env, device="cpu" if requested == "cpu" else "auto", source={GPU_CONTEXT: json.dumps(context)})
    context = read_context(result)
    if context["devices"] and workers > len(context["devices"]):
        raise GpuPartitionError(f"GPU_PARTITION: num_workers={workers} exceeds allowed GPU capacity")
    if stage == "infer" and workers > 1:
        context = {**context, "devices": context["devices"][:workers]}
        result = enforce_gpu_context(result, source={GPU_CONTEXT: json.dumps(context)})
    return result


def partition_command(
    command: list[str] | str, env: Mapping[str, str], stage: str, *, device: str = "auto", workers: int = 1,
) -> tuple[list[str] | str, dict[str, str]]:
    result = partition_environment(env, stage, device=device, workers=workers)
    if GPU_CONTEXT not in result:
        return command, result
    if isinstance(command, str):
        raise GpuPartitionError("GPU_PARTITION: use an argv array instead of a shell command")
    if not command:
        raise GpuPartitionError("GPU_PARTITION: command is empty")
    if is_docker_executable(command[0]):
        command = restrict_docker_command(command, result)
    return command, result
