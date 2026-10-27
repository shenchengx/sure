"""GPU visibility enforcement shared by host launchers and model-side helpers.

This module is stdlib-only. Context is supplied by the host launcher, never
read from model configuration. It is cooperative isolation, not a sandbox.
"""
from __future__ import annotations

import json
import os
import re
from typing import Mapping

GPU_CONTEXT = "SURE_GPU_PARTITION_CONTEXT"
GPU_UUID_PATTERN = r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
GPU_ENV_KEYS = {GPU_CONTEXT, "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "DEVICE", "SURE_DEVICE", "SURE_EVAL_DEVICE_ACTUAL"}
STAGES = {"onboard", "infer", "trans", "approve", "eval", "agent_eval"}


class GpuPartitionError(ValueError):
    """A device request conflicts with the configured stage partition."""


def is_docker_executable(executable: str) -> bool:
    if os.path.basename(executable) == "docker":
        return True
    # These existing launcher overrides are trusted host configuration, not
    # values taken from the model's merged child environment.
    return any(
        value and os.path.abspath(os.path.expanduser(value)) == os.path.abspath(executable)
        for key in ("SURE_DOCKER_BIN", "SURE_ONBOARD_DOCKER_BIN", "SURE_EVAL_DOCKER_BIN")
        for value in [os.environ.get(key)]
    )


def read_context(source: Mapping[str, str]) -> dict | None:
    raw = source.get(GPU_CONTEXT)
    if raw is None:
        return None
    try:
        value = json.loads(raw)
        devices = value["devices"]
        onboard = value["onboard_uuids"]
        if value["stage"] not in STAGES or not isinstance(devices, list):
            raise ValueError("invalid stage/devices")
        if (not isinstance(onboard, list) or not onboard
                or any(not isinstance(item, str) or not re.fullmatch(GPU_UUID_PATTERN, item) for item in onboard)
                or len(set(onboard)) != len(onboard)):
            raise ValueError("invalid onboard UUID")
        if any(not isinstance(item, str) or not re.fullmatch(GPU_UUID_PATTERN, item) for item in devices):
            raise ValueError("invalid device UUID")
        if len(set(devices)) != len(devices):
            raise ValueError("duplicate devices")
        if any((item in onboard) != (value["stage"] == "onboard") for item in devices):
            raise ValueError("device crosses stage boundary")
        return {**value, "onboard_uuids": onboard}
    except (ValueError, TypeError, KeyError) as exc:
        raise GpuPartitionError(f"GPU_PARTITION: invalid inherited context: {exc}") from exc


def narrow_devices(raw: str, devices: list[str], indices: Mapping[str, str]) -> list[str]:
    if raw in {"", "-1", "none", "void"}:
        return []
    selected = []
    for token in raw.split(","):
        token = token.strip()
        uuid = indices.get(token, token)
        if uuid not in devices:
            raise GpuPartitionError(f"GPU_PARTITION: device {token!r} is outside the allowed GPU set")
        if uuid in selected:
            raise GpuPartitionError("GPU_PARTITION: duplicate GPU identity")
        selected.append(uuid)
    return selected


def enforce_gpu_context(
    env: Mapping[str, str], *, device: str | None = None, source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Apply inherited constraints LAST, after merging model/artifact env.

    source is the launcher's environment, not the merged model environment.
    Model-declared context values are removed even when policy is disabled.
    """
    source = os.environ if source is None else source
    result = dict(env)
    result.pop(GPU_CONTEXT, None)
    context = read_context(source)
    if context is None:
        return result
    devices = list(context["devices"])
    # A worker may have narrowed its parent's ordered UUID list to one GPU.
    visible = source.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = narrow_devices(visible, devices, {str(i): item for i, item in enumerate(devices)})
    requested = (device or "auto").lower()
    if requested not in {"cpu", "auto", "cuda"} and not (requested.startswith("cuda:") and requested[5:].isdigit()):
        raise GpuPartitionError(f"GPU_PARTITION: unsupported device {requested!r}")
    if requested == "cpu":
        devices = []
    elif requested.startswith("cuda:"):
        # At this boundary ordinals are process-local, not host indices.
        ordinal = requested.split(":", 1)[1]
        devices = narrow_devices(ordinal, devices, {str(i): item for i, item in enumerate(devices)})
    if not devices and requested not in {"cpu", "auto"}:
        raise GpuPartitionError("GPU_PARTITION: no permitted GPU for this request")
    context = {**context, "devices": devices}
    result[GPU_CONTEXT] = json.dumps(context, sort_keys=True)
    result["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    result["NVIDIA_VISIBLE_DEVICES"] = ",".join(devices) if devices else "void"
    actual = "cuda:0" if len(devices) == 1 else "cuda" if devices else "cpu"
    result.update(DEVICE=actual, SURE_DEVICE="cuda" if devices else "cpu", SURE_EVAL_DEVICE_ACTUAL=actual)
    return result


def restrict_docker_command(command: list[str], env: Mapping[str, str]) -> list[str]:
    """Normalize a structured docker run; refuse unhandled option syntax.

    Only Docker options before the image are interpreted. Application argv is
    preserved, including application flags named --device/--gpus.
    """
    context = read_context(env)
    if context is None:
        return command
    if len(command) < 3 or not is_docker_executable(command[0]) or command[1] != "run":
        raise GpuPartitionError("GPU_PARTITION: expected a structured docker run command")
    takes_value = {"--mount", "--volume", "-v", "--workdir", "-w", "--entrypoint", "--name", "--network", "--net", "--ipc", "--shm-size", "--ulimit", "--user", "-u", "--label", "-l", "--memory", "-m", "--cpus", "--runtime", "--hostname", "--pids-limit", "--security-opt", "--cap-drop", "--pull"}
    switches = {"--rm", "--init", "-i", "-t", "-it", "--interactive", "--tty", "--read-only", "--detach", "-d"}
    prefix = command[:2]
    devices = list(context["devices"])
    i = 2
    gpu_seen = False
    while i < len(command) and command[i].startswith("-"):
        token = command[i]
        key, equal, inline = token.partition("=")
        if key in {"--privileged", "--device", "--device-cgroup-rule", "--env-file", "--cap-add"}:
            raise GpuPartitionError(f"GPU_PARTITION: unsupported Docker option {key}")
        if key in switches and not equal:
            prefix.append(token)
            i += 1
            continue
        if key not in takes_value | {"--gpus", "--env", "-e"}:
            raise GpuPartitionError(f"GPU_PARTITION: unsupported Docker option {key}")
        if not equal and i + 1 >= len(command):
            raise GpuPartitionError(f"GPU_PARTITION: missing Docker value for {key}")
        value = inline if equal else command[i + 1]
        i += 1 if equal else 2
        if key == "--gpus":
            if gpu_seen:
                raise GpuPartitionError("GPU_PARTITION: duplicate --gpus")
            gpu_seen = True
            value = value.strip('"')
            if value.startswith("device="):
                devices = narrow_devices(value[7:], devices, context.get("host_indices", {}))
            elif value.isdigit():
                count = int(value)
                if count < 1 or count > len(devices):
                    raise GpuPartitionError("GPU_PARTITION: Docker GPU count exceeds allowed capacity")
                devices = devices[:count]
            elif value != "all":
                raise GpuPartitionError("GPU_PARTITION: unsupported Docker GPU request")
        elif key in {"--env", "-e"} and value.split("=", 1)[0] in GPU_ENV_KEYS:
            continue
        else:
            if key in {"--mount", "--volume", "-v"} and "/dev" in value:
                raise GpuPartitionError("GPU_PARTITION: device mounts are not allowed")
            prefix.extend([key, value])
    if i >= len(command):
        raise GpuPartitionError("GPU_PARTITION: Docker image is missing")
    final = enforce_gpu_context(env, source={GPU_CONTEXT: json.dumps({**context, "devices": devices})})
    if devices:
        prefix.extend(["--gpus", '"device=' + ",".join(devices) + '"'])
    for key in sorted(GPU_ENV_KEYS):
        if key in final:
            prefix.extend(["--env", f"{key}={final[key]}"])
    return [*prefix, *command[i:]]
