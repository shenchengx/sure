# Multi-GPU Inference

`/sure_infer` runs one model server per GPU. Automatic selection is the default: a
large dataset uses the cards this node has, and a small one uses a single server.

This is process-level parallelism, not model batching. Each Worker loads its own copy
of the model and calls it one sample at a time, so no model wrapper needs a batch
argument and predictions are identical to a single-server run.

## Selecting Workers

`num_workers` is optional.

The automatic count is `ceil(selected_samples / 1000)`, capped at the GPUs
available to inference after stage partitioning and any inherited allocation.
Selected samples are counted after `max_samples` is applied.

| Selected samples | Workers before the GPU cap |
| --- | --- |
| 1–1000 | 1 |
| 1001–2000 | 2 |
| 2001–3000 | 3 |
| 5001–6000 | 6 |
| 6001–7000 | 7 |

Requested counts may be any positive integer that fits the available GPU count.
For example, on an eight-GPU node with two onboard GPUs, a 7060-sample dataset
plans six Workers. There is no fixed 1/2/4/7 tier or eight-Worker limit. With no
usable GPU, automatic planning keeps one Worker for the existing CPU path.

An explicit `num_workers` is honoured up to the node's GPU count. Above it, input
resolution fails before any model is loaded:

```text
num_workers=7 asks for 7 GPUs but this node provides 2
```

`num_workers=1` always uses the single-server path. More than one Worker requires
`device=auto` or `device=cuda`; `device=cuda:<index>` narrows the run to one card and
is refused, never widened.

A mixed run plans each dataset separately and records the maximum in the resolved
runtime. Datasets are processed in input order, one at a time: models are loaded
again for the next dataset rather than held across the whole run.

## Execution

Each Worker is a separate process pinned to one device through
`CUDA_VISIBLE_DEVICES`, and uses `cuda:0` inside that process. Model configuration
cannot widen that visibility.

Keys are split into contiguous shards whose sizes differ by at most one, in source
order. A dataset with fewer samples than Workers uses fewer Workers.

Both local surfaces work:

- **`local_docker`** — the launcher passes `--gpus "device=…"` for the planned cards
  and does not forward the host `CUDA_VISIBLE_DEVICES`, because device ids are
  renumbered inside a restricted container. The entrypoint probes CUDA in Model
  Python before loading anything and derives the container ordinals.
- **`local_python`** — the host allocation is already the truth, so the node's
  `CUDA_VISIBLE_DEVICES` reaches the entrypoint unchanged.

When `execution.gpu_partition.onboard_gpu_uuids` is configured, all listed physical
GPUs are reserved for onboard and excluded from inference before worker planning:

```yaml
execution:
  gpu_partition:
    onboard_gpu_uuids:
      - GPU-b7e42452-fcc6-f8a2-b5e6-cd294ccdc924
      - GPU-7c2abd9a-d3cb-b803-3569-0f20096294c8
```

Use full UUIDs queried on the execution node. A one-item list reserves one GPU.
The list must be nonempty and unique; the former singular `onboard_gpu_uuid` field
is rejected and must be migrated to the list. All configured UUIDs must exist on
the node; inherited visibility may further narrow each stage's allocation.
Launchers restrict Docker and Python environments to the stage's GPU UUIDs. Each
infer Worker still uses process-local `cuda:0`. Onboard sees its permitted pool;
reserving multiple GPUs does not add model parallelism. Onboard GPU commands
now acquire one exclusive card from the pool: two free cards can run two commands
concurrently, and further commands wait until a card is released. Run contexts
record the pool in `onboard_uuids` and the selected card in `devices` plus
`onboard_lease`. Model code continues to use `cuda:0`.

The allocator covers each GPU execution segment (environment probe, validation,
server smoke or `run_gpu_command.py`), releasing the card between segments.
Nested helpers inherit the existing lease. It is a node-local, same-Unix-user
allocator shared across checkouts and policy files, using persistent lock files
under `/tmp/sure-onboard-gpu-<uid>/`. Other Unix accounts and programs that bypass
the controlled launchers do not participate. Waiting polls all eligible cards;
strict FIFO ordering is not guaranteed. No free-memory heuristic is used.

Waiting is cancellable and does not consume the model command's execution
timeout; an enclosing gate or user cancellation can still stop the operation.
The supervisor stops workload process groups and removes its own Docker
container before releasing a card. A killed caller is detected and cleaned up.
If the supervisor itself is killed with SIGKILL, or cleanup cannot be confirmed,
an owner receipt prevents reuse of that card. Check the recorded processes and
container before removing that specific `.owner.json` receipt; never delete the
`.lock` file. This prevents an uncertain cleanup from silently causing overlap.

Validation runs append `waiting`, `acquired`, `timeout`/`cancelled` and `released`
events to `artifacts/<kind>_execution.gpu-lease.jsonl` (or beside a custom log).
`run_gpu_command.py --lease-log <path>` can persist the same events; they also
appear on stderr. Acquisition events record the final UUID and actual command.
The command execution timeout begins after acquisition; logged total validation
duration includes waiting.

The smoke pass runs first, on one GPU, under `parallel_inference/smoke/`, so a broken
model server is caught before the full dataset is started.

## Failures and recovery

Each Worker appends every successful sample to its own `completed.jsonl` and flushes
it. A Worker that fails does not terminate its peers, and a truncated final line is
read as "that sample did not complete" rather than as a prediction.

Sample errors leave the rest of the Worker's shard running when the server remains
usable. Only missing keys are retried, on their original GPU, at most twice after the
first attempt. Proven input or configuration errors are not retried automatically.

A stdio exchange that exceeds the approved server's `timeout` or `startup_timeout_sec`
(300 seconds by default) ends that attempt and cleans up its server process.

If a dataset still has missing keys, the execution fails, the dataset is not marked
completed, and every successful shard stays on disk. Later datasets in the same run
are still processed.

To reuse those keys, resume the same product directory:

```bash
"$HARNESS_PYTHON_BIN" scripts/run_infer.py \
  --run-dir <original_control_run_dir> \
  --resume-predictions
```

Recovery verifies the dataset projection, model configuration, approved runtime,
protocol and sample scope before reusing anything, then starts new attempt
directories and loads models only for the missing keys. A changed identity requires a
new product directory. Ordinary runs do not reuse previous attempts: a multi-Worker
run starts with `NO_RESUME=1` by default.

A shared out-of-memory condition can stop every Worker at once. It is reported from
the Worker logs as `gpu_oom` or `host_oom`, and the journals remain usable for
recovery. There is currently no memory preflight: N Workers each load a full model,
so a Worker count that fits the GPUs may still not fit host RAM.

## Artifacts

```text
<output_dir>/
├── predictions/<dataset>.txt
├── predictions/<dataset>.jsonl
├── prediction_generation_status.json
└── parallel_inference/<dataset>/
    ├── plan.json
    ├── merge_report.json
    ├── sessions/<session>.json
    └── workers/worker_<rank>/attempt_<number>/
        ├── attempt.json
        ├── keys.txt
        ├── completed.jsonl
        └── worker.log
```

`merge_report.json` is the authority on completeness: a successful dataset has
`status: completed`, `expected == generated` and `missing_keys: []`.

`execution_result.json` adds a `parallelism` block with the requested Worker count,
the selection mode (`auto` / `explicit`), the GPU list, and per dataset the planned and
effective Worker counts, sample scope, status and report path. The execution gate
rejects a succeeded run whose plan, generation status, merge report or GPU list
disagrees, so a tampered or drifted Worker count cannot pass.
