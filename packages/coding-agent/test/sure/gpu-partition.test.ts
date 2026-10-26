import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import { resolveSitePolicy } from "../../../../sure/site/loader.ts";

const uuid = "GPU-00000007-0000-0000-0000-000000000000";
const directories: string[] = [];
const loader = fileURLToPath(new URL("../../../../sure/site/loader.py", import.meta.url));

afterEach(() => {
	for (const directory of directories.splice(0)) rmSync(directory, { recursive: true });
});

describe("GPU partition site policy parity", () => {
	it.each([
		{ onboard_gpu_uuid: uuid },
		null,
		{},
		[],
		{ onboard_gpu_uuid: "7" },
		{ onboard_gpu_uuid: [uuid] },
		{ onboard_gpu_uuid: uuid, enabled: true },
		{ onboard_gpu_uuid: "GPU-ABCDEF00-0000-0000-0000-000000000000" },
	])("agrees with Python for %j", (partition) => {
		const directory = mkdtempSync(join(tmpdir(), "sure-gpu-policy-"));
		directories.push(directory);
		const path = join(directory, "site.yaml");
		writeFileSync(
			path,
			JSON.stringify({
				schema: "sure.site.policy.v1",
				site_id: "test",
				policy_version: 1,
				storage: {
					approved_models_roots: ["/models"],
					forbidden_output_roots: ["/models"],
					runtime_root: "/runtime",
				},
				datasets: { allowed_source_roots: { test: "/data" } },
				execution: { surfaces: ["local"], gpu_partition: partition },
			}),
		);
		const environment = { ...process.env, SURE_SITE_POLICY: path };
		const python = spawnSync("python3", [loader], { env: environment, encoding: "utf8" });
		if (
			partition &&
			!Array.isArray(partition) &&
			"onboard_gpu_uuid" in partition &&
			partition.onboard_gpu_uuid === uuid &&
			Object.keys(partition).length === 1
		) {
			expect(python.status, python.stderr).toBe(0);
			expect(resolveSitePolicy({ environment })?.policy).toEqual(JSON.parse(python.stdout).policy);
		} else {
			expect(python.status).not.toBe(0);
			expect(() => resolveSitePolicy({ environment })).toThrow(/gpu_partition/);
		}
	});
});
