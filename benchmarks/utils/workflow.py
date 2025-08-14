# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from benchmarks.utils.genai import run_concurrency_sweep
from benchmarks.utils.plot import generate_plots
from deploy.utils.dynamo_deployment import DynamoDeploymentClient


async def deploy_and_wait(client: DynamoDeploymentClient, manifest_path: str) -> None:
    await client.create_deployment(manifest_path)
    await client.wait_for_deployment_ready(timeout=1800)


async def teardown(client: DynamoDeploymentClient) -> None:
    try:
        await client.delete_deployment()
    except Exception:
        pass


async def run_benchmark_workflow(
    namespace: str,
    agg_manifest: str,
    disagg_manifest: str,
    isl: int,
    std: int,
    osl: int,
    concurrency: int,
    model: str,
    output_dir: str,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Deploy and benchmark aggregated
    agg_name = Path(agg_manifest).stem
    agg_client = DynamoDeploymentClient(namespace=namespace, deployment_name=agg_name)
    await deploy_and_wait(agg_client, agg_manifest)
    try:
        run_concurrency_sweep(
            service_url=agg_client.get_service_url(),
            model_name=model,
            isl=isl,
            osl=osl,
            stddev=std,
            output_dir=Path(output_dir) / "agg",
        )
    finally:
        await teardown(agg_client)

    # Deploy and benchmark disaggregated
    disagg_name = Path(disagg_manifest).stem
    disagg_client = DynamoDeploymentClient(
        namespace=namespace, deployment_name=disagg_name
    )
    await deploy_and_wait(disagg_client, disagg_manifest)
    try:
        run_concurrency_sweep(
            service_url=disagg_client.get_service_url(),
            model_name=model,
            isl=isl,
            osl=osl,
            stddev=std,
            output_dir=Path(output_dir) / "disagg",
        )
    finally:
        await teardown(disagg_client)

    # Generate plots across outputs
    generate_plots(base_output_dir=Path(output_dir))
