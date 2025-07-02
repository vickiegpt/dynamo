#  SPDX-FileCopyrightText: Copyright (c) 2020 Atalaya Tech. Inc
#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  Modifications Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES

from __future__ import annotations

import typing as t

import typer
from rich.console import Console
from rich.panel import Panel

from dynamo.sdk.cli.utils import build_docker_dict, build_env_dicts
from dynamo.sdk.core.deploy.consts import DeploymentTargetType
from dynamo.sdk.core.deploy.kubernetes import KubernetesDeploymentManager
from dynamo.sdk.core.protocol.deployment import (
    Deployment,
    DeploymentConfig,
    DeploymentManager,
    DeploymentResponse,
)

DEFAULT_DOCKER_IMAGE = "default_docker_image"

app = typer.Typer(
    help="Deploy Dynamo applications to Dynamo Cloud Platform",
    add_completion=True,
    no_args_is_help=True,
)

console = Console(highlight=False)


def get_deployment_manager(target: str, endpoint: str) -> DeploymentManager:
    """Return the appropriate DeploymentManager for the given target and endpoint."""
    try:
        target_enum = DeploymentTargetType(target)
    except ValueError:
        valid_targets = ", ".join([e.value for e in DeploymentTargetType])
        console.print(
            Panel(
                f"Invalid deployment target: {target}\nSupported targets: {valid_targets}",
                title="Error",
                style="red",
            )
        )
        raise typer.Exit(1)
    if target_enum == DeploymentTargetType.KUBERNETES:
        return KubernetesDeploymentManager(endpoint)
    else:
        raise ValueError(f"Unknown deployment target: {target_enum}")


def display_deployment_info(
    deployment_manager: DeploymentManager, deployment: DeploymentResponse
) -> None:
    """Display deployment summary, status, and endpoint URLs using rich panels."""
    name = deployment.get("name") or deployment.get("uid") or deployment.get("id")
    status = deployment_manager.get_status(name)
    urls = deployment_manager.get_endpoint_urls(name)
    created_at = deployment.get("created_at", "")
    summary = (
        f"[white]Name:[/] [cyan]{name}[/]\n"
        f"[white]Status:[/] [{status.color}]{status.value}[/]"
    )
    if created_at:
        summary += f"\n[white]Created:[/] [magenta]{created_at}[/]"
    if urls:
        summary += f"\n[white]URLs:[/] [blue]{' | '.join(urls)}[/]"
    else:
        summary += "\n[white]URLs:[/] [blue]None[/]"
    console.print(Panel(summary, title="Deployment", style="cyan"))


def _handle_deploy_create(
    ctx: typer.Context,
    config: DeploymentConfig,
) -> DeploymentResponse:
    """Handle deployment creation. This is a helper function for the create and deploy commands.

    Args:
        ctx: typer context
        config: DeploymentConfig object
    """

    deployment_manager = get_deployment_manager(config.target, config.endpoint)
    env_dicts = build_env_dicts(
        config_file=config.config_file,
        args=ctx.args,
        envs=config.envs,
        envs_from_secret=config.envs_from_secret,
        env_secrets_name=config.env_secrets_name,
    )
    docker_dict = build_docker_dict(
        docker=config.docker, default_docker_image_key=DEFAULT_DOCKER_IMAGE
    )
    deployment = Deployment(
        name=config.name or (config.graph if config.graph else "unnamed-deployment"),
        namespace="default",
        graph=config.graph,
        envs=env_dicts,
        docker=docker_dict,
    )
    try:
        console.print("[bold green]Creating deployment...")
        deployment = deployment_manager.create_deployment(
            deployment,
            dev=config.dev,
        )
        console.print(f"[bold green]Deployment '{config.name}' created.")
        if config.wait:
            deployment, ready = deployment_manager.wait_until_ready(
                config.name, timeout=config.timeout
            )
            if ready:
                console.print(
                    Panel(
                        f"Deployment [bold]{config.name}[/] is [green]ready[/]",
                        title="Status",
                    )
                )
            else:
                console.print(
                    Panel(
                        f"Deployment [bold]{config.name}[/] did not become ready in time.",
                        title="Status",
                        style="red",
                    )
                )
        display_deployment_info(deployment_manager, deployment)
        return deployment
    except Exception as e:
        if isinstance(e, RuntimeError) and isinstance(e.args[0], tuple):
            status, msg, url = e.args[0]
            if status == 409:
                console.print(
                    Panel(
                        f"Cannot create deployment because deployment with name '{config.name}' already exists.",
                        title="Error",
                        style="red",
                    )
                )
            elif status in (400, 422):
                console.print(
                    Panel(f"Validation error:\n{msg}", title="Error", style="red")
                )
            elif status == 404:
                console.print(
                    Panel(f"Not found: {url} \n{msg}", title="Error", style="red")
                )
            elif status == 500:
                console.print(
                    Panel(f"Internal server error:\n{msg}", title="Error", style="red")
                )
            else:
                console.print(
                    Panel(
                        f"Failed to create deployment:\n{msg}",
                        title="Error",
                        style="red",
                    )
                )
        else:
            console.print(Panel(str(e), title="Error", style="red"))
        raise typer.Exit(1)


@app.command()
def create(
    ctx: typer.Context,
    graph: str = typer.Argument(..., help="Dynamo graph to deploy"),
    name: t.Optional[str] = typer.Option(None, "--name", "-n", help="Deployment name"),
    config_file: t.Optional[typer.FileText] = typer.Option(
        None, "--config-file", "-f", help="Configuration file path"
    ),
    wait: bool = typer.Option(
        True, "--wait/--no-wait", help="Do not wait for deployment to be ready"
    ),
    timeout: int = typer.Option(
        3600, "--timeout", help="Timeout for deployment to be ready in seconds"
    ),
    endpoint: str = typer.Option(
        ..., "--endpoint", "-e", help="Dynamo Cloud endpoint", envvar="DYNAMO_CLOUD"
    ),
    envs: t.Optional[t.List[str]] = typer.Option(
        None,
        "--env",
        help="Environment variable(s) to set (format: KEY=VALUE). Note: These environment variables will be set on ALL services in your Dynamo graph.",
    ),
    envs_from_secret: t.Optional[t.List[str]] = typer.Option(
        None,
        "--env-from-secret",
        help="Environment variable(s) from secret (format: KEY=SECRET_KEY). These will be set from your Dynamo secrets.",
    ),
    target: str = typer.Option(
        DeploymentTargetType.KUBERNETES.value,
        "--target",
        "-t",
        help="Deployment target",
    ),
    dev: bool = typer.Option(False, "--dev", help="Development mode for deployment"),
    env_secrets_name: t.Optional[str] = typer.Option(
        "dynamo-env-secrets",
        "--env-secrets-name",
        help="Environment secrets name",
        envvar="DYNAMO_ENV_SECRETS",
    ),
    docker: t.Optional[t.List[str]] = typer.Option(
        None,
        "--docker",
        help="Docker image tag. Use without '=' for default image, or 'ComponentName=image:tag' for component-specific overrides.",
    ),
) -> DeploymentResponse:
    """Create a deployment on Dynamo Cloud."""
    config = DeploymentConfig(
        graph=graph,
        endpoint=endpoint,
        name=name,
        config_file=config_file,
        wait=wait,
        timeout=timeout,
        envs=envs,
        envs_from_secret=envs_from_secret,
        target=target,
        dev=dev,
        env_secrets_name=env_secrets_name,
        docker=docker,
    )
    return _handle_deploy_create(ctx, config)


@app.command()
def get(
    name: str = typer.Argument(..., help="Deployment name"),
    target: str = typer.Option(
        DeploymentTargetType.KUBERNETES.value,
        "--target",
        "-t",
        help="Deployment target",
    ),
    endpoint: str = typer.Option(
        ..., "--endpoint", "-e", help="Dynamo Cloud endpoint", envvar="DYNAMO_CLOUD"
    ),
) -> DeploymentResponse:
    """Get details for a specific deployment by name."""
    deployment_manager = get_deployment_manager(target, endpoint)
    try:
        with console.status(f"[bold green]Getting deployment '{name}'..."):
            deployment = deployment_manager.get_deployment(name)
            display_deployment_info(deployment_manager, deployment)
            return deployment
    except Exception as e:
        if isinstance(e, RuntimeError) and isinstance(e.args[0], tuple):
            status, msg, _ = e.args[0]
            if status == 404:
                console.print(
                    Panel(
                        f"Deployment '{name}' not found.\n{msg}",
                        title="Error",
                        style="red",
                    )
                )
            else:
                console.print(
                    Panel(
                        f"Failed to get deployment:\n{msg}", title="Error", style="red"
                    )
                )
        else:
            console.print(Panel(str(e), title="Error", style="red"))
        raise typer.Exit(1)


@app.command("list")
def list_deployments(
    target: str = typer.Option(
        DeploymentTargetType.KUBERNETES.value,
        "--target",
        "-t",
        help="Deployment target",
    ),
    endpoint: str = typer.Option(
        ..., "--endpoint", "-e", help="Dynamo Cloud endpoint", envvar="DYNAMO_CLOUD"
    ),
) -> None:
    """List all deployments."""
    deployment_manager = get_deployment_manager(target, endpoint)
    try:
        with console.status("[bold green]Listing deployments..."):
            deployments = deployment_manager.list_deployments()
            if not deployments:
                console.print(
                    Panel("No deployments found.", title="Deployments", style="yellow")
                )
            else:
                console.print(Panel("[bold]Deployments List[/]", style="blue"))
                for dep in deployments:
                    display_deployment_info(deployment_manager, dep)
    except Exception as e:
        if isinstance(e, RuntimeError) and isinstance(e.args[0], tuple):
            status, msg, url = e.args[0]
            if status == 404:
                console.print(
                    Panel(
                        f"Endpoint not found: {url}\n{msg}", title="Error", style="red"
                    )
                )
            else:
                console.print(
                    Panel(
                        f"Failed to list deployments:\n{msg}",
                        title="Error",
                        style="red",
                    )
                )
        else:
            console.print(Panel(str(e), title="Error", style="red"))
        raise typer.Exit(1)


@app.command()
def update(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Deployment name to update"),
    target: str = typer.Option(
        DeploymentTargetType.KUBERNETES.value,
        "--target",
        "-t",
        help="Deployment target",
    ),
    config_file: t.Optional[typer.FileText] = typer.Option(
        None, "--config-file", "-f", help="Configuration file path"
    ),
    envs: t.Optional[t.List[str]] = typer.Option(
        None,
        "--env",
        help="Environment variable(s) to set (format: KEY=VALUE). Note: These environment variables will be set on ALL services in your Dynamo graph.",
    ),
    envs_from_secret: t.Optional[t.List[str]] = typer.Option(
        None,
        "--env-from-secret",
        help="Environment variable(s) from secret (format: KEY=SECRET_KEY). These will be set from your Dynamo secrets.",
    ),
    endpoint: str = typer.Option(
        ..., "--endpoint", "-e", help="Dynamo Cloud endpoint", envvar="DYNAMO_CLOUD"
    ),
    env_secrets_name: t.Optional[str] = typer.Option(
        "dynamo-env-secrets",
        "--env-secrets-name",
        help="Environment secrets name",
        envvar="DYNAMO_ENV_SECRETS",
    ),
    docker: t.Optional[t.List[str]] = typer.Option(
        None,
        "--docker",
        help="Docker image tag. Use without '=' for default image, or 'ComponentName=image:tag' for component-specific overrides.",
    ),
) -> None:
    """Update an existing deployment on Dynamo Cloud.

    Update a deployment using parameters or a config yaml file.
    """
    deployment_manager = get_deployment_manager(target, endpoint)
    try:
        with console.status(f"[bold green]Updating deployment '{name}'..."):
            env_dicts = build_env_dicts(
                config_file=config_file,
                args=ctx.args,
                envs=envs,
                envs_from_secret=envs_from_secret,
                env_secrets_name=env_secrets_name,
            )
            docker_dict = build_docker_dict(
                docker=docker, default_docker_image_key=DEFAULT_DOCKER_IMAGE
            )
            deployment = Deployment(
                name=name,
                namespace="default",
                envs=env_dicts,
                docker=docker_dict,
            )
            deployment_manager.update_deployment(
                deployment_id=name, deployment=deployment
            )
            console.print(
                Panel(
                    "[yellow]Update submitted. It may take a short time for the new pods to become active. Please wait a bit before accessing the deployment to ensure your changes are live.[/yellow]",
                    title="Status",
                )
            )
    except Exception as e:
        if isinstance(e, RuntimeError) and isinstance(e.args[0], tuple):
            status, msg, url = e.args[0]
            if status == 404:
                console.print(
                    Panel(
                        f"Deployment '{name}' not found.\n{msg}",
                        title="Error",
                        style="red",
                    )
                )
            else:
                console.print(
                    Panel(
                        f"Failed to update deployment:\n{msg}",
                        title="Error",
                        style="red",
                    )
                )
        else:
            console.print(Panel(str(e), title="Error", style="red"))
        raise typer.Exit(1)


@app.command()
def delete(
    name: str = typer.Argument(..., help="Deployment name"),
    target: str = typer.Option(
        DeploymentTargetType.KUBERNETES.value,
        "--target",
        "-t",
        help="Deployment target",
    ),
    endpoint: str = typer.Option(
        ..., "--endpoint", "-e", help="Dynamo Cloud endpoint", envvar="DYNAMO_CLOUD"
    ),
) -> None:
    """Delete a deployment by name."""
    deployment_manager = get_deployment_manager(target, endpoint)
    try:
        with console.status(f"[bold green]Deleting deployment '{name}'..."):
            deployment_manager.delete_deployment(name)
            console.print(
                Panel(f"Deleted deployment {name}", title="Success", style="green")
            )
    except Exception as e:
        if isinstance(e, RuntimeError) and isinstance(e.args[0], tuple):
            status, msg, _ = e.args[0]
            if status == 404:
                console.print(
                    Panel(
                        f"Deployment '{name}' not found.",
                        title="Error",
                        style="red",
                    )
                )
            else:
                console.print(
                    Panel(
                        f"Failed to delete deployment:\n{msg}",
                        title="Error",
                        style="red",
                    )
                )
        else:
            console.print(Panel(str(e), title="Error", style="red"))
        raise typer.Exit(1)


def deploy(
    ctx: typer.Context,
    graph: str = typer.Argument(..., help="Dynamo graph to deploy"),
    name: t.Optional[str] = typer.Option(None, "--name", "-n", help="Deployment name"),
    config_file: t.Optional[typer.FileText] = typer.Option(
        None, "--config-file", "-f", help="Configuration file path"
    ),
    wait: bool = typer.Option(
        True, "--wait/--no-wait", help="Do not wait for deployment to be ready"
    ),
    timeout: int = typer.Option(
        3600, "--timeout", help="Timeout for deployment to be ready in seconds"
    ),
    endpoint: str = typer.Option(
        ..., "--endpoint", "-e", help="Dynamo Cloud endpoint", envvar="DYNAMO_CLOUD"
    ),
    envs: t.Optional[t.List[str]] = typer.Option(
        None,
        "--env",
        help="Environment variable(s) to set (format: KEY=VALUE). Note: These environment variables will be set on ALL services in your Dynamo graph.",
    ),
    envs_from_secret: t.Optional[t.List[str]] = typer.Option(
        None,
        "--env-from-secret",
        help="Environment variable(s) from secret (format: KEY=SECRET_KEY). These will be set from your Dynamo secrets.",
    ),
    target: str = typer.Option(
        DeploymentTargetType.KUBERNETES.value,
        "--target",
        "-t",
        help="Deployment target",
    ),
    dev: bool = typer.Option(False, "--dev", help="Development mode for deployment"),
    env_secrets_name: t.Optional[str] = typer.Option(
        "dynamo-env-secrets",
        "--env-secrets-name",
        help="Environment secrets name",
        envvar="DYNAMO_ENV_SECRETS",
    ),
    docker: t.Optional[t.List[str]] = typer.Option(
        None,
        "--docker",
        help="Docker image tag. Use without '=' for default image, or 'ComponentName=image:tag' for component-specific overrides.",
    ),
) -> DeploymentResponse:
    """Deploy a Dynamo graph (same as deployment create)."""
    config = DeploymentConfig(
        graph=graph,
        endpoint=endpoint,
        name=name,
        config_file=config_file,
        wait=wait,
        timeout=timeout,
        envs=envs,
        envs_from_secret=envs_from_secret,
        target=target,
        dev=dev,
        env_secrets_name=env_secrets_name,
        docker=docker,
    )
    return _handle_deploy_create(ctx, config)
