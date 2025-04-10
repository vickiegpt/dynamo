#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import json
import sys
from pathlib import Path

from circusd import CircusController
from local_connector import LocalConnector


async def test_circus_controller(namespace: str, component: str = None):
    """Test CircusController functionality directly."""
    print(f"Testing CircusController with namespace: {namespace}")

    try:
        circus = CircusController.from_state_file(namespace)
        print(f"Connected to circus endpoint: {circus.endpoint}")

        # List all watchers
        watchers = await circus.list_watchers()
        print(f"\nFound {len(watchers)} watchers:")
        for watcher in watchers:
            print(f"  - {watcher}")

        if component:
            # Component-specific operations
            watcher_name = f"dynamo_{component}"
            if watcher_name in watchers:
                # Get process count
                process_count = await circus.get_watcher_processes(watcher_name)
                print(f"\nComponent '{component}' has {process_count} processes")

                # Get stats
                stats = await circus.get_watcher_info(watcher_name)
                print(f"\nStats for '{component}':")
                print(json.dumps(stats, indent=2))
            else:
                print(f"\nComponent '{component}' not found in watchers")

    except Exception as e:
        print(f"Error testing CircusController: {e}")
        return False

    return True


async def test_local_connector(namespace: str, component: str = None):
    """Test LocalConnector functionality."""
    print(f"Testing LocalConnector with namespace: {namespace}")

    try:
        connector = LocalConnector(namespace)

        # List all components
        components = await connector.list_components()
        print(f"\nFound {len(components)} components:")
        for comp in components:
            print(
                f"  - {comp['name']}: {comp['replicas']} replicas, status: {comp['status']}"
            )

        # Get system topology
        topology = await connector.get_system_topology()
        print("\nSystem Topology:")
        print(json.dumps(topology, indent=2))

        if component:
            # Component-specific operations
            # Get component replicas
            replicas = await connector.get_component_replicas(component)
            print(f"\nComponent '{component}' has {replicas} replicas")

            # Get resource usage
            resources = await connector.get_resource_usage(component)
            print(f"\nResource usage for '{component}':")
            print(json.dumps(resources, indent=2))

            # Test scaling if component exists
            if replicas > 0:
                # Interactive scaling
                should_scale = (
                    input(
                        f"\nDo you want to test scaling '{component}'? (y/n): "
                    ).lower()
                    == "y"
                )
                if should_scale:
                    target = int(
                        input(
                            f"Enter target number of replicas for '{component}' (current: {replicas}): "
                        )
                    )
                    print(f"Scaling '{component}' to {target} replicas...")
                    success = await connector.scale_component(component, target)
                    print(f"Scaling {'succeeded' if success else 'failed'}")

                    # Verify new count
                    new_replicas = await connector.get_component_replicas(component)
                    print(f"Component '{component}' now has {new_replicas} replicas")

                # Interactive restart
                should_restart = (
                    input(
                        f"\nDo you want to test restarting '{component}'? (y/n): "
                    ).lower()
                    == "y"
                )
                if should_restart:
                    print(f"Restarting '{component}'...")
                    success = await connector.restart_component(component)
                    print(f"Restart {'succeeded' if success else 'failed'}")
            else:
                print(
                    f"\nComponent '{component}' is not running, cannot test scaling/restart"
                )

    except Exception as e:
        print(f"Error testing LocalConnector: {e}")
        return False

    return True


async def main():
    parser = argparse.ArgumentParser(
        description="Test the CircusController and LocalConnector"
    )
    parser.add_argument("namespace", help="Dynamo namespace to use")
    parser.add_argument("--component", "-c", help="Component name to test (optional)")
    parser.add_argument(
        "--circus-only", action="store_true", help="Test only CircusController"
    )
    parser.add_argument(
        "--connector-only", action="store_true", help="Test only LocalConnector"
    )

    args = parser.parse_args()

    # Check if namespace state file exists
    state_file = Path.home() / ".dynamo" / "state" / f"{args.namespace}.json"
    if not state_file.exists():
        print(f"Error: State file not found: {state_file}")
        print("Make sure you've started Dynamo with the specified namespace")
        return 1

    if args.circus_only:
        await test_circus_controller(args.namespace, args.component)
    elif args.connector_only:
        await test_local_connector(args.namespace, args.component)
    else:
        # Test both
        print("=== Testing CircusController ===")
        await test_circus_controller(args.namespace, args.component)

        print("\n=== Testing LocalConnector ===")
        await test_local_connector(args.namespace, args.component)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
