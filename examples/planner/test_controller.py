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


async def test_state_management(connector: LocalConnector) -> bool:
    """Test state file operations."""
    print("\n=== Testing State Management ===")
    try:
        # Test load state
        state = await connector.load_state()
        print("✓ Load state successful")
        
        # Test save state (with a copy)
        success = await connector.save_state(state)
        print(f"{'✓' if success else '✗'} Save state {'successful' if success else 'failed'}")
        
        return True
    except Exception as e:
        print(f"✗ State management test failed: {e}")
        return False

async def test_add_component(connector: LocalConnector) -> bool:
    """Test adding a component."""
    print("\n=== Testing Add Component ===")
    try:
        component = "VllmWorker"
        success = await connector.add_component(component)
    except Exception as e:
        print(f"✗ Add component test failed: {e}")
        return False

async def test_remove_component(connector: LocalConnector) -> bool:
    """Test removing a component."""
    print("\n=== Testing Remove Component ===")
    try:
        # Get state to find a component to remove
        state = await connector.load_state()
        
        # Look for VllmWorker components
        component_type = "VllmWorker"
        base_name = f"{connector.namespace}_{component_type}_"
        
        # Find all components with numbered suffixes
        matching_components = []
        for watcher_name in state["components"].keys():
            if watcher_name.startswith(base_name):
                try:
                    suffix = int(watcher_name.replace(base_name, ""))
                    matching_components.append((suffix, watcher_name))
                except ValueError:
                    continue
        
        if not matching_components:
            # No numbered components found, check for the base component
            base_component = f"{connector.namespace}_{component_type}"
            if base_component in state["components"]:
                print(f"Found base component {base_component} to remove")
                success = await connector.remove_component(component_type)
                print(f"{'✓' if success else '✗'} Remove component {'successful' if success else 'failed'}")
                return success
            else:
                print(f"✗ No {component_type} components found to remove")
                return False
        
        # Remove the component with highest suffix
        success = await connector.remove_component(component_type)
        
        # Verify removal
        new_state = await connector.load_state()
        highest_suffix = max(suffix for suffix, _ in matching_components)
        removed_component = f"{base_name}{highest_suffix}"
        
        if success and removed_component not in new_state["components"]:
            print(f"✓ Successfully removed {removed_component}")
            return True
        else:
            print(f"✗ Failed to remove {removed_component}")
            return False
            
    except Exception as e:
        print(f"✗ Remove component test failed: {e}")
        return False

async def main():
    parser = argparse.ArgumentParser(description="Test the LocalConnector")
    parser.add_argument("namespace", help="Dynamo namespace to use")
    parser.add_argument("--test", choices=["state", "add", "remove"], 
                      default="state", help="Specific test to run")
    
    args = parser.parse_args()
   
    # Check if namespace state file exists
    state_file = Path.home() / ".dynamo" / "state" / f"{args.namespace}.json"
    if not state_file.exists():
        print(f"Error: State file not found: {state_file}")
        return 1

    connector = LocalConnector(args.namespace)
    
    tests = {
        "state": test_state_management,
        "add": test_add_component,
        "remove": test_remove_component
    }

    await tests[args.test](connector)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
