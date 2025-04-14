#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

from circusd import CircusController
from local_connector import LocalConnector

ComponentType = Literal["VllmWorker", "PrefillWorker"]
VALID_COMPONENTS = ["VllmWorker", "PrefillWorker"]

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

async def test_add_component(connector: LocalConnector, component: ComponentType) -> bool:
    """Test adding a component."""
    print(f"\n=== Testing Add Component: {component} ===")
    try:
        success = await connector.add_component(component)
        print(f"{'✓' if success else '✗'} Add {component} {'successful' if success else 'failed'}")
        return success
    except Exception as e:
        print(f"✗ Add {component} test failed: {e}")
        return False

async def test_remove_component(connector: LocalConnector, component: ComponentType) -> bool:
    """Test removing a component."""
    print(f"\n=== Testing Remove Component: {component} ===")
    try:
        # Get state to find a component to remove
        state = await connector.load_state()
        
        # Look for VllmWorker components
        base_name = f"{connector.namespace}_{component}_"
        
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
            base_component = f"{connector.namespace}_{component}"
            if base_component in state["components"]:
                print(f"Found base component {base_component} to remove")
                success = await connector.remove_component(component)
                print(f"{'✓' if success else '✗'} Remove {component} {'successful' if success else 'failed'}")
                return success
            else:
                print(f"✗ No {component} components found to remove")
                return False
        
        # Remove the component with highest suffix
        success = await connector.remove_component(component)
        
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
        print(f"✗ Remove {component} test failed: {e}")
        return False

async def main():
    parser = argparse.ArgumentParser(description="Test the LocalConnector")
    parser.add_argument("namespace", help="Dynamo namespace to use")
    parser.add_argument("--test", choices=["state", "add", "remove"], 
                      required=True, help="Specific test to run")
    parser.add_argument("--component", choices=VALID_COMPONENTS,
                      help="Component type (required for add/remove operations)")
    
    args = parser.parse_args()
    
    # Validate component argument for add/remove operations
    if args.test in ["add", "remove"]:
        if not args.component:
            parser.error(f"--component is required for {args.test} operation")
   
    # Check if namespace state file exists
    state_file = Path.home() / ".dynamo" / "state" / f"{args.namespace}.json"
    if not state_file.exists():
        print(f"Error: State file not found: {state_file}")
        return 1

    connector = LocalConnector(args.namespace)
    
    tests = {
        "state": lambda: test_state_management(connector),
        "add": lambda: test_add_component(connector, args.component),
        "remove": lambda: test_remove_component(connector, args.component)
    }

    success = await tests[args.test]()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
