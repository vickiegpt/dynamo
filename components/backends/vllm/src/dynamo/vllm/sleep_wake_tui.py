# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import sys
from dataclasses import dataclass
from typing import Any, Optional

import aiohttp

from dynamo.runtime import DistributedRuntime


@dataclass
class InstanceInfo:
    instance_id: int
    sleeping: Optional[bool] = None


async def fetch_sleep_state(is_sleeping_client, instance_id: int) -> bool:
    try:
        # Call the is_sleeping endpoint directly on the targeted instance
        stream = await is_sleeping_client.direct({}, instance_id)
        msg = await anext(stream)
        raw = msg.data() if hasattr(msg, "data") else msg
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", "ignore")
        if isinstance(raw, str):
            data = json.loads(raw)
        elif isinstance(raw, dict):
            data = raw
        else:
            # Unknown type; treat as awake
            return False
        return bool(data.get("sleeping", False))
    except StopAsyncIteration:
        return False
    except Exception:
        return False


async def call_sleep(who_client, instance_ids: list[int], level: int = 3) -> None:
    tasks = []
    for iid in instance_ids:
        tasks.append(_call_single(who_client, iid, {"level": level}))
    await asyncio.gather(*tasks)


async def call_wake(who_client, instance_ids: list[int]) -> None:
    tasks = []
    for iid in instance_ids:
        tasks.append(_call_single(who_client, iid, {}))
    await asyncio.gather(*tasks)


async def _call_single(client, instance_id: int, payload: dict[str, Any]) -> None:
    try:
        stream = await client.direct(payload, instance_id)
        # Drain one message for status
        await anext(stream)
    except StopAsyncIteration:
        pass
    except Exception as e:
        print(f"ERROR calling endpoint for instance {instance_id}: {e}")
        import traceback
        traceback.print_exc()


async def list_instances(generate_client, is_sleeping_client) -> list[InstanceInfo]:
    # Ensure discovery happened
    await generate_client.wait_for_instances()
    ids = generate_client.instance_ids()
    infos: list[InstanceInfo] = []
    # Query sleep state per instance
    for iid in ids:
        sleeping = await fetch_sleep_state(is_sleeping_client, iid)
        infos.append(InstanceInfo(instance_id=iid, sleeping=sleeping))
    return infos


def print_instances(infos: list[InstanceInfo]) -> None:
    if not infos:
        print("No instances found.")
        return
    print("\nInstances:")
    print("Idx  Instance ID           State")
    for idx, info in enumerate(infos):
        state = "SLEEPING" if info.sleeping else "AWAKE"
        print(f"{idx:>3}  {info.instance_id:<20}  {state}")


async def send_test_chat(http_url: str, model: str, instance_id: int) -> None:
    url = http_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Who won the world series in 2020?",
            },
        ],
        "max_tokens": 128,
        "nvext": {"backend_instance_id": instance_id},
        # deterministic small request
        "temperature": 0.0,
        "stream": False,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            text = await resp.text()
            if resp.status != 200:
                print(f"HTTP {resp.status}: {text}")
                return
            try:
                data = json.loads(text)
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "<no content>")
                )
                print("\n--- Test Chat Response (truncated) ---")
                print(content[:500])
                print("\n--------------------------------------")
            except Exception:
                print(text)


def _parse_indices(inp: str, max_idx: int) -> list[int]:
    parts = [p.strip() for p in inp.split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        if p.isdigit():
            i = int(p)
            if 0 <= i < max_idx:
                out.append(i)
    return sorted(set(out))


async def run_ui(namespace: str, http_url: str, model: str) -> None:
    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop, False)

    # Try backend first (decode workers)
    try:
        component = runtime.namespace(namespace).component("backend")
        generate_client = await component.endpoint("generate").client()
        await generate_client.wait_for_instances()
        is_sleeping_client = await component.endpoint("is_sleeping").client()
        sleep_client = await component.endpoint("sleep").client()
        wake_client = await component.endpoint("wake_up").client()
        print(f"Connected to {len(generate_client.instance_ids())} backend (decode) instances")
    except Exception as e:
        # Try prefill workers if backend not found
        print(f"No backend instances found ({e}), trying prefill...")
        component = runtime.namespace(namespace).component("prefill")
        generate_client = await component.endpoint("generate").client()
        await generate_client.wait_for_instances()
        is_sleeping_client = await component.endpoint("is_sleeping").client()
        sleep_client = await component.endpoint("sleep").client()
        wake_client = await component.endpoint("wake_up").client()
        print(f"Connected to {len(generate_client.instance_ids())} prefill instances")

    infos = await list_instances(generate_client, is_sleeping_client)

    while True:
        print_instances(infos)
        print(
            "\nActions: [r]efresh, [s]leep, [w]ake, [t]est chat, [q]uit"
        )
        choice = input("Select action: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            break
        if choice in ("r", "refresh"):
            infos = await list_instances(generate_client, is_sleeping_client)
            continue
        if choice in ("s", "sleep"):
            if not infos:
                print("No instances.")
                continue
            sel = input(
                "Enter indices to sleep (comma-separated), or 'all': "
            ).strip()
            if sel == "all":
                targets = [i.instance_id for i in infos]
            else:
                idxs = _parse_indices(sel, len(infos))
                targets = [infos[i].instance_id for i in idxs]
            if not targets:
                print("No valid selection.")
                continue
            print(f"Sleeping {len(targets)} instance(s) to level 3...")
            await call_sleep(sleep_client, targets, level=3)
            infos = await list_instances(generate_client, is_sleeping_client)
            continue
        if choice in ("w", "wake"):
            if not infos:
                print("No instances.")
                continue
            sel = input(
                "Enter indices to wake (comma-separated), or 'all': "
            ).strip()
            if sel == "all":
                targets = [i.instance_id for i in infos]
            else:
                idxs = _parse_indices(sel, len(infos))
                targets = [infos[i].instance_id for i in idxs]
            if not targets:
                print("No valid selection.")
                continue
            print(f"Waking {len(targets)} instance(s)...")
            await call_wake(wake_client, targets)
            infos = await list_instances(generate_client, is_sleeping_client)
            continue
        if choice in ("t", "test"):
            if not infos:
                print("No instances.")
                continue
            idxs = _parse_indices(input("Enter index to test: ").strip(), len(infos))
            if not idxs:
                print("No valid selection.")
                continue
            iid = infos[idxs[0]].instance_id
            await send_test_chat(http_url, model, iid)
            continue


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive TUI for Dynamo vLLM backends"
    )
    parser.add_argument(
        "--namespace", default="dynamo", help="Dynamo namespace (default: dynamo)"
    )
    parser.add_argument(
        "--http-url",
        default="http://localhost:8000",
        help="Frontend base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Model name to use for test chat requests",
    )

    args = parser.parse_args()

    try:
        asyncio.run(run_ui(args.namespace, args.http_url, args.model))
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()


