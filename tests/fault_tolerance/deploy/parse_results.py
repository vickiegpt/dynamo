# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parser for AI-Perf results in fault tolerance tests."""

import argparse
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tabulate import tabulate


def parse_test_log(
    file_path: str,
) -> Tuple[Optional[float], Optional[List[str]]]:
    """
    Parse test log for startup time and failure info.

    Args:
        file_path: Path to test.log.txt

    Returns:
        Tuple of (startup_time_seconds, failure_info)
    """
    start_time = None
    ready_time = None
    failure_info: Optional[List[str]] = None

    if not os.path.isfile(file_path):
        return None, None

    with open(file_path, "r") as f:
        for line in f:
            # Extract timestamp using regex to handle different log formats
            timestamp_match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", line)

            # Look for deployment start
            if "Starting Deployment" in line and timestamp_match:
                timestamp = timestamp_match.group(1)
                start_time = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")

            # Look for deployment ready
            if "Deployment fault-tolerance-test is ready" in line and timestamp_match:
                timestamp = timestamp_match.group(1)
                ready_time = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")

            # Look for fault injection
            if "Injecting failure for:" in line:
                # Extract failure details
                match = re.search(r"Failure\((.*?)\)", line)
                if match:
                    failure_str = match.group(1)
                    parts = failure_str.split(", ")
                    failure_dict = {}
                    for part in parts:
                        key_val = part.split("=")
                        if len(key_val) == 2:
                            failure_dict[key_val[0]] = key_val[1]

                    # Build command list from failure info
                    if failure_dict:
                        failure_info = [
                            failure_dict.get("pod_name", "unknown").strip("'\""),
                            failure_dict.get("command", "unknown").strip("'\""),
                        ]

    # Calculate startup time in seconds
    startup_time = None
    if start_time and ready_time:
        startup_time = (ready_time - start_time).total_seconds()

    return startup_time, failure_info


def calculate_recovery_time(
    failure_info: Optional[List[str]],
    process_logs_dir: str,
) -> Optional[float]:
    """
    Calculate recovery time by comparing last timestamp in .previous.log with first in current log.
    This avoids timezone issues between test.log.txt and container logs.

    Args:
        failure_info: List with [pod_name, command] from fault injection
        process_logs_dir: Directory containing process log files

    Returns:
        Recovery time in seconds or None if not found
    """
    if not failure_info:
        return None

    # Determine component type from failure info (strip any quotes)
    component_type = failure_info[0].strip("'\"")  # e.g., "Frontend" or "decode"
    component_dir = os.path.join(process_logs_dir, component_type)

    if not os.path.exists(component_dir):
        return None

    last_timestamp_before = None
    first_timestamp_after = None

    # Find the last timestamp from .previous.log (container before restart)
    for log_file in os.listdir(component_dir):
        if log_file.endswith(".previous.log"):
            log_path = os.path.join(component_dir, log_file)
            try:
                with open(log_path, "r") as f:
                    # Read last few lines to find last valid timestamp
                    lines = f.readlines()
                    for line in reversed(lines[-10:]):  # Check last 10 lines
                        if '"time":"' in line:
                            try:
                                log_entry = json.loads(line)
                                timestamp_str = log_entry.get("time", "")[:19]
                                last_timestamp_before = datetime.strptime(
                                    timestamp_str, "%Y-%m-%dT%H:%M:%S"
                                )
                                break
                            except (json.JSONDecodeError, ValueError):
                                continue
            except IOError as e:
                logging.debug(f"Could not read {log_file}: {e}")

    # Find the first timestamp from current container log
    for log_file in os.listdir(component_dir):
        if log_file.endswith(".log") and not log_file.endswith(
            (".previous.log", ".metrics.log")
        ):
            log_path = os.path.join(component_dir, log_file)
            try:
                with open(log_path, "r") as f:
                    first_line = f.readline()
                    if first_line and '"time":"' in first_line:
                        log_entry = json.loads(first_line)
                        timestamp_str = log_entry.get("time", "")[:19]
                        first_timestamp_after = datetime.strptime(
                            timestamp_str, "%Y-%m-%dT%H:%M:%S"
                        )
            except (json.JSONDecodeError, ValueError, IOError) as e:
                logging.debug(f"Could not parse timestamp from {log_file}: {e}")

    # Calculate recovery time from container timestamps (both in UTC)
    if last_timestamp_before and first_timestamp_after:
        recovery_time = (first_timestamp_after - last_timestamp_before).total_seconds()
        # Sanity check - recovery should be seconds/minutes, not hours
        if recovery_time > 3600:  # More than 1 hour is likely wrong
            logging.warning(
                f"Recovery time {recovery_time}s seems too large, possible timezone issue"
            )
        return recovery_time

    return None


def parse_aiperf_client_results(log_dir: str) -> Dict[str, Any]:
    """
    Parse AI-Perf results from all client directories.

    Args:
        log_dir: Directory containing client result directories

    Returns:
        Dictionary with aggregated metrics and client count
    """
    all_metrics: Dict[str, Any] = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "latencies": [],
        "ttft": [],  # Time to First Token
        "itl": [],  # Inter-Token Latency
        "throughputs": [],
        "p50_latencies": [],
        "p90_latencies": [],
        "p99_latencies": [],
        "num_clients": 0,
    }

    # Iterate over actual client directories
    for item in sorted(os.listdir(log_dir)):
        if not item.startswith("client_") or not os.path.isdir(
            os.path.join(log_dir, item)
        ):
            continue

        client_dir = Path(log_dir) / item
        all_metrics["num_clients"] += 1

        # Look for AI-Perf results in attempt directories
        profile_json = None

        # Check for attempt directories (attempt_0, attempt_1, etc.)
        for attempt_dir in sorted(client_dir.glob("attempt_*")):
            json_path = attempt_dir / "profile_export_aiperf.json"
            if json_path.exists():
                profile_json = json_path
                break  # Use the first successful attempt

        if not profile_json:
            logging.warning(f"No AI-Perf results found for {item} in {client_dir}")
        else:
            try:
                with open(profile_json) as f:
                    client_metrics = json.load(f)

                # AI-Perf format has "records" dictionary at the top level
                records = client_metrics.get("records", {})

                # Extract request count (this is the total requests made)
                request_count_record = records.get("request_count", {})
                request_count = (
                    int(request_count_record.get("avg", 0))
                    if request_count_record
                    else 0
                )

                # Check for errors in error_summary
                error_summary = client_metrics.get("error_summary", [])
                error_count = len(error_summary)

                # Check if test was cancelled
                was_cancelled = client_metrics.get("was_cancelled", False)
                if was_cancelled:
                    error_count = request_count  # Mark all as failed if cancelled

                all_metrics["total_requests"] += request_count
                all_metrics["successful_requests"] += request_count - error_count
                all_metrics["failed_requests"] += error_count

                # Extract latency from request_latency record
                request_latency = records.get("request_latency", {})

                if request_latency:
                    # Convert milliseconds to seconds for consistency
                    if "avg" in request_latency:
                        all_metrics["latencies"].append(request_latency["avg"] / 1000.0)
                    if "p50" in request_latency:
                        all_metrics["p50_latencies"].append(
                            request_latency["p50"] / 1000.0
                        )
                    if "p90" in request_latency:
                        all_metrics["p90_latencies"].append(
                            request_latency["p90"] / 1000.0
                        )
                    if "p99" in request_latency:
                        all_metrics["p99_latencies"].append(
                            request_latency["p99"] / 1000.0
                        )

                # Time to first token (if available in records)
                ttft = records.get("time_to_first_token", {}) or records.get("ttft", {})
                if ttft and "avg" in ttft:
                    all_metrics["ttft"].append(ttft["avg"] / 1000.0)  # Convert ms to s

                # Inter-token latency (if available in records)
                itl = records.get("inter_token_latency", {}) or records.get("itl", {})
                if itl and "avg" in itl:
                    all_metrics["itl"].append(itl["avg"] / 1000.0)  # Convert ms to s

                # Throughput from request_throughput record
                request_throughput = records.get("request_throughput", {})
                req_throughput = request_throughput.get("avg", 0)
                if req_throughput:
                    all_metrics["throughputs"].append(req_throughput)

            except Exception as e:
                logging.error(f"Error parsing {item} results: {e}")

    return all_metrics


def print_summary_table(
    log_dir: str,
    num_clients: int,
    startup_time: Optional[float],
    recovery_time: Optional[float],
    metrics: Dict[str, Any],
    tablefmt: str = "grid",
    sla: Optional[float] = None,
) -> None:
    """
    Print formatted summary table with AI-Perf metrics.

    Args:
        log_dir: Test directory path
        num_clients: Number of client processes
        startup_time: Time to start deployment (seconds)
        recovery_time: Time to recover from fault (seconds)
        metrics: Aggregated metrics from AI-Perf
        tablefmt: Table format for output
        sla: Service level agreement for latency (optional)
    """
    headers = ["Metric", "Value"]
    rows = []

    # Test info
    rows.append(["Test Directory", log_dir])
    rows.append(["Number of Clients", str(num_clients)])
    rows.append(["", ""])

    # Deployment metrics
    rows.append(["=== Deployment Metrics ===", ""])
    if startup_time:
        rows.append(["Startup Time", f"{startup_time:.2f} sec"])
    else:
        rows.append(["Startup Time", "N/A"])

    if recovery_time:
        rows.append(["Recovery Time", f"{recovery_time:.2f} sec"])
    else:
        rows.append(["Recovery Time", "N/A"])
    rows.append(["", ""])

    # Request metrics
    rows.append(["=== Request Metrics ===", ""])
    rows.append(["Total Requests", metrics["total_requests"]])
    rows.append(["Successful Requests", metrics["successful_requests"]])
    rows.append(["Failed Requests", metrics["failed_requests"]])

    if metrics["total_requests"] > 0:
        success_rate = (
            metrics["successful_requests"] / metrics["total_requests"]
        ) * 100
        rows.append(["Success Rate", f"{success_rate:.2f}%"])
    rows.append(["", ""])

    # Latency metrics
    rows.append(["=== Latency Metrics (seconds) ===", ""])

    if metrics["latencies"]:
        mean_latency = np.mean(metrics["latencies"])
        rows.append(["Mean Latency", f"{mean_latency:.3f}"])

        # Check SLA if provided
        if sla is not None:
            sla_status = "✓ PASS" if mean_latency <= sla else "✗ FAIL"
            rows.append(["SLA Status", f"{sla_status} (target: {sla:.3f}s)"])

    if metrics["p50_latencies"]:
        rows.append(["P50 Latency", f"{np.mean(metrics['p50_latencies']):.3f}"])

    if metrics["p90_latencies"]:
        rows.append(["P90 Latency", f"{np.mean(metrics['p90_latencies']):.3f}"])

    if metrics["p99_latencies"]:
        rows.append(["P99 Latency", f"{np.mean(metrics['p99_latencies']):.3f}"])
    rows.append(["", ""])

    # Token generation metrics
    rows.append(["=== Token Generation Metrics ===", ""])

    if metrics["ttft"]:
        rows.append(
            ["Time to First Token (mean)", f"{np.mean(metrics['ttft']):.3f} sec"]
        )

    if metrics["itl"]:
        rows.append(
            ["Inter-Token Latency (mean)", f"{np.mean(metrics['itl']):.4f} sec"]
        )
    rows.append(["", ""])

    # Throughput metrics
    rows.append(["=== Throughput Metrics ===", ""])

    if metrics["throughputs"]:
        total_throughput = sum(metrics["throughputs"])
        rows.append(["Total Throughput", f"{total_throughput:.2f} req/s"])
        rows.append(
            ["Avg Client Throughput", f"{np.mean(metrics['throughputs']):.2f} req/s"]
        )

    # Print table
    print("\n" + "=" * 60)
    print("FAULT TOLERANCE TEST SUMMARY - AI-PERF")
    print("=" * 60)
    print(tabulate(rows, headers=headers, tablefmt=tablefmt))
    print("=" * 60 + "\n")


def process_single_test(
    log_dir: str, tablefmt: str = "grid", sla: Optional[float] = None
) -> Dict[str, Any]:
    """
    Process a single test log directory.

    Args:
        log_dir: Directory containing test results
        tablefmt: Table format for output
        sla: Service level agreement for latency (optional)

    Returns:
        Dictionary with test results
    """
    # Parse test configuration
    test_log = os.path.join(log_dir, "test.log.txt")
    startup_time, failure_info = parse_test_log(test_log)

    # Calculate recovery time only if fault was injected
    recovery_time = None
    if failure_info:
        recovery_time = calculate_recovery_time(failure_info, log_dir)

    # Parse AI-Perf results (also counts clients)
    metrics = parse_aiperf_client_results(log_dir)

    # Extract client count from metrics
    num_clients = metrics.get("num_clients", 0)

    # Print summary
    print_summary_table(
        log_dir, num_clients, startup_time, recovery_time, metrics, tablefmt, sla
    )

    return {
        "log_dir": log_dir,
        "num_clients": num_clients,
        "startup_time": startup_time,
        "recovery_time": recovery_time,
        "metrics": metrics,
    }


def main(
    logs_dir: Optional[str] = None,
    log_paths: Optional[List[str]] = None,
    tablefmt: str = "grid",
    sla: Optional[float] = None,
):
    """
    Main parser entry point with support for multiple log paths.

    Args:
        logs_dir: Base directory for logs (optional)
        log_paths: List of log directories to process
        tablefmt: Table format for output
        sla: Service level agreement for latency (optional)

    Returns:
        Combined results from all processed tests
    """
    # Handle different input formats
    if log_paths:
        # Process multiple log paths
        all_results = []
        for log_path in log_paths:
            if logs_dir:
                full_path = os.path.join(logs_dir, log_path)
            else:
                full_path = log_path

            if os.path.isdir(full_path):
                print(f"\nProcessing: {full_path}")
                results = process_single_test(full_path, tablefmt, sla)
                all_results.append(results)
            else:
                print(f"Warning: {full_path} is not a valid directory, skipping...")

        # If multiple tests, also print combined summary
        if len(all_results) > 1:
            print("\n" + "=" * 60)
            print("COMBINED TEST SUMMARY")
            print("=" * 60)

            total_requests = sum(r["metrics"]["total_requests"] for r in all_results)
            total_successful = sum(
                r["metrics"]["successful_requests"] for r in all_results
            )
            total_failed = sum(r["metrics"]["failed_requests"] for r in all_results)

            print(f"Total Tests: {len(all_results)}")
            print(f"Total Requests: {total_requests}")
            print(f"Total Successful: {total_successful}")
            print(f"Total Failed: {total_failed}")

            if total_requests > 0:
                print(
                    f"Overall Success Rate: {(total_successful/total_requests)*100:.2f}%"
                )

            print("=" * 60 + "\n")

        return all_results

    elif logs_dir:
        # Process single directory
        return process_single_test(logs_dir, tablefmt, sla)
    else:
        print("Error: Must provide either logs_dir or log_paths")
        return None


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Parse fault tolerance test results")
    parser.add_argument(
        "log_dir", type=str, help="Directory containing test logs and results"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        logging.error(f"{args.log_dir} is not a valid directory")
        exit(1)

    main(args.log_dir)
