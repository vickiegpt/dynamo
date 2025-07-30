#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Fetch benchmark results from Kubernetes PVC and plot locally

set -e

NAMESPACE=${NAMESPACE:-"hzhou-dynamo"}
PVC_NAME="profiling-pvc"
TEMP_POD_NAME="fetch-results-$(date +%s)"

echo "Fetching benchmark results and generating plots"

cleanup() {
    kubectl delete pod $TEMP_POD_NAME -n $NAMESPACE --ignore-not-found=true >/dev/null 2>&1
}
trap cleanup EXIT

echo "Creating temporary pod..."
kubectl run $TEMP_POD_NAME -n $NAMESPACE --image=busybox --restart=Never --rm -i --tty=false \
  --overrides='{"spec":{"containers":[{"name":"fetch","image":"busybox","command":["sleep","300"],"volumeMounts":[{"name":"data","mountPath":"/data"}]}],"volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"'$PVC_NAME'"}}]}}' \
  >/dev/null &

echo "Waiting for pod to be ready..."
kubectl wait --for=condition=Ready pod/$TEMP_POD_NAME -n $NAMESPACE --timeout=60s >/dev/null

echo "Downloading results..."
mkdir -p results/kv_two_routers

kubectl cp $NAMESPACE/$TEMP_POD_NAME:/data/results_summary.json results/kv_two_routers/results_summary.json 2>/dev/null || \
kubectl cp $NAMESPACE/$TEMP_POD_NAME:/data/multi_router_results/results_summary.json results/kv_two_routers/results_summary.json 2>/dev/null || \
echo "Could not find results_summary.json"

if [ -f "results/kv_two_routers/results_summary.json" ]; then
    echo "Found results file"
else
    echo "No results file found. Available files in PVC:"
    kubectl exec $TEMP_POD_NAME -n $NAMESPACE -- find /data -name "*.json" | head -10
    exit 1
fi

echo "Generating plots..."
python benchmarks/multi_router/plot_prefix_ratio_comparison.py

echo "Done. Check plots.png and plots.pdf"
