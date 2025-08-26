#!/bin/bash

# Check etcd (without access to etcdctl if outside container) for served model instances/endpoints
curl -L http://localhost:2379/v3/kv/range   -X POST   -d '{"key": "Lw==", "range_end": "AA=="}' | jq '.kvs[] | {key: (.key | @base64d), value: (.value | @base64d)}'
