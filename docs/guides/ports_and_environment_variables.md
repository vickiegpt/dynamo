# Ports and Environment Variables Reference

This page provides a comprehensive reference for all ports and environment variables used by Dynamo components.

## Default Ports

### Frontend Services
| Service | Default Port | Purpose | Configuration |
|---------|--------------|---------|---------------|
| **HTTP Frontend** | 8000 | OpenAI-compatible API server | `--http-port 8000` |
| **Health Check** | 8000 | Health endpoint (`/v1/health`) | Same as HTTP port |
| **Metrics** | 8000 | Prometheus metrics (`/metrics`) | Same as HTTP port |

### Backend Services
| Service | Default Port | Purpose | Configuration |
|---------|--------------|---------|---------------|
| **Backend Metrics** | 8081 | Worker metrics endpoint | `DYN_SYSTEM_PORT=8081` |
| **KV Metrics** | 9090 | KV cache metrics publishing | `--metrics-endpoint-port 9090` |

### Infrastructure Services
| Service | Default Port | Purpose | Configuration |
|---------|--------------|---------|---------------|
| **ETCD** | 2379, 2380 | Service discovery and metadata | Standard etcd ports |
| **NATS** | 4222, 6222, 8222 | Message broker and streaming | Standard NATS ports |
| **Prometheus** | 9090 | Metrics collection server | Standard Prometheus port |
| **Grafana** | 3001 | Metrics visualization | `docker-compose.yml` |
| **DCGM Exporter** | 9401 | GPU metrics | Custom port to avoid conflicts |
| **NATS Prometheus Exporter** | 7777 | NATS metrics for Prometheus | Custom port |

### SGLang-Specific Services
| Service | Default Port | Purpose | Configuration |
|---------|--------------|---------|---------------|
| **SGLang HTTP Server** | 9001 | Native SGLang endpoints | `--port 9001` |
| **SGLang Worker** | 8000 | Worker HTTP interface | `--host 0.0.0.0 --port 8000` |

## Environment Variables

### Core Runtime Variables
| Variable | Default | Purpose | Example |
|----------|---------|---------|---------|
| `DYN_LOG` | `info` | Logging level | `DYN_LOG=debug` |
| `DYN_SYSTEM_ENABLED` | `false` | Enable system metrics | `DYN_SYSTEM_ENABLED=true` |
| `DYN_SYSTEM_PORT` | `8081` | System metrics port | `DYN_SYSTEM_PORT=8081` |
| `DYN_TOKEN_ECHO_DELAY_MS` | `10` | Echo engine token delay | `DYN_TOKEN_ECHO_DELAY_MS=1` |

### Deployment Variables
| Variable | Purpose | Example |
|----------|---------|---------|
| `NAMESPACE` | Kubernetes namespace | `NAMESPACE=<your-dynamo-namespace>` |
| `IMAGE_TAG` | Docker image tag | `IMAGE_TAG=0.4.0` |
| `RELEASE_VERSION` | Helm chart version | `RELEASE_VERSION=0.4.0` |

### Model and Authentication Variables
| Variable | Purpose | Example |
|----------|---------|---------|
| `HF_TOKEN` | HuggingFace API token | `HF_TOKEN=hf_your_token_here` |
| `MODEL_PATH` | Path to model files | `MODEL_PATH=Qwen/Qwen3-0.6B` |
| `SERVED_MODEL_NAME` | Model name for API | `SERVED_MODEL_NAME=Qwen/Qwen3-0.6B` |
| `DYNAMO_IMAGE` | Base runtime image | `DYNAMO_IMAGE=nvcr.io/nvidia/ai-dynamo/<select-your-runtime>` |

### Backend-Specific Variables
| Variable | Backend | Purpose | Example |
|----------|---------|---------|---------|
| `DISAGGREGATION_STRATEGY` | TRT-LLM, SGLang | Request flow strategy | `DISAGGREGATION_STRATEGY=decode_first` |
| `PREFILL_ENGINE_ARGS` | TRT-LLM | Prefill worker config | `PREFILL_ENGINE_ARGS=engine_configs/prefill.yaml` |
| `DECODE_ENGINE_ARGS` | TRT-LLM | Decode worker config | `DECODE_ENGINE_ARGS=engine_configs/decode.yaml` |
| `AGG_ENGINE_ARGS` | TRT-LLM | Aggregated worker config | `AGG_ENGINE_ARGS=./engine_configs/agg.yaml` |
| `MODALITY` | TRT-LLM | Model modality | `MODALITY=multimodal` |

### GPU and Hardware Variables
| Variable | Purpose | Example |
|----------|---------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU selection | `CUDA_VISIBLE_DEVICES=0,2` |
| `NVIDIA_VISIBLE_DEVICES` | Container GPU access | `NVIDIA_VISIBLE_DEVICES=all` |

## Port Configuration Examples

### Frontend with Custom Port
```bash
python -m dynamo.frontend --http-port 8080 --router-mode kv
```

### Port Forwarding for Kubernetes
```bash
# Forward frontend service
kubectl port-forward svc/frontend-service 8000:8000 -n $NAMESPACE

# Forward multiple services
kubectl port-forward svc/prometheus 9090:9090 -n monitoring &
kubectl port-forward svc/grafana 3001:3001 -n monitoring &
```

### Docker Compose Port Mapping
```yaml
services:
  frontend:
    ports:
      - "8000:8000"  # HTTP API
  prometheus:
    ports:
      - "9090:9090"  # Metrics collection
  grafana:
    ports:
      - "3001:3001"  # Dashboard
```

## Environment Variable Usage

### Setting Variables for Local Development
```bash
export DYN_LOG=debug
export NAMESPACE=dynamo-dev
export HF_TOKEN=your_token_here

python -m dynamo.frontend --http-port 8000
```

### Kubernetes Secret Creation
```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}

kubectl create secret docker-registry docker-imagepullsecret \
  --docker-server=${DOCKER_SERVER} \
  --docker-username=${DOCKER_USERNAME} \
  --docker-password=${DOCKER_PASSWORD} \
  --namespace=${NAMESPACE}
```

### Docker Compose Environment
```yaml
services:
  dynamo-frontend:
    environment:
      - DYN_LOG=info
      - DYN_SYSTEM_ENABLED=true
      - DYN_SYSTEM_PORT=8081
    ports:
      - "8000:8000"
```

## Troubleshooting Port Conflicts

### Check Port Usage
```bash
# Check if port is in use
netstat -tulpn | grep :8000
lsof -i :8000

# Find available ports
ss -tulpn | grep :80
```

### Common Port Conflicts
| Port | Common Conflict | Solution |
|------|----------------|----------|
| 8000 | Development servers | Use `--http-port 8001` |
| 8080 | Jenkins, Tomcat | Use `--http-port 8000` (default) |
| 9090 | Other Prometheus | Use custom Prometheus port |
| 3000 | Grafana default | Use port 3001 (Dynamo default) |

### Firewall Configuration
```bash
# Ubuntu/Debian
sudo ufw allow 8000/tcp
sudo ufw allow 9090/tcp

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload
```

## Related Documentation

- [Installation Guide](dynamo_deploy/dynamo_cloud.md) - Platform installation and configuration
- [Metrics Guide](metrics.md) - Metrics collection and monitoring
- [Backend Configuration](backend.md) - Backend-specific settings
