# gRPC/KServe Frontend

High-performance gRPC and KServe Triton integration for Dynamo.

## Overview

The gRPC/KServe frontend provides high-performance, low-latency inference capabilities through:
- Native gRPC protocol support for reduced overhead
- KServe v2 inference protocol compatibility  
- Triton Inference Server integration
- Bidirectional streaming for real-time applications

## gRPC Interface

### Service Definition

The gRPC service implements the standard inference protocol with the following key methods:

```protobuf
service GRPCInferenceService {
  // Check server health and readiness
  rpc ServerLive(ServerLiveRequest) returns (ServerLiveResponse) {}
  rpc ServerReady(ServerReadyRequest) returns (ServerReadyResponse) {}
  
  // Model management
  rpc ModelReady(ModelReadyRequest) returns (ModelReadyResponse) {}
  rpc ModelMetadata(ModelMetadataRequest) returns (ModelMetadataResponse) {}
  
  // Inference
  rpc ModelInfer(ModelInferRequest) returns (ModelInferResponse) {}
  rpc ModelStreamInfer(stream ModelInferRequest) returns (stream ModelInferResponse) {}
}
```

### Authentication

gRPC authentication supports multiple methods:

**TLS/SSL:**
```bash
# Server with TLS
python -m dynamo.frontend --grpc-port 8001 --use-tls --cert-file server.crt --key-file server.key

# Client connection
channel = grpc.secure_channel('localhost:8001', grpc.ssl_channel_credentials())
```

**Token-based Authentication:**
```python
# Client with token
metadata = [('authorization', 'Bearer your-token-here')]
response = stub.ModelInfer(request, metadata=metadata)
```

### Streaming

Bidirectional streaming enables real-time inference:

```python
import grpc
from dynamo_grpc import inference_pb2_grpc, inference_pb2

def streaming_inference():
    channel = grpc.insecure_channel('localhost:8001')
    stub = inference_pb2_grpc.GRPCInferenceServiceStub(channel)
    
    # Create streaming request
    def request_generator():
        for i in range(10):
            request = inference_pb2.ModelInferRequest()
            request.model_name = "your-model"
            # Configure request...
            yield request
    
    # Stream inference
    responses = stub.ModelStreamInfer(request_generator())
    for response in responses:
        print(f"Response: {response}")
```

## KServe Integration

### KServe v2 Protocol

Dynamo implements the KServe v2 inference protocol for compatibility with existing KServe deployments:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: dynamo-llm
spec:
  predictor:
    containers:
    - name: dynamo
      image: nvcr.io/nvidia/dynamo:latest-vllm
      args:
        - python -m dynamo.frontend --grpc-port 8001 --kserve-mode
      ports:
      - containerPort: 8001
        protocol: TCP
```

### Triton Inference Server

Integration with Triton enables advanced model management:

**Model Repository Structure:**
```
model_repository/
├── llm_model/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py
```

**Configuration Example:**
```python
# config.pbtxt
name: "llm_model"
backend: "python"
max_batch_size: 8

input [
  {
    name: "text_input"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }
]

output [
  {
    name: "text_output"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }
]
```

### Model Management

Dynamic model loading and unloading:

```python
# Load model
load_request = inference_pb2.RepositoryModelLoadRequest()
load_request.model_name = "new-model"
load_request.parameters["model_path"] = "s3://bucket/model"
stub.RepositoryModelLoad(load_request)

# Unload model  
unload_request = inference_pb2.RepositoryModelUnloadRequest()
unload_request.model_name = "old-model"
stub.RepositoryModelUnload(unload_request)
```

## Performance Optimization

### gRPC Configuration

Optimize gRPC performance with these settings:

```python
# Server options
options = [
    ('grpc.keepalive_time_ms', 30000),
    ('grpc.keepalive_timeout_ms', 5000),
    ('grpc.keepalive_permit_without_calls', True),
    ('grpc.http2.max_pings_without_data', 0),
    ('grpc.http2.min_time_between_pings_ms', 10000),
    ('grpc.http2.min_ping_interval_without_data_ms', 300000)
]

server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
```

### Connection Pooling

Use connection pooling for high-throughput scenarios:

```python
class GRPCConnectionPool:
    def __init__(self, target, pool_size=10):
        self.channels = [
            grpc.insecure_channel(target) 
            for _ in range(pool_size)
        ]
        self.current = 0
    
    def get_stub(self):
        channel = self.channels[self.current % len(self.channels)]
        self.current += 1
        return inference_pb2_grpc.GRPCInferenceServiceStub(channel)
```

## Client Examples

### Python Client

```python
import grpc
from dynamo_grpc import inference_pb2_grpc, inference_pb2

def create_inference_request(model_name, input_text):
    request = inference_pb2.ModelInferRequest()
    request.model_name = model_name
    
    # Configure input
    input_tensor = inference_pb2.ModelInferRequest.InferInputTensor()
    input_tensor.name = "text_input"
    input_tensor.datatype = "BYTES"
    input_tensor.shape.extend([1])
    
    # Set input data
    input_data = input_text.encode('utf-8')
    request.raw_input_contents.append(input_data)
    request.inputs.extend([input_tensor])
    
    return request

def main():
    channel = grpc.insecure_channel('localhost:8001')
    stub = inference_pb2_grpc.GRPCInferenceServiceStub(channel)
    
    # Check server health
    health_request = inference_pb2.ServerLiveRequest()
    health_response = stub.ServerLive(health_request)
    print(f"Server live: {health_response.live}")
    
    # Perform inference
    request = create_inference_request("llm-model", "Hello, world!")
    response = stub.ModelInfer(request)
    
    # Process response
    output_data = response.raw_output_contents[0].decode('utf-8')
    print(f"Response: {output_data}")
```

### Go Client

```go
package main

import (
    "context"
    "log"
    
    "google.golang.org/grpc"
    pb "path/to/inference_pb"
)

func main() {
    conn, err := grpc.Dial("localhost:8001", grpc.WithInsecure())
    if err != nil {
        log.Fatal(err)
    }
    defer conn.Close()
    
    client := pb.NewGRPCInferenceServiceClient(conn)
    
    // Create request
    req := &pb.ModelInferRequest{
        ModelName: "llm-model",
        Inputs: []*pb.ModelInferRequest_InferInputTensor{
            {
                Name:     "text_input",
                Datatype: "BYTES",
                Shape:    []int64{1},
            },
        },
        RawInputContents: [][]byte{[]byte("Hello, world!")},
    }
    
    // Perform inference
    resp, err := client.ModelInfer(context.Background(), req)
    if err != nil {
        log.Fatal(err)
    }
    
    log.Printf("Response: %s", string(resp.RawOutputContents[0]))
}
```

## Monitoring

### gRPC Metrics

gRPC metrics are automatically exposed via Prometheus:

```yaml
# Prometheus scrape config
- job_name: 'dynamo-grpc'
  static_configs:
    - targets: ['localhost:8001']
  metrics_path: '/metrics'
```

Key metrics include:
- `grpc_server_handled_total` - Total handled requests
- `grpc_server_handling_seconds` - Request duration
- `grpc_server_msg_received_total` - Messages received
- `grpc_server_msg_sent_total` - Messages sent

### Tracing

Distributed tracing with OpenTelemetry:

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)
span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
```

## Configuration Reference

### Command Line Options

```bash
python -m dynamo.frontend \
  --grpc-port 8001 \
  --grpc-workers 4 \
  --use-tls \
  --cert-file server.crt \
  --key-file server.key \
  --kserve-mode \
  --max-message-size 4194304
```

### Environment Variables

- `DYNAMO_GRPC_PORT` - gRPC server port (default: 8001)
- `DYNAMO_GRPC_WORKERS` - Number of gRPC worker threads
- `DYNAMO_TLS_CERT` - Path to TLS certificate file
- `DYNAMO_TLS_KEY` - Path to TLS private key file
- `DYNAMO_MAX_MESSAGE_SIZE` - Maximum gRPC message size

## Troubleshooting

### Common Issues

**Connection Refused:**
```bash
# Check if service is running
netstat -tlnp | grep 8001
# Check firewall
sudo ufw status
```

**TLS Certificate Errors:**
```bash
# Verify certificate
openssl x509 -in server.crt -text -noout
# Check key match
openssl x509 -noout -modulus -in server.crt | openssl md5
openssl rsa -noout -modulus -in server.key | openssl md5
```

**Performance Issues:**
- Increase `grpc-workers` for CPU-bound workloads
- Tune connection pooling for high-throughput scenarios
- Monitor gRPC metrics for bottlenecks