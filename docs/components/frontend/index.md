# Frontend Components

Dynamo's frontend provides multiple API interfaces for inference requests.

## Available Interfaces

- [OpenAI API](openai.md) - OpenAI-compatible HTTP API
- [gRPC/KServe](grpc_kserve.md) - High-performance gRPC and KServe integration

## Architecture

The frontend acts as the entry point for all inference requests, handling:
- Request validation
- Authentication/authorization
- Request routing to appropriate backends
- Response formatting

## Choosing an Interface

- **OpenAI API**: Best for compatibility with existing OpenAI clients
- **gRPC/KServe**: Best for high-performance, low-latency requirements