# OpenAI API Frontend

OpenAI-compatible HTTP API for seamless integration.

## Features

- Full OpenAI API compatibility
- Chat completions endpoint
- Streaming support
- Function calling support

## Configuration

Reference existing frontend configuration in deployment guides.

## Usage

Standard OpenAI client libraries work out of the box.

### Example Request

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ],
    "stream": false,
    "max_tokens": 100
  }'
```

### List Models

```bash
curl localhost:8000/v1/models
```

### Health Check

```bash
curl localhost:8000/v1/health
```

## Endpoints

- `/v1/chat/completions` - Chat completions endpoint
- `/v1/models` - List available models
- `/v1/health` - Health check endpoint