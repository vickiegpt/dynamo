# Health Check Guide

## Quick Health Checks

### Component Status
```bash
kubectl get pods -n dynamo
kubectl describe pod <pod-name> -n dynamo
```

### Service Health
```bash
curl http://localhost:8000/health
```

## Automated Health Monitoring

Reference Kubernetes liveness and readiness probes.