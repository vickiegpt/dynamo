# Planner

The planner is a component that monitors the state of the system and makes adjustments to the number of workers to ensure that the system is running efficiently.

## Usage
After you've deployed a dynamo graph - you can start the planner with the following command:
```bash
python components/planner.py --namespace <namespace>
```

## Backends
We currently support two backends:
1. `local` - uses dynamo serve to start/stop a system
2. `kuberentes` - uses the kuberentes api to adjust replicas of each component's resource definition