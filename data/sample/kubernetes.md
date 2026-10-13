# Kubernetes Pod Lifecycle

## Restart Policy

Kubernetes restarts a failed container according to the Pod's restartPolicy.
The kubelet on the node observes container exits and, depending on the
configured restartPolicy (Always, OnFailure, Never), restarts the
container with an exponential backoff delay.

## Readiness vs Liveness Probes

Readiness probes determine whether a container is ready to accept traffic.
Liveness probes determine whether a container should be restarted. A
failing liveness probe triggers a container restart, while a failing
readiness probe only removes the Pod from Service endpoints.
