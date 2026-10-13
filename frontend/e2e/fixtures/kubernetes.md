# Kubernetes container restarts

When a container fails, the kubelet restarts it according to the pod restartPolicy.
The policy can be Always, OnFailure or Never. Repeated failures trigger CrashLoopBackOff,
which increases the delay between restart attempts.
