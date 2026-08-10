#!/bin/sh

# Wait until locust-master pod is available
until LOCUST_POD=$(kubectl get pod -n trainticket --no-headers | awk '{print $1}' | grep locust-master); do
  echo "Waiting for locust-master pod to be ready..."
  sleep 2
done

# Start locust load generation in the background
kubectl exec "$LOCUST_POD" -n trainticket -- locust --headless -f /app/load_generator.py &
