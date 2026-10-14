#!/usr/bin/env python3
"""Set shared-gateway cluster weights to ready endpoint counts.

Run on the VM alongside an experiment; the gateway itself carries only traffic.
The frozen rule is one unit of weight per ready service endpoint.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone


BACKENDS = {
    "auth": "ts-auth-service",
    "route": "ts-route-service",
    "travel": "ts-travel-service",
}
CLUSTERS = ("cluster2", "cluster3")


def run(*args: str, input_text: str | None = None) -> str:
    result = subprocess.run(
        args,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    if result.returncode:
        raise RuntimeError(f"{' '.join(args)}: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()


def ready_endpoint_count(payload: dict) -> int:
    """Count distinct ready Pod addresses, not desired or merely running replicas."""
    addresses: set[str] = set()
    for item in payload.get("items", []):
        for endpoint in item.get("endpoints", []):
            if endpoint.get("conditions", {}).get("ready") is not True:
                continue
            if endpoint.get("targetRef", {}).get("kind") != "Pod":
                continue
            addresses.update(endpoint.get("addresses", []))
    return len(addresses)


def ready_count(cluster: str, namespace: str, service: str) -> int:
    payload = json.loads(run(
        "kubectl", f"--context={cluster}", "-n", namespace,
        "get", "endpointslices.discovery.k8s.io",
        "-l", f"kubernetes.io/service-name={service}", "-o", "json",
    ))
    return ready_endpoint_count(payload)


def admin_command(namespace: str, command: str) -> str:
    return run(
        "kubectl", "--context=cluster1", "-n", namespace,
        "exec", "-i", "deployment/trainticket-gateway", "-c", "haproxy",
        "--", "nc", "-w", "2", "127.0.0.1", "9999",
        input_text=command + "\n",
    )


def apply_weights(namespace: str, weights: dict[str, dict[str, int]]) -> None:
    for backend in BACKENDS:
        for cluster in CLUSTERS:
            count = weights[backend][cluster]
            reply = admin_command(namespace, f"set weight {backend}/{cluster} {count}")
            if reply and re.search(r"(^|\n)(error|unknown|invalid)", reply, re.IGNORECASE):
                raise RuntimeError(f"HAProxy rejected {backend}/{cluster} weight {count}: {reply}")
            observed = admin_command(namespace, f"get weight {backend}/{cluster}")
            match = re.match(r"^(\d+)\b", observed)
            if not match or int(match.group(1)) != count:
                raise RuntimeError(
                    f"HAProxy weight verification failed for {backend}/{cluster}: expected {count}, got {observed!r}"
                )


def update(namespace: str) -> dict[str, dict[str, int]]:
    weights = {
        backend: {
            cluster: ready_count(cluster, namespace, service)
            for cluster in CLUSTERS
        }
        for backend, service in BACKENDS.items()
    }
    apply_weights(namespace, weights)
    return weights


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="update once and exit")
    parser.add_argument("--interval", type=float, default=10.0, help="watch interval in seconds")
    parser.add_argument("--namespace", default="trainticket")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    while True:
        weights = update(args.namespace)
        print(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ready_endpoint_weights": weights,
        }, sort_keys=True), flush=True)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.TimeoutExpired, ValueError) as exc:
        print(f"GATEWAY_WEIGHT_ERROR {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
