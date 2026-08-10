#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
shutdown_once=0
child_pid=""
shutdown_enabled="${HEADLIGHT_SHUTDOWN_ON_EXIT:-0}"

for arg in "$@"; do
  if [[ "$arg" = "--dry-run" ]] || [[ "$arg" = "--smoke" ]]; then
    shutdown_enabled=0
  fi
done

shutdown_vm() {
  local original_exit="$1"
  if [[ "$shutdown_once" = "1" ]] || [[ "$shutdown_enabled" != "1" ]]; then
    return 0
  fi
  shutdown_once=1
  sync
  if [[ -n "${HEADLIGHT_SHUTDOWN_COMMAND:-}" ]]; then
    # Test-only override; no shell evaluation is performed.
    read -r -a shutdown_command <<< "$HEADLIGHT_SHUTDOWN_COMMAND"
  else
    shutdown_command=(systemctl poweroff)
  fi
  if ! "${shutdown_command[@]}"; then
    echo "primary poweroff failed after campaign exit $original_exit; trying fallback" >&2
    if [[ -n "${HEADLIGHT_SHUTDOWN_FALLBACK_COMMAND:-}" ]]; then
      read -r -a fallback_command <<< "$HEADLIGHT_SHUTDOWN_FALLBACK_COMMAND"
    else
      fallback_command=(/usr/sbin/shutdown -h now)
    fi
    "${fallback_command[@]}" || echo "fallback shutdown also failed" >&2
  fi
  return 0
}

on_exit() {
  local original_exit="$?"
  trap - EXIT INT TERM
  shutdown_vm "$original_exit"
  exit "$original_exit"
}

on_signal() {
  local signal_name="$1"
  if [[ -n "$child_pid" ]]; then
    kill -s "$signal_name" "$child_pid" >/dev/null 2>&1 || true
    wait "$child_pid" || true
  fi
  if [[ "$signal_name" = "INT" ]]; then
    exit 130
  fi
  exit 143
}

trap on_exit EXIT
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

# The shell is the shutdown owner for the documented production entry point.
HEADLIGHT_SHUTDOWN_ON_EXIT=0 python3 "$repo_root/experiments/run_capacity_qualification.py" "$@" &
child_pid="$!"
wait "$child_pid"
