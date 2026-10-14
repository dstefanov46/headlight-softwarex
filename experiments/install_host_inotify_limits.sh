#!/usr/bin/env bash
set -euo pipefail

# Persist the inotify capacity required by the four-node local Kind/Karmada
# testbed. Run this script as root; it is intentionally separate from the
# unprivileged testbed scripts so a failed privilege escalation cannot leave a
# partially rebuilt cluster behind.
target="/etc/sysctl.d/99-headlight-kind.conf"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "run as root: sudo $0" >&2
  exit 1
fi

temporary="$(mktemp)"
trap 'rm -f "$temporary"' EXIT
cat > "$temporary" <<'EOF'
# HeadLight local Kind/Karmada testbed
fs.inotify.max_user_instances = 8192
fs.inotify.max_user_watches = 1048576
EOF

if ! cmp -s "$temporary" "$target" 2>/dev/null; then
  install -o root -g root -m 0644 "$temporary" "$target"
fi

sysctl --load "$target"

instances="$(cat /proc/sys/fs/inotify/max_user_instances)"
watches="$(cat /proc/sys/fs/inotify/max_user_watches)"
if [[ "$instances" -lt 8192 || "$watches" -lt 1048576 ]]; then
  echo "failed to establish required inotify limits: instances=$instances watches=$watches" >&2
  exit 1
fi

echo "persistent inotify limits installed at $target"
echo "fs.inotify.max_user_instances=$instances"
echo "fs.inotify.max_user_watches=$watches"
