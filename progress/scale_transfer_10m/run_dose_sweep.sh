#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/../.."
exec /usr/bin/caffeinate -dimsu uv run --locked --all-packages python progress/scale_transfer_10m/dose_sweep.py --run
