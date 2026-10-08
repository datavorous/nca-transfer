#!/bin/sh
# Run the frozen model-seed follow-up only if all three seed-0 corpus blocks pass.
# Invoke from the SampleEff repository root. Each probe resumes or skips itself.
set -eu

test -f pyproject.toml
uv run --all-packages python progress/nca_search/summarize_patch_replication.py
python3 - <<'PY'
import csv
from pathlib import Path

rows = list(csv.DictReader(Path('progress/nca_search/replication_1m.csv').open()))
by_block = {(r['block'], r['arm']): float(r['validation_loss'])
            for r in rows if int(r['seed']) == 0}
assert len(by_block) == 9, 'complete every block A/B/C seed-0 arm first'
assert all(by_block[b, 'high'] < by_block[b, 'unfiltered'] for b in 'abc'), \
    'high does not beat unfiltered in all three blocks'
assert sum(by_block[b, 'high'] < by_block[b, 'low'] for b in 'abc') >= 2, \
    'high does not beat low in at least two blocks'
print('three-block seed-0 gate passed')
PY

for block in a b c; do
    for seed in 1 2 3 4; do
        for arm in high unfiltered low; do
            root="runs/nca-data-search/patch-replication/block-${block}/${arm}/seeds/seed_${seed}"
            mkdir -p "$root"
            echo "RUN block=$block arm=$arm seed=$seed"
            uv run --all-packages python progress/nca_search/probe_low_change.py \
                --block "$block" --replication-arm "$arm" --seed "$seed" \
                > "$root/probe.log" 2>&1
            uv run --all-packages python progress/nca_search/summarize_patch_replication.py
        done
    done
done
