#!/bin/sh
# Invoke from SampleEff root only after the active 1M runner finishes.
# The gate exits nonzero before any continuation if the 45-run screen fails.
set -eu

test -f pyproject.toml
uv run --all-packages python progress/nca_search/gate_patch_replication_10m.py
for block in a b c; do
    for seed in 0 1 2 3 4; do
        for arm in high unfiltered low; do
            root="runs/nca-data-search/patch-replication/block-${block}/${arm}/seeds/seed_${seed}"
            echo "10M block=$block arm=$arm seed=$seed"
            uv run --all-packages python progress/nca_search/continue_patch_replication_10m.py \
                --block "$block" --arm "$arm" --seed "$seed" \
                >> "$root/continue_10m.log" 2>&1
        done
    done
done
