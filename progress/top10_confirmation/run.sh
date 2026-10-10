#!/bin/sh
# Frozen end-to-end queue. No outcome-based stopping or retuning.
set -eu
cd "$(dirname "$0")/../.."
root=runs/nca-data-search/top10-confirmation
mkdir -p "$root"
for block in a b c; do
    printf 'SAMPLE block=%s\n' "$block"
    mkdir -p "$root/block-$block"
    uv run --locked --all-packages python progress/top10_confirmation/sample.py --block "$block" \
        > "$root/block-$block/sampling.log" 2>&1
done
printf 'AUDIT all corpus blocks\n'
uv run --locked --all-packages python progress/top10_confirmation/audit.py
for block in a b c; do
    for seed in 0 1 2 3 4; do
        for arm in ordinary designed; do
            run="$root/block-$block/$arm/seeds/seed_$seed"
            mkdir -p "$run"
            printf 'TRAIN block=%s seed=%s arm=%s\n' "$block" "$seed" "$arm"
            uv run --locked --all-packages python progress/top10_confirmation/train.py \
                --block "$block" --arm "$arm" --seed "$seed" > "$run/train.log" 2>&1
        done
    done
done
printf 'REPORT complete confirmation\n'
uv run --locked --all-packages python progress/top10_confirmation/report.py
printf 'DONE frozen Top-10 confirmation\n'
