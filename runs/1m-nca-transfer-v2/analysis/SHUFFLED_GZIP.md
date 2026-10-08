# Gzip bands of the shuffled NCA control

This is a read-only analysis of the five derived shuffled packed corpora. The live training queue and all packed files were left unchanged. There is no single band for an entire corpus: the generator assigns a gzip band **per trajectory**.

## All 25,000 trajectories per seed

| Corpus | Mean ± SD gzip ratio | 30–40 | 40–50 | 50+ | Still 40–50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original 40–50 source | 0.45525 ± 0.02818 | 0 | 25,000 | 0 | 100.00% |
| Shuffled seed 0 | 0.45893 ± 0.02679 | 27 | 24,267 | 706 | 97.07% |
| Shuffled seed 1 | 0.45892 ± 0.02678 | 27 | 24,287 | 686 | 97.15% |
| Shuffled seed 2 | 0.45892 ± 0.02679 | 22 | 24,289 | 689 | 97.16% |
| Shuffled seed 3 | 0.45892 ± 0.02680 | 36 | 24,247 | 717 | 96.99% |
| Shuffled seed 4 | 0.45893 ± 0.02681 | 29 | 24,308 | 663 | 97.23% |

No trajectory fell below the 30–40 band. The five shuffled means lie between 0.45892 and 0.45893; the original mean is 0.45525. Shuffling raises the mean ratio by about 0.00367.

## Selected training worlds (19,975 per seed)

| Seed | 30–40 | 40–50 | 50+ |
| ---: | ---: | ---: | ---: |
| 0 | 21 | 19,390 | 564 |
| 1 | 21 | 19,400 | 554 |
| 2 | 19 | 19,406 | 550 |
| 3 | 29 | 19,364 | 582 |
| 4 | 23 | 19,411 | 541 |

## Method and interpretation

The original `nca.analyze.gzip_ratio` metric compresses the row-major stream of 972 `int32` 2×2 patch IDs for each 27-frame trajectory with gzip level 9 and `mtime=0`, then divides compressed bytes by 3,888 raw bytes. Grid start/end markers are excluded. This analysis applies exactly that byte-level operation to the already packed trajectories. The original first-row result matches its generation receipt, and two shuffled rows matched the public `gzip_ratio` function after decoding back to grids. The band cutoffs are [0.30,0.40), [0.40,0.50), and ≥0.50 for the columns above.

**Conclusion:** The shuffled control remains overwhelmingly in the same gzip-complexity band as its 40–50 source, although roughly 3% of trajectories cross a cutoff. Gzip ratio alone does not measure whether rule-following or temporal structure survives; the shuffle deliberately permutes patch-token order. The five rows are deterministic reshuffles of the same 25,000 source trajectories, not 125,000 independent worlds.

Exact counts, range, quantiles, and calculation metadata are in [shuffled_gzip_bands.json](shuffled_gzip_bands.json).
