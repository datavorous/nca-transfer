# Frozen Top-10 fresh-corpus confirmation

Completed after freezing [PROTOCOL.md](PROTOCOL.md). Three disjoint corpus blocks are the independent generation units; five model initializations per block are paired within-block variation. Designed and Ordinary use contemporaneous, pre-feature hash-assigned rules; Scratch is the saved seed-matched deterministic control. Lower loss/AUC and higher speedup are better.

## Every block and seed

| Block | Seed | Scratch loss | Ordinary loss | Designed loss | Scratch−Ordinary | Scratch−Designed | Ordinary−Designed | Ordinary PPL | Designed PPL | Ordinary AUC | Designed AUC | Ordinary crossing (M) | Designed crossing (M) | Ordinary speedup | Designed speedup |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A | 0 | 4.31401 | 4.21683 | 4.13273 | +0.09718 | +0.18128 | +0.08410 | 67.82 | 62.35 | 4.94581 | 4.84822 | 6.304 | 4.918 | 1.586 | 2.033 |
| A | 1 | 4.31728 | 4.20153 | 4.16916 | +0.11575 | +0.14812 | +0.03237 | 66.79 | 64.66 | 4.96195 | 4.89267 | 6.260 | 5.602 | 1.597 | 1.785 |
| A | 2 | 4.32128 | 4.16766 | 4.13907 | +0.15362 | +0.18221 | +0.02859 | 64.56 | 62.74 | 4.89151 | 4.82967 | 5.494 | 4.923 | 1.820 | 2.031 |
| A | 3 | 4.35562 | 4.19442 | 4.12813 | +0.16121 | +0.22750 | +0.06629 | 66.32 | 62.06 | 4.95591 | 4.85707 | 5.503 | 4.777 | 1.817 | 2.093 |
| A | 4 | 4.30064 | 4.24920 | 4.13466 | +0.05144 | +0.16598 | +0.11454 | 70.05 | 62.47 | 5.00679 | 4.83760 | 7.106 | 5.003 | 1.407 | 1.999 |
| B | 0 | 4.31401 | 4.25047 | 4.14961 | +0.06354 | +0.16440 | +0.10085 | 70.14 | 63.41 | 4.99312 | 4.85961 | 6.890 | 4.964 | 1.451 | 2.015 |
| B | 1 | 4.31728 | 4.17819 | 4.16005 | +0.13909 | +0.15723 | +0.01814 | 65.25 | 64.08 | 4.90821 | 4.87810 | 5.639 | 5.287 | 1.773 | 1.891 |
| B | 2 | 4.32128 | 4.17932 | 4.13520 | +0.14196 | +0.18609 | +0.04412 | 65.32 | 62.50 | 4.89308 | 4.83495 | 5.447 | 4.942 | 1.836 | 2.024 |
| B | 3 | 4.35562 | 4.18467 | 4.17238 | +0.17095 | +0.18325 | +0.01230 | 65.67 | 64.87 | 4.93071 | 4.90990 | 5.281 | 5.080 | 1.894 | 1.968 |
| B | 4 | 4.30064 | 4.19936 | 4.14966 | +0.10128 | +0.15099 | +0.04970 | 66.64 | 63.41 | 4.94415 | 4.85504 | 6.316 | 5.229 | 1.583 | 1.912 |
| C | 0 | 4.31401 | 4.23220 | 4.15803 | +0.08181 | +0.15598 | +0.07416 | 68.87 | 63.95 | 4.98032 | 4.88110 | 6.665 | 5.302 | 1.500 | 1.886 |
| C | 1 | 4.31728 | 4.20114 | 4.18156 | +0.11614 | +0.13572 | +0.01958 | 66.76 | 65.47 | 4.93470 | 4.90893 | 6.145 | 5.843 | 1.627 | 1.712 |
| C | 2 | 4.32128 | 4.16517 | 4.12502 | +0.15612 | +0.19626 | +0.04015 | 64.40 | 61.87 | 4.88327 | 4.83751 | 5.265 | 4.871 | 1.899 | 2.053 |
| C | 3 | 4.35562 | 4.17290 | 4.15449 | +0.18273 | +0.20114 | +0.01841 | 64.90 | 63.72 | 4.91134 | 4.85729 | 4.951 | 4.861 | 2.020 | 2.057 |
| C | 4 | 4.30064 | 4.19222 | 4.14831 | +0.10842 | +0.15233 | +0.04392 | 66.17 | 63.33 | 4.95566 | 4.86682 | 6.198 | 5.448 | 1.613 | 1.836 |

## Corpus-level analysis

| Block | Mean Ordinary loss | Mean Designed loss | Mean paired loss gain | Within-block seed SD of gain | Designed wins | Mean AUC gain | Mean speedup gain |
|:--|--:|--:|--:|--:|--:|--:|--:|
| A | 4.20593 | 4.14075 | +0.06518 | 0.03609 | 5/5 | +0.09935 | 0.34264 |
| B | 4.19840 | 4.15338 | +0.04502 | 0.03512 | 5/5 | +0.06633 | 0.25451 |
| C | 4.19272 | 4.15348 | +0.03924 | 0.02271 | 5/5 | +0.06273 | 0.17660 |

Across-block mean ± sample SD of paired 10M loss gain: **+0.04981 ± 0.01361** (N=3 independent corpus blocks).
Across-block mean ± sample SD of paired 0–10M AUC gain: **+0.07614 ± 0.02018**.
Designed wins: **15/15 seed pairs** and **3/3 block means**. Seed pairs do not enlarge the independent-corpus N.

Ordinary versus Scratch: across-block mean ± sample SD loss gain **+0.12275 ± 0.00662**; wins **15/15 seed pairs**, **3/3 block means**.
Designed versus Scratch: across-block mean ± sample SD loss gain **+0.17256 ± 0.00732**; wins **15/15 seed pairs**, **3/3 block means**.
Across-block mean ± sample SD of Designed−Ordinary scratch-target speedup: **+0.258 ± 0.083×**.

## Full downstream curve

Each condition is averaged within corpus block over five model seeds, then across the three block means. Scratch is the same saved seed-matched reference in every block, so repetition does not make it three independent scratch corpora.

| TinyStories tokens | Scratch loss | Ordinary loss | Designed loss | Ordinary−Designed |
|--:|--:|--:|--:|--:|
| 0 | 10.96859 | 11.45111 | 11.41117 | +0.03995 |
| 251,904 | 9.95452 | 9.61935 | 9.57284 | +0.04651 |
| 501,760 | 8.44224 | 7.40889 | 7.32338 | +0.08551 |
| 1,001,472 | 6.53635 | 6.02473 | 5.92466 | +0.10007 |
| 2,000,896 | 5.56658 | 5.32870 | 5.17862 | +0.15008 |
| 3,000,320 | 5.05119 | 4.83588 | 4.72697 | +0.10891 |
| 5,001,088 | 4.53661 | 4.38377 | 4.32328 | +0.06049 |
| 7,501,696 | 4.34979 | 4.22404 | 4.17368 | +0.05036 |
| 10,000,000 | 4.32177 | 4.19902 | 4.14920 | +0.04981 |

The selected-corpus ~2× speedup was exploratory because Top-10 was chosen on that corpus. This report tests the frozen policy on disjoint rule populations. Horizontal recurrence can still covary with other trajectory properties; a positive result validates a selection policy, not a unique mechanism. Crossings count downstream language tokens only.

[Corpus audit](../../runs/nca-data-search/top10-confirmation/audit.json) verifies 121,386 distinct selected rule IDs and no prior selected-rule overlap.
