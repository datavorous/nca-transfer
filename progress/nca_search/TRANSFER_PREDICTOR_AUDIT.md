# Corpus-level transfer predictor audit

Reproduce: `uv run --all-packages python progress/nca_search/audit_transfer_predictors.py` 
(this script rewrites only this report). Sources: `replication_1m.csv`, the three 
`block-*/feature_audit.json` files, and `experiments.csv`. 
Snapshot rule: use seed 0 in **every** fresh block; extra completed model seeds 
are reported separately. The feature audits systematically 
sample 1,017 worlds per arm; gzip is from all 20,231 selected worlds per arm.

## Fresh independent rule blocks: A/B/C

The 9 arm-level outcomes are three **blocks**, each with three disjoint corpora. 
Repeated model seeds on a corpus and the 1,017 measured worlds are not independent 
transfer observations. Each arm has the same 19,715,325 NCA labels, 1,001,472 
TinyStories tokens, and original five gzip-bin quotas. Gain is unfiltered loss minus 
selected-arm loss, so positive is better.

| Block | High gain | Low gain | High AUC gain | High−unfiltered copy | High−unfiltered gzip |
|:--|--:|--:|--:|--:|--:|
| A | +0.08426 | -0.03266 | +0.09577 | +0.00916 | +0.000000 |
| B | +0.07415 | -0.04362 | +0.17088 | +0.00934 | -0.000137 |
| C | +0.04103 | -0.04323 | +0.01080 | +0.01037 | -0.000073 |

Across blocks: high-copy gain +0.06648 ± 0.02261; low-copy gain -0.03983 ± 0.00622 (mean ± sample SD, N=3 blocks). Leaving out A/B/C gives high-copy mean gains +0.05759 / +0.06265 / +0.07920.

For each feature, the correlation below centers the three arms **within each block** 
then pools their 9 centered points. Positive means higher feature accompanies lower 
1M loss. Leave-one-block-out values recompute that same descriptive correlation on 
the two remaining blocks. They are sensitivity checks, **not** cross-validated 
prediction accuracy or independent p values.

| Feature | Block-centered r | Omit A | Omit B | Omit C | High−unfiltered feature shift A/B/C |
|:--|--:|--:|--:|--:|:--|
| Horizontal copy excess | +0.971 | +0.977 | +0.963 | +0.981 | +0.00916 / +0.00934 / +0.01037 |
| Cell-change fraction | +0.588 | +0.386 | +0.363 | +0.911 | +0.00904 / +0.00275 / -0.00683 |
| Patch entropy | -0.407 | -0.545 | -0.358 | -0.360 | +0.00511 / -0.01099 / -0.02361 |
| Lag-2 recurrence | +0.968 | +0.985 | +0.948 | +0.982 | +0.01514 / +0.02367 / +0.02022 |
| Transition MI | +0.960 | +1.000 | +0.948 | +0.954 | +0.00659 / +0.01162 / +0.00656 |
| Spatial excess agreement | +0.637 | +0.665 | +0.583 | +0.734 | +0.00118 / +0.00088 / +0.00130 |
| Gzip ratio | -0.625 | -0.684 | -0.693 | -0.565 | +0.00000 / -0.00014 / -0.00007 |

### Paired direction check

For each of the 3 blocks, compare high vs unfiltered and unfiltered vs low. 
This produces 6 **dependent** directional contrasts. The count below asks whether 
the arm with more of the feature also has lower loss; it is not a significance test.

| Feature | Concordant contrasts |
|:--|--:|
| Horizontal copy excess | 6/6 |
| Cell-change fraction | 4/6 |
| Patch entropy | 3/6 |
| Lag-2 recurrence | 6/6 |
| Transition MI | 6/6 |
| Spatial excess agreement | 5/6 |
| Gzip ratio | 3/6 |

### Model-seed coverage

Complete three-arm model seeds per block: A=[0, 1, 2, 3, 4], B=[0, 1, 2, 3, 4], C=[0]. Common seeds across blocks: [0]. Extra seeds within one block test model initialization sensitivity there, not corpus-to-corpus uncertainty.

- Block A: high vs unfiltered mean paired 1M gain +0.05686, range -0.03456 to +0.11170; 4/5 model seeds win.
- Block B: high vs unfiltered mean paired 1M gain +0.04543, range +0.00126 to +0.08205; 5/5 model seeds win.

## Earlier exploratory selectors (iterations 01–06)

Each row below is one selected **corpus** compared with the original 40–50 corpus. The initial corpora share rule streams and many rules, and the six selectors were chosen sequentially. The reported paired model-seed mean is an outcome summary, not six independent causal experiments. AUC is normalized 0–1M loss area; positive gain is better. These data cannot form a common multifeature correlation table because the same full-corpus features were not saved for every iteration.

| Iteration | Selection | Model seeds | Mean endpoint gain | Mean AUC gain | Endpoint wins |
|:--|:--|--:|--:|--:|--:|
| 01 | low cell change | 1 | -0.03457 | -0.05608 | 0/1 |
| 02 | high cell change | 5 | +0.00302 | -0.03908 | 3/5 |
| 03 | high spatial agreement | 1 | -0.03915 | -0.10612 | 0/1 |
| 04 | high horizontal copy | 5 | +0.05917 | +0.02691 | 5/5 |
| 05 | low horizontal copy | 5 | -0.04275 | -0.08126 | 0/5 |
| 06 | high transition MI | 1 | -0.05833 | -0.08480 | 0/1 |

## Interpretation and limits

1. Across the three fresh blocks, seed-0 endpoint loss ranks high < unfiltered < low in every block. The high-copy selector beats its within-block unfiltered control by 0.04103–0.08426 loss units. The low-copy control loses in all three blocks. This supports the **selection policy** at the 1M checkpoint.
2. Horizontal copy is a useful ranking marker here, but lag-2 recurrence and transition MI also shift with the selection in every block. All three have 6/6 directional concordance. These are engineered arm contrasts rather than nine independent random corpus draws, so the high correlations cannot identify which property causes transfer; a prospective matched-factor sampling test is required.
3. Gzip means differ by less than 0.00014 between high and unfiltered arms, while the observed loss differences are much larger. Matched gzip-bin quotas limit coarse gzip confounding, but do not balance all token statistics.
4. Cell change and patch entropy do not shift consistently in the same direction across blocks. The single-corpus high-change result is seed dependent; high MI and high spatial agreement lose in their one-seed exploratory probes.
5. NCA validation loss was intentionally omitted from the predictor ranking. Each selected arm has its own filtered validation distribution. Those losses are comparable for training progress within an arm, but cannot serve as a shared cross-corpus representation-quality target. TinyStories validation uses the common downstream evaluation set.
6. With only three fresh blocks and seed-0 outcomes used in the correlations, they are descriptive and high-variance. The selectors were discovered after earlier probes; no unadjusted p value or apparent leave-one-block stability is confirmatory.

**Next decisions:** complete the queued A/B/C model-seed follow-up. For a distinct data target, test the frozen visible vertical lag-6 selector in `VERTICAL_LAG6_FROZEN_PROTOCOL.md`. For the horizontal-copy mechanism, later sample high/low copy arms matched on lag-2 recurrence and transition MI as scoped in `CONTROLLED_COPY_PROTOCOL.md`. The latter tests whether copy remains useful after its strongest measured co-movements are balanced.
