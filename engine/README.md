# Fast Merged! engine

A C++ port of `mergedMain.py`, about 1000x faster: ~680k random games/s on 4 cores.
It is checked against the Python engine move for move, so every agent and
training method can rely on it.

## Build and test

```bash
make -C engine
python -m pytest tests/test_engine_parity.py   # PARITY_CASES=100000 for a longer run
```

`tests/test_engine_parity.py` compares the C++ engine with `mergedMain.py`. It checks
results after every move (board, score, unlocks), legal moves, the domino-room check,
and the next-piece distribution, across random-play and dense synthetic boards
(chains, partner merges, M explosions). Six deliberate engine bugs were each caught by it.

## Train the n-tuple value network (Phase 2)

```bash
engine/build/merged_train --out weights.bin --games 2000000 --alpha 0.05 --alpha-end 0.01
engine/build/merged_eval --agent ntuple --weights weights.bin --depth 1 --games 2000
engine/build/merged_eval --agent ntuple --weights weights.bin --depth 2 --games 200
```

`ntuple.hpp`: seven lookup tables (four 2×3 blocks and three full rows, each applied to all
8 rotations/reflections of the board) plus a table on the number of empty cells. That is 1.1M
weights (4.6 MB) and 57 lookups per evaluation. The value is the expected future points from
the position right after a move. `train.cpp` learns it by self-play with TD(0): each
position's value moves towards (next move's points + next position's value). It uses all
cores, saves a checkpoint every `--report` games, and `--in` resumes from a checkpoint.

## Top-score mode

The goal is the best single game rather than the best average. `merged_train --moments-out M.bin`
trains a second network on E[G²] of the future score, which gives its spread
(std = √(E[G²] − V²)). `merged_eval --moments M.bin --risk λ` then picks moves by V + λ·std.

Measured (500 games each, same seeds, 1-step search, network after ~1.2M training games):

| λ | Mean | p99 | Max |
|---|---|---|---|
| −0.15 | 9,644 | 43,464 | 61,011 |
| −0.05 | 16,446 | 81,605 | 99,127 |
| **0** | **17,626** | **82,892** | **122,601** |
| 0.05 | 18,989 | 88,420 | 125,371 |
| 0.25 | 13,254 | 68,385 | 83,224 |
| 0.5 | 2,667 | 14,758 | 20,613 |

Seeking variance does not raise the top score. The spread is largest in near-death
positions, and in this game big scores come from surviving longer, so the mean-optimal
policy is also (within noise) the top-score policy. **For a record game, use the mean
model with the deepest affordable search and run many games:** the best of N grows with
N. `merged_eval` prints the seed of the best game so it can be replayed exactly.

## Largest-merge mode

Goal: the most points in a single move. All on the same 1,000 seeds with 1-step search, except the last row.

| Model | Avg biggest move per game | Median | Best | Avg total score |
|---|---|---|---|---|
| Main model (`--objective sum`) | 1,179 | 402 | 58,320 | 22,845 |
| Pure max objective (`--objective max`, from scratch) | 1,419 | 120 | 7,200 | 1,611 |
| Big-merge (`--objective convex:5000`, fine-tuned from main) | 2,426 | 348 | 62,640 | 6,148 |
| Big-merge, 2-step search (100 games) | 10,434 | 1,008 | 65,520 | 20,176 |

Sweep of K for `convex:K` (each fine-tuned 60k games from the main checkpoint; 1,000 games, 1-step):

| K | Avg biggest move | Best | Games with a 6-step chain | Avg total score |
|---|---|---|---|---|
| 500 (from the K=5000 model) | 759 | 61,200 | 0.6% | 1,466 |
| 2,000 | 979 | 61,200 | 0.9% | ~3,500 |
| 5,000 | 2,426 | 62,640 | 2.3% | 6,148 |
| **15,000** | **3,050** | **64,080** | **2.4%** | ~18,000 |
| 50,000 | 1,360 | 10,680 | 0% | 27,834 |

Too strong a weighting kills survival, and with it the chances to set up a chain; too weak and it
plays like the main model. K = 15,000 is the best big-merge setting. Note that K = 50,000 scored
27,834 total on the same seeds where `main_final.bin` scored 25,631 (about 1.9 standard errors):
a mild reward for big merges may also help total score. Not yet confirmed.

The largest single move possible (found by `build/maxmerge`, annealing over legal boards) is
70,560: a 2+4 domino, 2→3→4, hand-off, 5→6→M→explosion, 98 × 720.

`--objective max` learns "biggest single move still to come" and plays max(points, V). It learns
to rush one explosion and die (games of ~45 moves), so its ceiling is low. Fine-tuning the main
model with it made games collapse the same way. `convex:K` keeps the sum (so surviving still
pays) but rewards points + points²/K, which makes big chains worth far more than small merges.
It doubles the average biggest move and keeps the high ceiling. The 65,520 move is a 6-step chain
(91 points × 720) ending in an M explosion.

`merged_eval --record-by merge` records the game with the biggest single move instead of the
highest score.

## Evaluate an agent

```bash
engine/build/merged_eval --agent greedy --games 20000
engine/build/merged_eval --agent expectimax --depth 2 --games 200 --csv scores.csv
```

Game `i` uses seed `seed + i`, so runs are reproducible. The output gives mean ± standard
error, median, p90, p99, max, moves per game, and the highest value unlocked.

## Results (iOS rules, seed 1)

| Agent | Games | Mean ± s.e. | Median | Max |
|---|---|---|---|---|
| Random | 200k | 84 ± 0.1 | 72 | 1,197 |
| Old sklearn MLP (`nnai.sav`, Python, mergedMain rules) | 200 | 476 ± 23 | 392 | 2,862 |
| Greedy: points + 10·empty + 3·v·(equal pairs) | 20k | 3,076 ± 21 | 2,156 | 32,792 |
| Expectimax depth 2, same heuristic (old domino rule) | 200 | 44,281 ± 3,010 | 30,916 | 211,133 |
| **n-tuple `models/main_final.bin`, depth 1** | 2,000 | 25,765 ± 563 | 18,322 | 181,055 |
| **n-tuple `models/main_final.bin`, depth 2** | 40 | 125,880 ± 20,696 | 89,163 | 486,247 |

Rows from the n-tuple model use the current rules (dominoes spawn whenever an adjacent empty
pair exists). The 486,247 game (seed 95035, 34,935 moves) is the one in the record replay page.

## Rules: what is confirmed and what is not

`merged::Rules` defaults follow screen recordings of the iOS game, and
`tests/test_ios_observations.py` replays the transcribed moves. `Rules::legacy()`
(`--legacy-scoring`) reproduces `mergedMain.py`, which the parity test uses.

Confirmed from recordings (all matched exactly, including board after the move):
- 3+ orthogonally connected equal tiles merge onto the placed cell; points = sum of merged values.
- Chains multiply everything scored in the move by 1·2·3·… (one factor per merge step):
  a partner chain of 2's→3, joining the domino's 3→4→5 scored (6+9+12)×6 = 162.
- **Changed from mergedMain:** if a domino's two halves merge in separate groups, both
  score (6 + 12 = 18; mergedMain kept only the 12).
- **Changed from mergedMain:** three M's explode the 3×3 around the placed M, and the
  destroyed tiles score their values (21 + 10 = 31; mergedMain gave 21).
- M's spawn as pieces once unlocked, including in dominoes ([M,3], [4,M]).
- A domino spawns whenever it can be placed, i.e. some adjacent pair of cells is empty
  (game owner). mergedMain only checked interior cells (`--double-room interior`).
- Roughly half of pieces are dominoes (12 of 23 seen); every domino seen had two different values.

Still unconfirmed:

| Setting | Current default | Alternative | Greedy mean |
|---|---|---|---|
| Spawn weight per value | uniform over unlocked (assumed) | `--spawn-weights 4,4,2,2,1,1,1` | large drop |
| Exact domino chance | 0.5 | `--p-double P` | very sensitive |
| Opening pieces | fully random | reviews mention a gentler ramp | small |

The 16 pieces seen with M unlocked were spread roughly evenly (1:1, 2:1, 3:3, 4:3, 5:2, 6:4, M:2),
which fits uniform weights better than a low-value bias, but the sample is small.
