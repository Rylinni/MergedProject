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

## Evaluate an agent

```bash
engine/build/merged_eval --agent greedy --games 20000
engine/build/merged_eval --agent expectimax --depth 2 --games 200 --csv scores.csv
```

Game `i` uses seed `seed + i`, so runs are reproducible. The output gives mean ± standard
error, median, p90, p99, max, moves per game, and the highest value unlocked.

## Results (mergedMain rules, seed 1)

| Agent | Games | Mean ± s.e. | Median | Max |
|---|---|---|---|---|
| Random | 200k | 82 ± 0.1 | 71 | 1,197 |
| Old sklearn MLP (`nnai.sav`, Python) | 200 | 476 ± 23 | 392 | 2,862 |
| Greedy: points + 10·empty + 3·v·(equal pairs) | 20k | 2,693 ± 18 | 1,913 | 22,684 |
| Expectimax depth 2, same heuristic | 200 | 37,374 ± 2,752 | 24,698 | 225,472 |

## Rules we are not sure match iOS

These are fields of `merged::Rules` (and flags of `merged_eval`). The defaults reproduce
`mergedMain.py`. **They change scores more than any choice of model**, so pin them down
from the real game before training seriously. Greedy-agent mean, 20k games each:

| Setting | Default (mergedMain) | Alternative | Greedy mean |
|---|---|---|---|
| Do M's spawn once unlocked? | yes (`--max-spawn 7`) | never (`--max-spawn 6`) | 2,693 → 558 |
| Spawn weight per value | uniform over unlocked | e.g. `--weights 4,4,2,2,1,1,1` | 2,693 → 862 |
| Chance a piece is a domino | 0.5 | `--p-double 0.3` | 2,693 → 6,188 |
| Room for a domino | interior cells only (looks like a bug) | anywhere (`--double-anywhere`) | 2,693 → 2,068 |
| Domino halves | always different values | may match (`--double-same-ok`) | — |
| Scoring | sum of merged values × (1·2·3·…) per chain; partner-merge caveat | ? | — |
| Opening | fully random from {1,2} | reviews mention a gentler ramp | — |

Public sources (launch-era reviews and guides) confirm merges of 3+ orthogonal neighbours
landing on the placed tile, rotatable dominoes, M from three 6's, and three M's clearing a 3×3
area. They do not give spawn rates, M spawning, or the combo formula.
