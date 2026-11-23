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

## Results (iOS rules, seed 1)

| Agent | Games | Mean ± s.e. | Median | Max |
|---|---|---|---|---|
| Random | 200k | 84 ± 0.1 | 72 | 1,197 |
| Old sklearn MLP (`nnai.sav`, Python, mergedMain rules) | 200 | 476 ± 23 | 392 | 2,862 |
| Greedy: points + 10·empty + 3·v·(equal pairs) | 20k | 3,076 ± 21 | 2,156 | 32,792 |
| Expectimax depth 2, same heuristic | 200 | 44,281 ± 3,010 | 30,916 | 211,133 |

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
- Roughly half of pieces are dominoes (12 of 23 seen); every domino seen had two different values.

Still unconfirmed. Greedy-agent mean, 20k games each:

| Setting | Current default | Alternative | Greedy mean |
|---|---|---|---|
| Spawn weight per value | uniform over unlocked | `--weights 4,4,2,2,1,1,1` | 3,076 → 914 |
| Room for a domino | interior cells only (mergedMain quirk) | anywhere (`--double-anywhere`) | 3,076 → 2,322 |
| Exact domino chance | 0.5 | `--p-double P` | very sensitive |
| Opening pieces | fully random | reviews mention a gentler ramp | small |

The 16 pieces seen with M unlocked were spread roughly evenly (1:1, 2:1, 3:3, 4:3, 5:2, 6:4, M:2),
which fits uniform weights better than a low-value bias, but the sample is small.
