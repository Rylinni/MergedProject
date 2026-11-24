# Adding "largest merge" mode to an existing Merge Coach.js

For a Merge Coach.js that has local changes (convexK setting, Base64 input, DrawContext image
output). Apply these changes on top instead of replacing the file. The reference implementation is
`coach/dist/Merge Coach.js` on branch `claude/stoic-hypatia-kgxlm7`, built from `coach/scriptable_main.js`.

## What the mode does

The goal is exactly the largest possible move, **70,560**: a 2+4 domino chaining
2→3→4→hand-off→5→6→M→explosion (proof: `models/README-maxmerge.md`). The model ranks moves by how
soon they lead to that move. It also reports:
- **progress**: how much of the 70,560 setup is on the board (0–100%)
- **ready**: the setup is complete, so keep it and wait for a 2+4 domino (about 1 piece in 42)
- **reset**: starting a new game beats continuing this board

## 1. New file

Copy `coach/merge-coach-maxmerge.bin` (2,293,824 bytes) into the Scriptable iCloud folder, next to the
existing weights. It uses the same MNQ1 format and the same 7 tuples; only the scale differs, and that
is stored in its header.

## 2. Library code (replace two sections of the bundle)

The bundle is split by marker lines `// ---- engine ----`, `// ---- png ----`, `// ---- reader ----`,
`// ---- model ----`, `// ---- main ----`. Your local changes are all in `main`.

- Replace your `// ---- model ----` section with the one from the reference file. It is unchanged
  except for the new `bestMoveGoal()` (goal decision + hold rule).
- Insert the reference file's `// ---- maxboard ----` section (it registers `MC.MergeMaxBoard`)
  just before `// ---- main ----`.

`engine`, `png` and `reader` are unchanged.

## 3. Main script changes

**SETTINGS**: add these fields (keep your `convexK`):

```js
objective: "maxmerge",                       // "score" = previous behaviour
maxmergeWeightsFile: "merge-coach-maxmerge.bin",
maxmerge: { beta: 0.4, power: 8, gamma: 0.9995, margin: 0.15 },   // must match training
```

**Weights**: load `SETTINGS.maxmergeWeightsFile` instead of the usual file when
`SETTINGS.objective === "maxmerge"`. `MC.MergeModel.loadWeights` handles both files.

**Move choice**: in maxmerge mode, replace the `bestMove(...)` call with:

```js
const g = MC.MergeModel.bestMoveGoal(W, MC.MergeMaxBoard, read.board, read.piece, unlocked, SETTINGS.maxmerge);
// g.reset === true   -> advise "Start a new game" (g.reason: "no room" | "fresh game is better")
// otherwise g has the same fields as bestMove's result (vals, a, b, points, chain, after)
// plus g.progress (0..1, before the move) and g.ready (true if the board stays max-ready).
```

Do not apply convexK in this mode: `bestMoveGoal` has its own ranking.

**Output** (adapt to your DrawContext image):
- `g.reset`: no move to draw. Show "START A NEW GAME" over the board.
- `g.points >= 70560`: this is the maximum move; highlight it.
- `g.ready`: add a line "Setup complete – wait for 2+4".
- otherwise: add a line `Math.round(100 * g.progress) + "% of the 70,560 setup"`.

## 4. Check

- Run the script in the Scriptable app on a screenshot. The board read and the move should look like
  before, with the extra progress line.
- `bestMoveGoal` was checked against the C++ trainer on 3,000 positions: the best-move value matches to
  within 2e-5 with the 16-bit weights, and the reset decision matches in every case. It takes under
  1 s per move in tests, even without JIT.

## Expected performance (simulator, 1-step search)

| | |
|---|---|
| Reaches a max-ready board | about once per 51,000 turns |
| Converts a ready board into 70,560 | ~15% (the setup lasts ~5 pieces) |
| Turns until 70,560 | median ~514,000 (~1,400 games); 68% of runs within 1M turns |
| Untrained / previous models | never reached 70,560 in 400,000 turns |
