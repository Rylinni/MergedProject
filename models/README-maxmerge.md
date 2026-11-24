# The largest possible merge, and reaching it fast

Goal (Ryan): only the single largest merge possible counts. Reach it in as few turns as possible;
resetting the game is allowed.

## The largest merge is 70,560, and here is every board that allows it

A move scores k! × W: k is the number of consecutive merges, W the sum of every tile merged
or destroyed by the explosion. Board states between moves never hold 3 touching equal tiles
(they would already have merged).

1. **At most 6 merges, and only with a domino.** Each merge needs a fresh matching group touching
   the landing cell, and cleared cells stay empty, so every merge uses up one of the landing cell's
   neighbours. A single piece lands in one cell: at most 4 merges. A domino lands in S (small half)
   and, after the hand-off, F (big half): 3 free neighbours each. The hand-off merge needs one
   extra matching tile next to S or F, so at most 6 merges (×720).
2. **The domino is 2+4.** Six merges ending in the explosion run 2→3→4→5→6→M→explosion, so the small
   half is a 2. If the big half is a 5, its extra tile has to touch F, which makes F's group 3 tiles as
   soon as the domino lands, and then the game scores the halves separately (confirmed from screen
   recordings), breaking the chain. With a 4, the extra tile sits next to S.
3. **Groups.** Each merge takes the landing tile plus one pair through one neighbour: 3 tiles. The
   hand-off takes F + S + a pair of 4s: 4 tiles. W from merges: 2·3 + 3·3 + 4·4 + 5·3 + 6·3 + 7·3 = 85.
4. **Debris.** The explosion clears the 3×3 around F. F's side neighbours were emptied by the merges,
   and its two near diagonals are S's neighbours, also emptied. Only F's two far diagonals can hold
   debris, and each must differ from the two group tiles it touches. Best: the 5 opposite S, an M on
   the diagonal next to the 6, a 6 on the diagonal next to the M: 13.

**Maximum = (85 + 13) × 720 = 70,560.** Simulated annealing over all legal boards (`build/maxmerge`,
14 long restarts) never exceeded it.

**Max board**, for any adjacent S, F in the inner 3×3 (24 placements × 6 orders × 2 mirror = 288 layouts):

| Cell | Holds |
|---|---|
| S, F | empty (they receive the 2 and the 4) |
| S's three other neighbours | one tile each of a 2 pair, a 3 pair and a 4 pair, any order |
| F's neighbour opposite S | a 5 (of a 5 pair) |
| F's two side neighbours | a 6 (of a 6 pair) and an M (of an M pair) |
| F's far diagonal beside the 6 | M |
| F's far diagonal beside the M | 6 |
| next to each of the six pair tiles, outside these cells | its partner |

That is 14 tiles; the other 9 cells can all be empty (annealing found such boards), so a finished
board can wait for the 2+4 domino (1 piece in 42) while extra pieces go into the spare cells.
`engine/maxboard.hpp` encodes the layouts; `engine/test_maxboard.cpp` builds 20,000 random boards
from this description and the engine confirms every one allows 70,560.

## Training: progress shaping and a curriculum

`engine/goal.cpp` (`build/merged_goal`) trains and evaluates "reach the target in as few turns as
possible, resets allowed". It maximises E[γ^T] (T = turns, counting every game in the episode) with
an n-tuple network: reaching the target is worth 1, a reset or game over is worth V(empty board).
Decisions use U = β·progress^p + V, where progress ∈ [0, 1] measures how much of the nearest max
layout is in place (`max_progress`). The network learns V = U − shaping (potential-based shaping,
which leaves the best policy unchanged).

Findings so far (all at 1-step search, 4-thread runs):

| Agent | Reaches | Turns |
|---|---|---|
| Big-merge model (convex:15000), no resets | any 6-step chain | mean 74,042, median 49,132 (200 episodes) |
| same | any 5-step chain | mean 12,082, median 8,064 |
| same | exactly 70,560 | none in 400,000 turns |
| big-merge values + progress only (β = 1, linear) | — | dies in ~30 turns: progress alone ignores survival |
| big-merge values + β = 1 × progress⁸, untrained | progress ≥ 0.85 | mean 3,460, median 2,353 |
| same | progress ≥ 0.95 | mean 40,779, median 25,467 |
| same | max-ready board | 1 in 400,000 turns |

The exact max is too rare to learn from directly, so training climbs a curriculum of targets,
each stage starting from the previous weights: progress ≥ 0.85 → ≥ 0.95 → max-ready → 70,560.

## Results

Each stage starts from the previous one's weights, with resets off during training (see "Resets").
Settings: β = 0.4, power 8, γ = 0.9995, initial weights = bigmerge_convex15000 scaled to V(empty) = 0.2.
Episodes are evaluated on fixed seeds.

| Target | Before training | After training |
|---|---|---|
| progress ≥ 0.85 | 3,060 turns | **705** (stage 1, `maxmerge_stage1.bin`; 200 episodes) |
| progress ≥ 0.95 | 40,779 | **5,784** (stage 2 checkpoint, `maxmerge_stage2_ckpt600.bin`; 150 episodes) |
| max-ready board | ~1 per 400,000 turns | **mean 46,381, median 34,352** (stage 2; 40/40 episodes) |
| **exactly 70,560** | never in 400,000 turns | **27 of 40 runs within 1M turns; median ~514,000 turns (~1,400 games)** |

Training past the stage-2 checkpoint made it worse: 10,310 turns to progress 0.95 after 900 more
episodes. Once successes are rare, TD slowly erodes the survival skill inherited from the big-merge
model, so each stage keeps the best checkpoint on held-out episodes.

**Converting a ready board.** It gets ready about once per 51,000 turns, but only ~15% of ready
boards become the 70,560 move: the 9 free cells sit between the required tiles, and after about
5 pieces nothing more fits without disturbing the setup. The domino needed is 1 piece in 42. A
"hold" rule (on a ready board, only moves that keep it ready) and a one-piece lookahead for it made
little difference (14.7% → 15.7% in simulation from a spacious ready board).

**Resets.** Resetting whenever V(fresh) beats the best move loops (a reset every ~8 turns): the
network under-values many early boards. With a margin (reset only if better by 0.15) it resets
3.3 times per success at stage 1, mostly a few moves before a game over, with no speed-up (695 vs
705 turns). The coach uses margin 0.15.

**Stage 3 (target: max-ready) failed.** After 150 episodes the policy collapsed: 290,152 turns and
848 games per success during training, and 38 of 40 evaluation episodes hit the 300,000-turn cap
(stage 2: 40 of 40 reached, median 34,352). With successes this rare, TD learns mostly "the future is
worth ~0" and erodes the inherited survival skill. The stage-2 checkpoint remains the model.

**What would help next.** Steer towards max boards whose free cells form one block, so the setup can
be held for longer, and train stage 3 (target: max-ready) with checkpoints chosen on held-out episodes.

## Phone

`coach/merge-coach-maxmerge.bin` (stage-2 model, MNQ1 with automatic scale) and `objective:
"maxmerge"` in Merge Coach. `coach/MAXMERGE-PATCH.md` lists the changes for an existing, locally
modified Merge Coach.js.
