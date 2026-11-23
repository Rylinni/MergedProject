# Merged! strategy from trained agents

Data (all replayed move by move with replay/engine.js; every recorded point total was reproduced):
- **Main** (survival/score model, main_checkpoint.bin): 300 depth-1 games (seeds 1-300, 606,278 moves, mean 25,195, median 19,630, 2,021 moves/game), plus best_run.json (depth 2, 13,082 moves, 179,706). The best_run numbers match the depth-1 numbers to within about 1 point on every statistic below, so the depth-1 games stand in for it.
- **Greedy** baseline (points + 10*empty + 3*v*pairs): 300 games (76,686 moves, mean 2,312, 256 moves/game).
- **Big-merge** (bigmerge_convex5000.bin, convex:5000): 3,000 depth-1 games (1.0M moves) and 200 depth-2 games (127k moves, mean 18,576), plus bigmerge_run.json (the 65,520 game). 6-step chains came up in 24/200 depth-2 games (12%) and 63/3,000 depth-1 games (2.1%), which matches the ~14% and ~2.3% given in the brief.

Cells are written as row,col with 0,0 at the top left. "Empties" means empty cells.

## 1. Vocabulary
- **Rim**: the 16 border cells. **Corners**: the 4 corner cells. **Ring**: the 8 cells around the center. **Hub**: cell 2,2.
- **Onion**: a layout where values rise from the outside in: 1-3 on the rim (1s in the corners), 4-5 on the rim and ring, 6s on the ring diagonals (1,1 1,3 3,1 3,3), Ms on the hub and the "plus" cells next to it.
- **Furnace**: the hub plus its four orthogonal neighbours, where Ms are parked as a pair waiting for a third M.
- **Loaded pair**: two orthogonally adjacent equal tiles with at least one empty orthogonal neighbour, so one matching tile placed there merges them. A pair with no empty neighbour is a **dead pair**.
- **Buried tile**: a tile with no empty orthogonal neighbour.
- **Ladder of height k**: an empty cell where one single tile would chain k times, e.g. a 2-pair and a 3-pair both touching the same empty cell (height 2).
- **Hand-off**: a domino partner merge. The small half climbs to the big half's value, then the big half merges using the small half's cell.
- **Socket**: in big-merge play, the two empty cells that will take the domino. **S** (starter) takes the small half and **F** (fire cell) takes the big half; F is where the explosion happens.
- **Outs**: the pieces that would fire a setup at its full height.
- **Cash-out**: firing part of a ladder (a 1-3 step merge) to win back space, which gives up the big setup.
- **Debris**: tiles destroyed by the explosion that were not part of any merging group. Each debris point is worth x720 in a 6-step chain.
- **Dump**: the corners and edge cells where new 1s and 2s go.

## 2. Score/survival heuristics (main model), ranked

**H1. Space comes first: keep about 14 empties, and below 10 take every merge.**
The main model averages 14.5 empties after its move, and fewer than 2% of its positions have 7 or fewer. How often it declines an available merge depends on space:

| Empties before the move | Merges declined |
|---|---|
| 20 or more | 32% |
| 16-19 | 26% |
| 13-15 | 21% |
| 10-12 | 13% |
| 7-9 | 4.9% |
| 6 or fewer | 1.5% |

With 4 or fewer empties it declined 6 merges out of 3,113 moves. Why it matters: games end quickly once the decline starts. Averaged over the last moves of all games, empties go 14.0 (60 moves before the end), 11.0 (20 before), 4.9 (5 before), 0. That is about one cell lost per move over the final 20. The trained model recovers from tight spots: every game reached 5 or fewer empties at some point, and after the first time it lasted a median 953 more moves. Greedy lasted a median 19 more.

**H2. Run a furnace in the center, and fire every third M at once.**
After M is unlocked (median move 26), one value in seven is an M, and an M can only leave the board by exploding.
- Where the main model keeps Ms: hub 44%, each plus cell about 12%, rim 3.7%. Greedy keeps 56% of its Ms on the rim.
- When an M arrived and an explosion was possible, the main model fired it 14,475 times out of 14,650 (99%).
- When no explosion was possible, the M went to the ring (14,880) or the hub (12,544), and almost never to the rim (855).
- 94% of explosions happen on the hub or ring, where the 3x3 blast covers the full 9 cells. Each explosion clears 3.5 tiles besides the Ms.
- Greedy carries 2.06 Ms per board that has any Ms; the main model carries 1.55.

**H3. Build the onion: dump low tiles on the rim and keep the middle for high tiles.**
Density of each value relative to a uniform layout (main model):

| Value | Corners | Ring | Hub |
|---|---|---|---|
| 1 | 2.9x | 0.13x | |
| 4 | 1.46x | 0.68x | |
| 6 | 0.06x | 2.1x | |
| M | | | 10.9x |

New 1s and 2s that are not merging go to a corner 33% of the time, an edge 61%, and the ring 6%. Greedy's layout is flat (all values 0.75-1.26x everywhere, and 37% of its low placements land in the ring or hub). Why it works: merges land on the cell where you place, so placing a low tile outward walks the merged result toward the middle. The center stays free to hold the furnace and to be blasted.

**H4. When there is room, make a pair instead of taking a small merge.**
Of 309k moves where a merge was available, the main model declined 18.9% (greedy 1.5%) and took a smaller merge than the best on offer 13.7% of the time. In 67.5% of the declines its placement made a new pair. The merge it gave up was small (median 12 points, 90th percentile 50). It declines merges of 1-5 about 20-26% of the time, merges of 6 11.5%, and explosions 0.7%. Pairs are stored value that can later chain at x2 to x24. Overall it scores 1.35x greedy per move (12.5 vs 9.3 points in the late game) and survives 8x longer (2,021 vs 256 moves).

**H5. Keep every pair loaded and every tile reachable.**
98% of the main model's pairs have an empty neighbour (greedy 86%). It has 1.10 buried tiles per board (greedy 2.30) and 5.0 singletons (greedy 5.6). Don't seal a pair in with unrelated tiles.

**H6. Stack pairs into ladders.**
In a sample of 1 in 5 boards, 37% of the main model's boards have a ladder of height 2 or more and 3.8% have height 3 or more (greedy 18.5% and 0.8%). Its moves are 3-step chains 6.8% of the time (greedy 4.6%) and 4-step chains 0.73% (greedy 0.19%). Put pairs of consecutive values (2-2 next to 3-3, and so on) so they both touch one empty cell.

**H7. Dominoes: place them so the halves feed different groups or hand off.**
About half of dominoes merge (49.5%). 8.1% of merging dominoes use a hand-off, and 7.3% merge both halves in separate groups. Evidence that this is a deliberate skill is weak: the rates are similar for greedy (5.9% hand-offs).

**H8. When down to about 4 empties, cash out and bomb.**
With 4 or fewer empties the main model merges on 49% of its moves and almost never declines a merge. An M explosion is its best clearing tool. Death boards had on average 19.4 groups, 5.6 (now dead) pairs and 6.9 distinct values, so the end usually comes from too many values competing for space, not from one bad piece.

## 3. The big-merge game plan (playing for one huge move)

This is a different game. The big-merge model scores 5,729 on average at depth 1 (main: 25,195) and holds dangerous positions on purpose.

**Target shape.** The only 6-step (x720) chains found came from a domino of 2+4 (74 of 90) or 2+5 (16 of 90). All 90 ended in an M explosion on a plus cell next to the hub.
- 2+4: the 2 on S climbs 2->3->4, hands off to the 4 on F, then 4->5->6->M->explosion.
- 2+5: the 2 on S climbs 2->3->4->5, then the 5 on F goes 5->6->M->explosion.
- 3+4 and 3+5 give only 5 steps (x120). There were 345 of those (147 with 3+4, 112 with 2+4 where one rung was missing, e.g. no M pair so the chain ends at M creation, 59 with 3+5).
- Each cell can only use its free neighbours, so S has 3 and F has 3. That is why 6 is the cap and why a domino containing an M stops at 4.
- Minimum sum: 3x(2+3+4+5+6+7) = 81, so 81 x 720 = 58,320, the most common 6-step result. Extra points come from groups of 4 (for example a 4-tile 4-group, +4) and from debris. Observed maximum: 90 x 720 = 64,800. The 70,560 (98) move was found by search; the agents never played it.

**Ladder recipe (2+4 version, 11 tiles plus 2 empty cells):**
1. Choose F on a plus cell next to the hub (1,2 2,1 2,3 3,2). All 90 observed F cells were plus cells (1,2: 20; 2,1: 27; 2,3: 21; 3,2: 22). The M pair should be the hub plus one more cell, touching F. Your survival play has usually parked Ms there already.
2. Around F, using its three neighbours other than S: one M pair, one 6 pair and one 5 pair. Only one tile of each pair needs to touch F; the other tile can lie on the rim side.
3. Choose S as F's neighbour on the side facing the rim. Around S: a 2 pair and a 3 pair touching S, plus one single 4 touching S or F. This lone 4 joins the domino's 4 and the climbed 4. For 2+5, S instead needs pairs of 2, 3 and 4, and a single 5 must touch F.
4. Leave S and F empty, and keep no other cell where a stray 2 or 3 could trigger part of the ladder.
5. Wait for the out. A specific unordered domino arrives with probability 0.5 x 2/42, about 2.4% per move once 7 values are unlocked, so expect about 40 moves per out. A 2+4 setup also fires as a 5-step with 3+4 (about 9-10k points). At firing time the model had 2.0 outs at 5 steps or more, and exactly one out at 6 steps.

**Build order (91 six-step chains, how long each rung stood before firing, median):** 3-pair 16 moves, 2-pair 11, 6-pair 10, 5-pair 8, the 4 rung 7, M-pair 5. The rung finished last was most often the M pair (28/91), then the 5 pair (19) and the 4 rung (18). So build the low rungs on the rim first and the high rungs last. Protect the M pair: when a single M arrived while a 6-step was ready, the model did not explode it 71 times out of 78, whereas the main model explodes 99% of the time.

**How long to hold, and when to bail out (200 depth-2 games).**
- A 6-step setup was ready on 0.9% of boards, and a 5-step or better on 2.7%.
- Of 300 episodes with a 6-step ready, 27 fired the 6 (9%), 22 took a 5-step instead, 248 were lost and 3 ended the game.
- Successful episodes lasted 2.6 moves on average; lost ones 4.1.
- Setups were lost almost entirely at low space: 223 of 251 endings had 2-4 empties, usually by cashing a 1-3 step merge out of the ladder.
- Boards where the setup was ready averaged 5.5 empties, and the model kept it intact on 78% of its non-firing moves.

The rule: hold while you have 5 or more empties. Take a 5-step out (3+4) if it comes first. Cash out once you are down to about 3.

**Debris.** Because each debris point is worth 720, a 7 left in the blast is 5,040 points. Debris comes from the cells around F that no group uses, mostly F's diagonal neighbours. The model barely uses this: 54 of 90 six-step chains had no debris, and the average was 1.4 points (maximum 7). The arithmetic says to leave high tiles (5, 6, or an M that does not touch the M pair) on F's diagonals. This advice is arithmetic, not observed behaviour.

**Before the 65,520 move (bigmerge_run.json, moves 875-880; board rows from top):**
- 875 `..M.1 / 6.6.4 / 6.MM4 / 3.226 / 344..`: piece 5, placed at 1,1, making the 5 rung.
- 876: piece 4+5 to 1,3 and 0,3, 12 points, merging 4s off the right side.
- 877 `..M51 / 6565. / 6.MM. / 3.226 / 344..`: piece 5+M to 0,1 and 0,0. The 5 completes the 5 pair 0,1-1,1; the M is parked in a corner, not exploded.
- 878, 879: a 1 to 4,4 and a 4 to 2,4 as fillers, keeping S=3,1 and F=2,1 empty.
- 880: piece 4+2, with the 4 on 2,1 and the 2 on 3,1. The chain: 2s (3,2 3,3) give a 3; 3s (3,0 4,0) give a 4; a 4-tile 4-group with 4,1 and 4,2 gives a 5; 5s (0,1 1,1) give a 6; 6s (1,0 2,0) give an M; Ms (2,2 2,3) explode. That is 91 x 720 = 65,520. A 6-step out existed for 5 consecutive moves.

## 4. Phase advice
- **Opening (only 1-3 unlocked; about 9 moves).** Merge rate 36%, about 20.7 empties. Dump 1s in corners and build pairs. The evidence here is thin and the phase is short: 4 is unlocked at a median of move 8, 6 at move 20, M at move 26.
- **Middle (4-6 unlocked; about 18 moves).** The highest merge rate (46%). Push the growing tiles inward so that the first 6s land on the ring diagonals and the first M near the hub.
- **Late (M unlocked; over 98% of all moves).** This is the real game: furnace (H2), onion (H3), the space discipline of H1, and pairs over small merges when roomy (H4). For big-merge play this is also when you build the socket, since the 6-step needs an M pair.

## 5. Example positions

Board rows are listed top to bottom (row 0 first).

**A. best_run.json, move 3245.** Piece: single 4. Board before:
```
22554
.4M.4
..M5.
.....
.....
```
AI: 4 to 1,0, scoring 0. A merge worth 64 at 1,3 was available. With 15 empties the AI makes a loaded 4 pair instead of cashing the merge (H4).

**B. best_run.json, move 5049.** Piece: M. Board before:
```
.2511
3....
3MM55
...64
...24
```
AI: M to 3,2, scoring 34 (explosion clearing 3 tiles). This is the furnace: an M pair on the hub and a plus cell, and the third M fired at once (H2).

**C. best_run.json, move 6928.** Piece: M. Board before (3 empties):
```
34141
514M3
.3M.5
32.23
11521
```
AI: M to 2,3, scoring 38. The explosion clears 5 extra tiles. An emergency bomb when nearly full (H8).

**D. best_run.json, move 8159.** Piece: 5+3 domino. Board before:
```
22.21
66M.1
3.M.3
3.553
44..4
```
AI: 5 to 2,1 and 3 to 3,1, scoring 9,600, the best move of the 179,706 game. The chain: 3s give a 4; 4s give a 5; hand-off through a 4-tile 5-group; 6s give an M; Ms explode. A 5-step ladder made inside survival play, with the socket at 2,1 next to the furnace.

**E. games/big2/30.json (seed 30, big-merge, depth 2), move 86.** Piece: 2+4. Board before:
```
..421
56631
5.MM6
3.2.1
342.2
```
AI: 2 to 3,1 and 4 to 2,1, scoring 58,320 (6 steps, with the minimal sum of 81). This is the textbook socket. S=3,1 has a 2 pair (3,2 4,2), a 3 pair (3,0 4,0) and a single 4 (4,1). F=2,1 has a 5 pair (1,0 2,0), a 6 pair (1,1 1,2) and an M pair (2,2 2,3). 3+4 was the second out, at 9,240.

**F. bigmerge_run.json, move 880.** Piece: 4+2. Board before:
```
M5M51
6565.
6.MM4
3.226
344.1
```
AI: 4 to 2,1 and 2 to 3,1, scoring 65,520. The same socket mirrored onto a vertical column, with an extra M parked in the corner rather than exploded (section 3).

## 6. Caveats
- These are correlations in what a value network does, not proven causes. For example, the reason for any single declined merge is not known.
- Most main-model statistics come from depth-1 games. best_run (depth 2) agrees closely, but it is a single game.
- The spawn rules (uniform values, 50% dominoes) are unconfirmed for the real iOS game. The opening is very short in this engine, so the opening advice is weak.
- The greedy baseline is weak. "Better than greedy" does not mean each habit matters on its own; none of the heuristics were tested by ablation.
- The hand-off skill in survival play (H7) shows little evidence over greedy.
- The debris advice is arithmetic only. The 70,560 position was never reached by an agent; the observed maximum is 64,800 at depth 1 and 65,520 in bigmerge_run.json.
- The big-merge episode statistics come from 300 episodes and 90 six-step chains. The 9% conversion figure is from depth-2 play.

Scripts and raw outputs are in /tmp/claude-0/-home-user-MergedProject/0c47ff55-561c-55a2-96ae-10c101052147/scratchpad/strategy/. The files are lib.js, analyze.js, extra.js, bigchains.js, sumchains.js, episodes.js, lost.js, rungs.js, mready.js and find.js, with outputs in out/*.json and games in games/{main,greedy,big,big2}.
