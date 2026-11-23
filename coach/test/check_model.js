// The JS model must choose the same moves as the C++ agent that recorded a game.
// usage: node coach/test/check_model.js RUN.json WEIGHTS.bin DEPTH [maxMoves]
const fs = require("fs");
const E = require("../../replay/engine.js");
const M = require("../model.js");
const [runFile, wFile, depthArg, maxArg] = process.argv.slice(2);
const run = JSON.parse(fs.readFileSync(runFile));
const W = M.loadWeights(new Uint8Array(fs.readFileSync(wFile)));
const depth = +depthArg, max = +(maxArg || 1e9);
const c = new Uint8Array(25);
let unl = E.START_UNLOCKED, same = 0, diff = 0, n = 0, worst = 0, ms = 0;
for (const [v0, v1, a, b] of run.moves) {
  if (n++ >= max) break;
  const vals = v1 ? [v0, v1] : [v0];
  const t0 = Date.now();
  const m = M.bestMove(W, c, vals, unl, { depth, budgetMs: 1e9 });
  ms += Date.now() - t0;
  const sameCells = (m.a === a && m.b === b) || (m.a === b && m.b === a && m.vals[0] === vals[1]);
  if (sameCells) same++;
  else {
    diff++;
    // How much worse is the recorded move by the JS model's own estimate? (near-tie check)
    const nb = Uint8Array.from(c), r = E.apply(nb, vals, a, b, unl);
    const theirs = r.points + M.value(W, nb);
    if (depth === 1) worst = Math.max(worst, m.expected - theirs);
  }
  const r = E.apply(c, vals, a, b, unl); unl = r.unlocked;
}
console.log(`depth ${depth}: ${same}/${same + diff} moves identical (${(100 * same / (same + diff)).toFixed(2)}%)` +
  (depth === 1 && diff ? `, largest value gap on a disagreement ${worst.toPrecision(3)} points` : "") + `, ${(ms / n).toFixed(1)} ms/move`);
