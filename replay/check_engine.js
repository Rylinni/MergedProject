const E = require('./engine.js');
const fs = require('fs');
function check(file) {
  const run = JSON.parse(fs.readFileSync(file));
  const c = new Uint8Array(25); let unl = E.START_UNLOCKED, score = 0;
  run.moves.forEach(([v0, v1, a, b, pts], k) => {
    const vals = v1 ? [v0, v1] : [v0];
    if (!E.legalMoves(c, vals).some(([x, y]) => x === a && y === b)) throw new Error(`illegal move ${k}`);
    const r = E.apply(c, vals, a, b, unl); unl = r.unlocked;
    if (r.points !== pts) throw new Error(`move ${k}: js ${r.points} vs c++ ${pts}`);
    score += r.points;
  });
  if (score !== run.score) throw new Error('final score mismatch');
  console.log(file, 'OK', run.moves.length, 'moves, score', score);
}
for (const f of process.argv.slice(2)) check(f);
// real-game cases from tests/test_ios_observations.py
const cases = [
 [[[0,0,3,6,6],[5,4,0,0,0],[0,6,1,0,4],[5,0,4,0,0],[2,2,1,1,0]],[6],[[1,4]],18],
 [[[0,3,0,0,6],[0,5,0,0,2],[0,0,0,0,2],[0,5,0,2,0],[0,0,0,0,0]],[2],[[2,3]],8],
 [[[6,0,3,3,7],[5,0,5,1,7],[0,6,1,0,4],[5,0,4,0,0],[2,2,1,1,0]],[3,6],[[0,1],[1,1]],9],
 [[[0,3,0,0,6],[2,0,4,4,2],[0,0,2,2,0],[0,0,4,0,1],[0,0,4,0,0]],[4,2],[[1,1],[2,1]],18],
 [[[0,3,0,0,6],[2,5,0,0,2],[2,3,0,0,0],[0,0,4,0,1],[0,0,4,0,0]],[2,3],[[3,0],[3,1]],162],
 [[[6,4,0,0,7],[5,6,5,1,7],[0,6,1,0,4],[5,0,4,0,0],[2,2,1,1,0]],[4,7],[[0,2],[0,3]],31],
];
for (const [bd, vals, cs, pts] of cases) {
  const c = Uint8Array.from(bd.flat());
  const r = E.apply(c, vals, cs[0][0]*5+cs[0][1], cs[1] ? cs[1][0]*5+cs[1][1] : -1, 254);
  if (r.points !== pts) throw new Error(`ios case ${pts}: got ${r.points}`);
}
console.log('6 real-game cases OK');
