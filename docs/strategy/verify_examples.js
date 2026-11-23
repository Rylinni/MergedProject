const E = require('../../replay/engine.js');
const parse = (rows) => Uint8Array.from(rows.join('').split('').map(ch => ch === '.' ? 0 : ch === 'M' ? 7 : +ch));
const ex = [
 {id:'A', src:'Record game (179,706 pts), move 3,246', rows:['22554','.4M.4','..M5.','.....','.....'], vals:[4], cells:[[1,0]], expect:0,
  text:'A merge worth 64 points was available at row 2, col 4. With 15 empty cells the AI instead makes a loaded pair of 4s.', tag:'Pairs over small merges'},
 {id:'B', src:'Record game, move 5,050', rows:['.2511','3....','3MM55','...64','...24'], vals:[7], cells:[[3,2]], expect:34,
  text:'The furnace: an M pair on the hub and a plus cell. The third M is fired the moment it arrives.', tag:'Furnace'},
 {id:'C', src:'Record game, move 6,929', rows:['34141','514M3','.3M.5','32.23','11521'], vals:[7], cells:[[2,3]], expect:38,
  text:'Three empty cells left. The explosion clears five tiles besides the Ms: an emergency bomb.', tag:'Cash out and bomb'},
 {id:'D', src:'Record game, move 8,160', rows:['22.21','66M.1','3.M.3','3.553','44..4'], vals:[5,3], cells:[[2,1],[3,1]], expect:9600,
  text:'The best move of the 179,706 game: a 5-step ladder built during ordinary survival play, with the socket beside the furnace.', tag:'Ladder'},
 {id:'E', src:'Big-merge model, seed 30, move 87', rows:['..421','56631','5.MM6','3.2.1','342.2'], vals:[2,4], cells:[[3,1],[2,1]], expect:58320,
  text:'The textbook socket. S at row 4, col 2 touches a 2 pair, a 3 pair and a single 4. F at row 3, col 2 touches a 5 pair, a 6 pair and the M pair.', tag:'Socket'},
 {id:'F', src:'Big-merge replay, move 881', rows:['M5M51','6565.','6.MM4','3.226','344.1'], vals:[4,2], cells:[[2,1],[3,1]], expect:65520,
  text:'The same socket as E, mirrored onto a column, plus a 4-tile group of 4s for extra points. A spare M sits parked in the corner instead of exploding.', tag:'Socket'},
];
const out = ex.map(e => {
  const c = parse(e.rows), before = Array.from(c);
  const [a, b] = e.cells.map(([r, k]) => r * 5 + k);
  const r = E.apply(c, e.vals, a, e.cells.length > 1 ? b : -1, 254);
  if (r.points !== e.expect) throw new Error(`${e.id}: got ${r.points}, report says ${e.expect}`);
  const steps = r.events.filter(x => x.type === 'merge').map(x => ({ v: x.value, n: x.cells.length }));
  return { ...e, before, after: Array.from(c), placed: [a, b].filter(x => x !== undefined && x >= 0), points: r.points, chain: r.chain, steps, booms: r.events.filter(x => x.type === 'boom').map(x => x.cells.length) };
});
require('fs').writeFileSync('examples.json', JSON.stringify(out));
for (const o of out) console.log(o.id, o.points, 'chain', o.chain, o.steps.map(s => `${s.n}x${s.v}`).join(' '), 'boom', o.booms);
