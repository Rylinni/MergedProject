// Merge-game engine (JS port of engine/merged.hpp, iOS rules).
// Board: Uint8Array(25), row-major, 0 = empty, 1..6 = numbers, 7 = M.
(function (root) {
  const N = 5, CELLS = 25, EXPLODE = 8;
  const NB = [], BOX = [];
  for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) {
    const nb = [], box = [];
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= N || ny >= N) continue;
      box.push(nx * N + ny);
      if ((dx === 0) !== (dy === 0)) nb.push(nx * N + ny);
    }
    NB.push(nb); BOX.push(box);
  }

  // Connected same-valued group containing start, never stepping onto exclude.
  function flood(c, value, start, exclude) {
    const seen = new Uint8Array(CELLS), out = [start], stack = [start];
    seen[start] = 1;
    while (stack.length) {
      const i = stack.pop();
      for (const j of NB[i]) {
        if (!seen[j] && j !== exclude && c[j] === value) { seen[j] = 1; out.push(j); stack.push(j); }
      }
    }
    return out;
  }

  // Place piece vals at cells a (and b) on board c (mutated); returns
  // { points, unlocked, events } where events describe merges and explosions in order.
  function apply(c, vals, a, b, unlocked) {
    let withheld = 0, mult = 1, merges = 0, banked = 0, maxChain = 0;
    const events = [];
    function upgrade(p) {
      c[p]++; merges++; mult *= merges; maxChain = Math.max(maxChain, merges);
      if (c[p] === EXPLODE) {
        const hit = [];
        for (const i of BOX[p]) {
          if (i !== p) { withheld += c[i]; if (c[i]) hit.push(i); }
          c[i] = 0;
        }
        events.push({ type: "boom", cell: p, cells: hit });
      } else {
        unlocked |= 1 << c[p];
      }
    }
    function evalPiece(value, coor, pcell, pval, first) {
      let comp = flood(c, value, coor, pcell);
      if (pcell >= 0 && comp.length < 3) comp = flood(c, value, coor, -1);
      if (comp.length >= 3 && !(pcell >= 0 && first && pval === value)) {
        for (const i of comp) { withheld += c[i]; if (i !== coor) c[i] = 0; }
        events.push({ type: "merge", cell: coor, value, cells: comp });
        upgrade(coor);
        return true;
      }
      return false;
    }
    if (b < 0) {
      c[a] = vals[0];
      let v = vals[0];
      while (evalPiece(v, a, -1, 0, false)) v++;
    } else {
      c[a] = vals[0]; c[b] = vals[1];
      let sv, sc, bv, bc;
      if (vals[0] < vals[1]) { sv = vals[0]; sc = a; bv = vals[1]; bc = b; }
      else { sv = vals[1]; sc = b; bv = vals[0]; bc = a; }
      const biggerCanEval = flood(c, bv, bc, -1).length > 2;
      let v = sv, up = evalPiece(v, sc, -1, 0, false);
      while (up) { v++; up = evalPiece(v, sc, bc, bv, true); }
      if (biggerCanEval) { banked = mult * withheld; withheld = 0; mult = 1; merges = 0; }
      if (c[bc] === bv) {
        v = bv; up = evalPiece(v, bc, sc, sv, false);
        while (up) { v++; up = evalPiece(v, bc, sc, sv, false); }
      }
    }
    return { points: banked + mult * withheld, unlocked, events, chain: maxChain };
  }

  function legalMoves(c, vals) {
    const out = [];
    for (let a = 0; a < CELLS; a++) {
      if (c[a]) continue;
      if (vals.length === 1) { out.push([a, -1]); continue; }
      for (const b of NB[a]) if (!c[b]) out.push([a, b]);
    }
    return out;
  }

  // A domino only spawns if it can be placed (some adjacent empty pair).
  function roomForDouble(c) {
    for (let a = 0; a < CELLS; a++) if (!c[a]) for (const b of NB[a]) if (!c[b]) return true;
    return false;
  }

  // Uniform over unlocked values (1..7); half of pieces are dominoes with two different values.
  function spawn(c, unlocked, rand) {
    const vals = [];
    for (let v = 1; v <= 7; v++) if (unlocked >> v & 1) vals.push(v);
    const first = vals[Math.floor(rand() * vals.length)];
    if (rand() < 0.5 && roomForDouble(c)) {
      const rest = vals.filter((v) => v !== first);
      return [first, rest[Math.floor(rand() * rest.length)]];
    }
    return [first];
  }

  // The hand-written 1-step heuristic, used for hints in play mode.
  function heuristic(c) {
    let empty = 0, h = 0;
    for (let i = 0; i < CELLS; i++) {
      const v = c[i];
      if (!v) { empty++; continue; }
      if (i % N < N - 1 && c[i + 1] === v) h += 3 * v;
      if (i + N < CELLS && c[i + N] === v) h += 3 * v;
    }
    return empty ? h + 10 * empty : 0;
  }
  function bestMove(c, vals, unlocked) {
    let best = null, bv = -Infinity;
    for (const [a, b] of legalMoves(c, vals)) {
      const nb = Uint8Array.from(c);
      const r = apply(nb, vals, a, b, unlocked);
      const v = r.points + heuristic(nb);
      if (v > bv) { bv = v; best = [a, b]; }
    }
    return best;
  }

  const api = { N, CELLS, NB, apply, legalMoves, roomForDouble, spawn, bestMove, START_UNLOCKED: (1 << 1) | (1 << 2) };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.MergeEngine = api;
})(this);
