// N-tuple value network + expectimax move search, a JS port of engine/ntuple.hpp
// and engine/agents.hpp. Weights come from coach/make_weights.py (16-bit, "MNQ1").
(function (root) {
  const E = (typeof module !== "undefined" && module.exports) ? require("../replay/engine.js") : root.MergeEngine;
  const N = 5, CELLS = 25;

  // Same tuples and symmetry order as ntuple.hpp.
  function symmetries() {
    const s = [];
    for (let k = 0; k < 8; k++) {
      const m = new Uint8Array(CELLS);
      for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) {
        let a = x, b = y;
        for (let r = 0; r < (k & 3); r++) { const t = a; a = b; b = N - 1 - t; }
        if (k >= 4) b = N - 1 - b;
        m[x * N + y] = a * N + b;
      }
      s.push(m);
    }
    return s;
  }
  function baseTuples() {
    const rect = (r, c, h, w) => { const t = []; for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) t.push((r + i) * N + c + j); return t; };
    const t = [];
    for (let r = 0; r < 2; r++) for (let c = 0; c < 2; c++) t.push(rect(r, c, 2, 3));
    for (let r = 0; r < 3; r++) t.push(rect(r, 0, 1, 5));
    return t;
  }

  // bytes: Uint8Array of a "MNQ1" file.
  function loadWeights(bytes) {
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const magic = dv.getUint32(0, true);
    // MNQ1: int16 x scale (phone format). MNT1: float32 straight from the trainer (tests).
    const float = magic === 0x4d4e5431;
    if (!float && magic !== 0x4d4e5131) throw new Error("not a weights file");
    const nt = dv.getUint32(4, true), scale = float ? 1 : dv.getFloat32(8, true);
    const tuples = baseTuples();
    if (nt !== tuples.length) throw new Error("tuple count mismatch");
    let off = float ? 8 : 12;
    const read = float ? (o) => dv.getFloat32(o, true) : (o) => dv.getInt16(o, true), sz = float ? 4 : 2;
    const tables = tuples.map((t) => {
      const n = 1 << (3 * t.length);
      const arr = float ? new Float32Array(n) : new Int16Array(n);
      for (let i = 0; i < n; i++, off += sz) arr[i] = read(off);
      return arr;
    });
    const empty = float ? new Float32Array(CELLS + 1) : new Int16Array(CELLS + 1);
    for (let i = 0; i <= CELLS; i++, off += sz) empty[i] = read(off);
    const sym = symmetries();
    const cells = tuples.map((t) => { const cs = []; for (const m of sym) for (const c of t) cs.push(m[c]); return Int32Array.from(cs); });
    return { tables, empty, scale, cells, lens: tuples.map((t) => t.length) };
  }

  // Expected future points from an afterstate.
  function value(W, c) {
    let empties = 0;
    for (let i = 0; i < CELLS; i++) if (!c[i]) empties++;
    if (!empties) return 0;
    let v = W.empty[empties];
    for (let t = 0; t < W.tables.length; t++) {
      const tab = W.tables[t], cs = W.cells[t], len = W.lens[t];
      for (let o = 0; o < cs.length; o += len) {
        let idx = 0;
        for (let k = 0; k < len; k++) idx |= c[cs[o + k]] << (3 * k);
        v += tab[idx];
      }
    }
    return v * W.scale;
  }

  // Next-piece distribution: uniform over unlocked values, half dominoes of two
  // different values when an adjacent empty pair exists (mirrors piece_distribution).
  function pieceDistribution(c, unlocked) {
    const vals = [];
    for (let v = 1; v <= 7; v++) if (unlocked >> v & 1) vals.push(v);
    const k = vals.length, pd = E.roomForDouble(c) && k > 1 ? 0.5 : 0;
    const out = vals.map((v) => ({ vals: [v], p: (1 - pd) / k }));
    if (pd) for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) out.push({ vals: [vals[i], vals[j]], p: pd * 2 / (k * (k - 1)) });
    return out;
  }

  function afterValue(W, c, unlocked, d) {
    let empty = false;
    for (let i = 0; i < CELLS; i++) if (!c[i]) { empty = true; break; }
    if (!empty) return 0;
    if (d === 0) return value(W, c);
    let ev = 0;
    for (const { vals, p } of pieceDistribution(c, unlocked)) {
      let best = 0, first = true;
      for (const [a, b] of E.legalMoves(c, vals)) {
        const nb = Uint8Array.from(c);
        const r = E.apply(nb, vals, a, b, unlocked);
        const v = r.points + afterValue(W, nb, r.unlocked, d - 1);
        if (first || v > best) { best = v; first = false; }
      }
      ev += p * best;
    }
    return ev;
  }

  // Best placement for the current piece. depth 1 = points + value; depth 2 also
  // averages over the next piece. Falls back to depth 1 if depth 2 exceeds budgetMs.
  function bestMove(W, board, piece, unlocked, opts = {}) {
    const depth = opts.depth || 1, budget = opts.budgetMs || 4000;
    const c = Uint8Array.from(board);
    const moves = [];
    const vals = piece;  // legalMoves covers every orientation of a domino
    {
      for (const [a, b] of E.legalMoves(c, vals)) {
        const nb = Uint8Array.from(c);
        const r = E.apply(nb, vals, a, b, unlocked);
        moves.push({ vals, a, b, points: r.points, chain: r.chain, board: nb, unlocked: r.unlocked, score: r.points + value(W, nb) });
      }
    }
    if (!moves.length) return null;
    moves.sort((x, y) => y.score - x.score);
    let usedDepth = 1;
    if (depth >= 2) {
      const t0 = Date.now();
      let complete = true;
      for (const m of moves) {
        if (Date.now() - t0 > budget) { complete = false; break; }
        m.score2 = m.points + afterValue(W, m.board, m.unlocked, 1);
      }
      if (complete) { moves.sort((x, y) => y.score2 - x.score2); usedDepth = 2; }
    }
    const m = moves[0];
    return { vals: m.vals, a: m.a, b: m.b, points: m.points, chain: m.chain, expected: usedDepth === 2 ? m.score2 : m.score, depth: usedDepth, after: Array.from(m.board) };
  }

  // Values unlocked so far: everything up to the highest tile seen (at least 1 and 2).
  function guessUnlocked(board, piece) {
    let hi = 2;
    for (const v of board) if (v > hi) hi = v;
    for (const v of piece) if (v > hi) hi = v;
    let m = 0;
    for (let v = 1; v <= hi; v++) m |= 1 << v;
    return m;
  }

  // "Largest merge, fewest turns" objective (engine/goal.cpp --target max): the value of
  // an afterstate is U = beta * progress^power + V (V = this network); a full board
  // means game over, worth the same as a fresh game. A move scoring 70,560 is worth 1,
  // any other move gamma * U(after). Advise a reset when a fresh game beats the best
  // move by more than margin. Needs MergeMaxBoard (coach/maxboard.js).
  function bestMoveGoal(W, MB, board, piece, unlocked, o) {
    const opt = Object.assign({ beta: 0.4, power: 8, gamma: 0.9995, margin: 0.15 }, o || {});
    const c = Uint8Array.from(board);
    const uFresh = value(W, new Uint8Array(25));
    let best = null;
    for (const [a, b] of E.legalMoves(c, piece)) {
      const nb = Uint8Array.from(c);
      const r = E.apply(nb, piece, a, b, unlocked);
      let q;
      if (r.points >= 70560) q = 1;
      else {
        let empty = false;
        for (let i = 0; i < 25; i++) if (!nb[i]) { empty = true; break; }
        const u = empty ? opt.beta * Math.pow(MB.progress(nb), opt.power) + value(W, nb) : uFresh;
        q = opt.gamma * u;
      }
      if (!best || q > best.q) best = { vals: piece, a, b, points: r.points, chain: r.chain, q, after: Array.from(nb) };
    }
    const inGame = board.some((v) => v !== 0);
    const progressNow = MB.progress(board);
    if (!best) return { reset: true, reason: "no room", progress: progressNow };
    if (inGame && uFresh > best.q + opt.margin) return { reset: true, reason: "fresh game is better", progress: progressNow, move: best };
    return Object.assign({ reset: false, progress: progressNow, ready: MB.ready(best.after) }, best);
  }

  const api = { loadWeights, value, bestMove, bestMoveGoal, guessUnlocked, pieceDistribution };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.MergeModel = api;
})(this);
