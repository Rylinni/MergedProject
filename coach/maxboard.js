// Max-board progress and readiness: a JS port of engine/maxboard.hpp.
// A max board allows the largest possible move, 70,560 (see models/README-maxmerge.md).
(function (root) {
  const E = (typeof module !== "undefined" && module.exports) ? require("../replay/engine.js") : root.MergeEngine;
  const N = 5, M = 7;

  function makeLayouts() {
    const out = [];
    const inner = (r, c) => r >= 1 && r <= 3 && c >= 1 && c <= 3;
    const dr = [-1, 1, 0, 0], dc = [0, 0, -1, 1];
    const perms = [[2, 3, 4], [2, 4, 3], [3, 2, 4], [3, 4, 2], [4, 2, 3], [4, 3, 2]];
    for (let sr = 1; sr <= 3; sr++) for (let sc = 1; sc <= 3; sc++) for (let d = 0; d < 4; d++) {
      const fr = sr + dr[d], fc = sc + dc[d];
      if (!inner(fr, fc)) continue;
      const pr = dc[d], pc = dr[d];
      const S = sr * N + sc, F = fr * N + fc;
      const sN = [(sr - dr[d]) * N + (sc - dc[d]), (sr + pr) * N + (sc + pc), (sr - pr) * N + (sc - pc)];
      const far = (fr + dr[d]) * N + (fc + dc[d]);
      const fL = (fr + pr) * N + (fc + pc), fR = (fr - pr) * N + (fc - pc);
      const dL = (fr + dr[d] + pr) * N + (fc + dc[d] + pc), dR = (fr + dr[d] - pr) * N + (fc + dc[d] - pc);
      for (const pm of perms) for (let side = 0; side < 2; side++) {
        const six = side ? fL : fR, em = side ? fR : fL, dSix = side ? dL : dR, dEm = side ? dR : dL;
        const cell = [sN[0], sN[1], sN[2], far, six, em, dSix, dEm];
        const value = [pm[0], pm[1], pm[2], 5, 6, M, M, 6];
        let core = (1 << S) | (1 << F);
        for (const c of cell) core |= 1 << c;
        out.push({ S, F, cell, value, core });
      }
    }
    return out;
  }
  const LAYOUTS = makeLayouts();
  const K_MAX = (2 + 3 + 4 + 5 + 6 + 7 + 7 + 6) + 6 * 2;

  // Progress towards any max board, in [0, 1] (same scoring as max_progress in C++).
  function progress(board) {
    let best = 0;
    for (const m of LAYOUTS) {
      let s = 0;
      if (board[m.S]) s -= 8;
      if (board[m.F]) s -= 8;
      for (let k = 0; k < 8; k++) {
        const v = board[m.cell[k]], want = m.value[k];
        if (v === want) {
          s += want;
          if (k < 6) for (const q of E.NB[m.cell[k]]) if (!(m.core >> q & 1) && board[q] === want) { s += 2; break; }
        } else if (v) s -= want;
      }
      if (s > best) best = s;
    }
    return best / K_MAX;
  }

  // True if a 2+4 domino placed now scores 70,560.
  function ready(board) {
    const c = Uint8Array.from(board);
    for (const [a, b] of E.legalMoves(c, [2, 4])) {
      const nb = Uint8Array.from(c);
      if (E.apply(nb, [2, 4], a, b, 254).points >= 70560) return true;
    }
    return false;
  }

  const api = { progress, ready, LAYOUTS };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.MergeMaxBoard = api;
})(this);
