// Variables used by Scriptable.
// These must be at the very top of the file. Do not edit.
// icon-color: deep-blue; icon-glyph: chess-board;
// Merge Coach: reads a screenshot of the merge game and suggests the best move.
// Built by coach/build.py from replay/engine.js, coach/png.js, coach/reader.js,
// coach/model.js, coach/maxboard.js and coach/scriptable_main.js. Edit those, not this file.

const MC = {};
// ---- engine ----
(function (module, require) {
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

}).call(MC, undefined, undefined);
// ---- png ----
(function (module, require) {
// Minimal PNG decoder (non-interlaced, 8 or 16-bit, grey/RGB/RGBA/grey+alpha)
// with its own zlib inflate, for environments without image APIs (Scriptable).
// decode(Uint8Array) -> { width, height, data: Uint8Array RGBA }
(function (root) {
  // ---- inflate (RFC 1950/1951), after zlib's puff.c ----
  const LBASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
  const LEXT = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
  const DBASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
  const DEXT = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
  const ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

  function huffman(lengths, n) {
    const count = new Uint16Array(16), symbol = new Uint16Array(n), offs = new Uint16Array(16);
    for (let i = 0; i < n; i++) count[lengths[i]]++;
    for (let i = 1; i < 15; i++) offs[i + 1] = offs[i] + count[i];  // length 0 = unused symbol
    for (let i = 0; i < n; i++) if (lengths[i]) symbol[offs[lengths[i]]++] = i;
    return { count, symbol };
  }

  function inflate(src, start, expected) {
    let pos = start, bitbuf = 0, bitcnt = 0;
    let out = new Uint8Array(Math.max(expected || 0, 1024)), olen = 0;
    const need = (n) => {
      if (olen + n <= out.length) return;
      const bigger = new Uint8Array(Math.max(out.length * 2, olen + n));
      bigger.set(out.subarray(0, olen)); out = bigger;
    };
    const bits = (n) => {
      let v = bitbuf;
      while (bitcnt < n) {
        if (pos >= src.length) throw new Error("PNG data ended early");
        v |= src[pos++] << bitcnt; bitcnt += 8;
      }
      bitbuf = v >>> n; bitcnt -= n;
      return v & ((1 << n) - 1);
    };
    const decode = (h) => {
      let code = 0, first = 0, index = 0;
      for (let len = 1; len < 16; len++) {
        code |= bits(1);
        const c = h.count[len];
        if (code - c < first) return h.symbol[index + (code - first)];
        index += c; first += c; first <<= 1; code <<= 1;
      }
      throw new Error("bad Huffman code");
    };
    const codes = (lh, dh) => {
      for (;;) {
        let sym = decode(lh);
        if (sym < 256) { need(1); out[olen++] = sym; continue; }
        if (sym === 256) return;
        sym -= 257;
        const len = LBASE[sym] + bits(LEXT[sym]);
        const ds = decode(dh);
        const dist = DBASE[ds] + bits(DEXT[ds]);
        if (dist > olen) throw new Error("bad distance");
        need(len);
        for (let i = 0; i < len; i++, olen++) out[olen] = out[olen - dist];
      }
    };
    let fixedL = null, fixedD = null;
    let last = 0;
    while (!last) {
      last = bits(1);
      const type = bits(2);
      if (type === 0) {
        bitbuf = 0; bitcnt = 0;
        const len = src[pos] | (src[pos + 1] << 8); pos += 4;
        need(len); out.set(src.subarray(pos, pos + len), olen); olen += len; pos += len;
      } else if (type === 1) {
        if (!fixedL) {
          const l = new Uint8Array(288);
          for (let i = 0; i < 144; i++) l[i] = 8;
          for (let i = 144; i < 256; i++) l[i] = 9;
          for (let i = 256; i < 280; i++) l[i] = 7;
          for (let i = 280; i < 288; i++) l[i] = 8;
          fixedL = huffman(l, 288);
          fixedD = huffman(new Uint8Array(30).fill(5), 30);
        }
        codes(fixedL, fixedD);
      } else if (type === 2) {
        const nlen = bits(5) + 257, ndist = bits(5) + 1, ncode = bits(4) + 4;
        const lens = new Uint8Array(320);
        for (let i = 0; i < ncode; i++) lens[ORDER[i]] = bits(3);
        const ch = huffman(lens, 19);
        const ll = new Uint8Array(nlen + ndist);
        for (let i = 0; i < nlen + ndist;) {
          const sym = decode(ch);
          if (sym < 16) { ll[i++] = sym; continue; }
          let rep, val = 0;
          if (sym === 16) { val = ll[i - 1]; rep = 3 + bits(2); }
          else if (sym === 17) rep = 3 + bits(3);
          else rep = 11 + bits(7);
          while (rep--) ll[i++] = val;
        }
        codes(huffman(ll.subarray(0, nlen), nlen), huffman(ll.subarray(nlen), ndist));
      } else throw new Error("bad block type");
    }
    return out.subarray(0, olen);
  }

  // ---- PNG ----
  function decode(bytes) {
    const sig = [137, 80, 78, 71, 13, 10, 26, 10];
    for (let i = 0; i < 8; i++) if (bytes[i] !== sig[i]) throw new Error("not a PNG");
    const u32 = (o) => ((bytes[o] << 24) | (bytes[o + 1] << 16) | (bytes[o + 2] << 8) | bytes[o + 3]) >>> 0;
    let pos = 8, width = 0, height = 0, depth = 0, ctype = 0, interlace = 0;
    const idat = [];
    let idatLen = 0;
    while (pos < bytes.length) {
      const len = u32(pos), type = String.fromCharCode(bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]);
      const d = pos + 8;
      if (type === "IHDR") { width = u32(d); height = u32(d + 4); depth = bytes[d + 8]; ctype = bytes[d + 9]; interlace = bytes[d + 12]; }
      else if (type === "IDAT") { idat.push(bytes.subarray(d, d + len)); idatLen += len; }
      else if (type === "IEND") break;
      pos = d + len + 4;
    }
    if (interlace) throw new Error("interlaced PNG not supported");
    const channels = { 0: 1, 2: 3, 4: 2, 6: 4 }[ctype];
    if (!channels || (depth !== 8 && depth !== 16)) throw new Error(`unsupported PNG (colour type ${ctype}, ${depth}-bit)`);
    const z = new Uint8Array(idatLen);
    let o = 0;
    for (const part of idat) { z.set(part, o); o += part.length; }
    const bpp = channels * depth / 8, stride = width * bpp;
    const raw = inflate(z, 2, height * (stride + 1));
    const img = new Uint8Array(width * height * 4), prev = new Uint8Array(stride), line = new Uint8Array(stride);
    let p = 0;
    for (let y = 0; y < height; y++) {
      const f = raw[p++];
      for (let i = 0; i < stride; i++) {
        const x = raw[p++], a = i >= bpp ? line[i - bpp] : 0, b = prev[i], c = i >= bpp ? prev[i - bpp] : 0;
        let v;
        if (f === 0) v = x;
        else if (f === 1) v = x + a;
        else if (f === 2) v = x + b;
        else if (f === 3) v = x + ((a + b) >> 1);
        else {
          const pp = a + b - c, pa = Math.abs(pp - a), pb = Math.abs(pp - b), pc = Math.abs(pp - c);
          v = x + (pa <= pb && pa <= pc ? a : pb <= pc ? b : c);
        }
        line[i] = v & 255;
      }
      const step = depth / 8;  // for 16-bit keep the high byte
      for (let xI = 0; xI < width; xI++) {
        const s = xI * bpp, t = (y * width + xI) * 4;
        const ch = (k) => line[s + k * step];
        if (channels >= 3) { img[t] = ch(0); img[t + 1] = ch(1); img[t + 2] = ch(2); img[t + 3] = channels === 4 ? ch(3) : 255; }
        else { img[t] = img[t + 1] = img[t + 2] = ch(0); img[t + 3] = channels === 2 ? ch(1) : 255; }
      }
      prev.set(line);
    }
    return { width, height, data: img };
  }

  const api = { decode, inflate };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.MergePNG = api;
})(this);

}).call(MC, undefined, undefined);
// ---- reader ----
(function (module, require) {
// Screenshot reader for the iOS game: pixels -> { board[25], piece[1 or 2] }.
// Calibrated on 1206x2622 screen recordings (dark theme). Works at any scale of
// that screenshot (e.g. resized to 402 px wide), since every coordinate is
// derived from the image width.
(function (root) {
  // Reference geometry at 1206 px wide.
  const REF_W = 1206;
  const CELL0_X = 18, PITCH = 240, CELL = 216, BOARD_TOP_HINT = 724;
  const TRAY = { x0: 330, x1: 1206, y0: 1990, y1: 2560 };

  // Tile colours. 7 = M (white tile with a dark letter). Each colour is listed as
  // measured from screen recordings (sRGB) and as the same colour encoded in Display P3,
  // since screenshots may be saved in either.
  const PALETTE = [
    [0, [51, 51, 51]],      // empty cell (neutral grey: same in both)
    [-1, [37, 37, 37]],     // background
    [1, [114, 66, 151]], [1, [107, 68, 146]],
    [2, [239, 99, 30]], [2, [222, 108, 53]],
    [3, [253, 42, 77]], [3, [233, 66, 83]],
    [4, [52, 126, 211]], [4, [72, 124, 205]],
    [5, [116, 178, 42]], [5, [130, 176, 67]],
    [6, [252, 192, 32]], [6, [243, 194, 74]],
    [7, [254, 254, 254]],
  ];

  // Nearest palette entry, or null for pips, glyphs and anti-aliased edges.
  function classify(r, g, b) {
    let best = null, bd = 1e9;
    for (const [v, [pr, pg, pb]] of PALETTE) {
      const d = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2;
      if (d < bd) { bd = d; best = v; }
    }
    return bd <= 45 * 45 ? best : null;
  }

  function px(img, x, y) {
    x = Math.max(0, Math.min(img.width - 1, Math.round(x)));
    y = Math.max(0, Math.min(img.height - 1, Math.round(y)));
    const i = (y * img.width + x) * 4;
    return classify(img.data[i], img.data[i + 1], img.data[i + 2]);
  }

  // Majority vote over a grid of points inside one cell.
  function readCell(img, s, top, r, c) {
    const x0 = (CELL0_X + c * PITCH) * s, y0 = top + r * PITCH * s, size = CELL * s;
    const votes = new Map();
    let seen = 0;
    for (let i = 0; i < 9; i++) for (let j = 0; j < 9; j++) {
      const v = px(img, x0 + size * (0.12 + 0.76 * i / 8), y0 + size * (0.12 + 0.76 * j / 8));
      if (v === null) continue;
      seen++;
      votes.set(v, (votes.get(v) || 0) + 1);
    }
    let best = null, bn = 0;
    for (const [v, n] of votes) if (n > bn) { bn = n; best = v; }
    return { value: best, confidence: seen ? bn / 81 : 0 };
  }

  // Find the board's top edge: the offset where all 25 cells read as tiles or empty cells.
  function findTop(img, s) {
    let best = BOARD_TOP_HINT * s, bestScore = -1;
    const step = Math.max(1, Math.round(4 * s));
    for (let t = Math.round(0.15 * img.height); t < 0.5 * img.height; t += step) {
      let score = 0;
      for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++) {
        // Probe each cell's centre and its top edge; background there means misaligned.
        const cx = (CELL0_X + c * PITCH + CELL / 2) * s, cy = t + (r * PITCH + CELL / 2) * s;
        const edge = px(img, cx, t + (r * PITCH + 8) * s);
        const mid = px(img, cx + 0.3 * CELL * s, cy);
        if (edge !== null && edge !== -1) score++;
        if (mid !== null && mid !== -1) score++;
        const gap = px(img, cx, t + (r * PITCH - 12) * s);
        if (gap === -1) score++;
      }
      if (score > bestScore || (score === bestScore && Math.abs(t - BOARD_TOP_HINT * s) < Math.abs(best - BOARD_TOP_HINT * s))) {
        bestScore = score; best = t;
      }
    }
    return { top: best, score: bestScore / 75 };
  }

  // Coloured blobs in the tray: one per tile of the waiting piece.
  function readTray(img, s) {
    const step = Math.max(1, Math.round(6 * s));
    const xs = [], ys = [];
    for (let x = TRAY.x0 * s; x < TRAY.x1 * s; x += step) xs.push(x);
    for (let y = TRAY.y0 * s; y < Math.min(img.height, TRAY.y1 * s); y += step) ys.push(y);
    const W = xs.length, H = ys.length, grid = new Int8Array(W * H).fill(-2);
    for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
      const v = px(img, xs[i], ys[j]);
      if (v !== null && v >= 1) grid[j * W + i] = v;
    }
    const seen = new Uint8Array(W * H), blobs = [];
    const minArea = Math.round(((CELL * 0.45) / 6) ** 2 * 0.5);
    for (let k = 0; k < W * H; k++) {
      if (grid[k] < 1 || seen[k]) continue;
      const v = grid[k], stack = [k];
      let n = 0, sx = 0, sy = 0;
      seen[k] = 1;
      while (stack.length) {
        const q = stack.pop(), qi = q % W, qj = (q - qi) / W;
        n++; sx += qi; sy += qj;
        for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const ni = qi + di, nj = qj + dj;
          if (ni < 0 || nj < 0 || ni >= W || nj >= H) continue;
          const nk = nj * W + ni;
          if (!seen[nk] && grid[nk] === v) { seen[nk] = 1; stack.push(nk); }
        }
      }
      if (n >= minArea) blobs.push({ value: v, area: n, x: sx / n, y: sy / n });
    }
    // Merge blobs of the same value (a tile split by its pips), keep the two largest values.
    const byValue = new Map();
    for (const b of blobs) byValue.set(b.value, (byValue.get(b.value) || 0) + b.area);
    const values = [...byValue.entries()].sort((a, b) => b[1] - a[1]).slice(0, 2).map(([v]) => v);
    return values;
  }

  // Main entry: img = { width, height, data: RGBA bytes }.
  function readScreen(img) {
    const s = img.width / REF_W;
    const { top, score } = findTop(img, s);
    const board = [], weak = [];
    for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++) {
      const { value, confidence } = readCell(img, s, top, r, c);
      board.push(value === null || value < 0 ? 0 : value);
      if (value === null || value < 0 || confidence < 0.4) weak.push(r * 5 + c);
    }
    const piece = readTray(img, s);
    return { board, piece, weakCells: weak, alignment: score, top: top / s };
  }

  const api = { readScreen, classify, REF_W };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.MergeReader = api;
})(this);

}).call(MC, undefined, undefined);
// ---- model ----
(function (module, require) {
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
    // Hold rule: on a max-ready board keep it ready (or score the max) whenever possible.
    const cand = [];
    for (const [a, b] of E.legalMoves(c, piece)) {
      const nb = Uint8Array.from(c);
      const r = E.apply(nb, piece, a, b, unlocked);
      cand.push({ a, b, nb, r });
    }
    let holding = MB.ready(board);
    if (holding) {
      for (const x of cand) x.keeps = x.r.points >= 70560 || MB.ready(x.nb);
      holding = cand.some((x) => x.keeps);
    }
    let best = null;
    for (const { a, b, nb, r, keeps } of cand) {
      if (holding && !keeps) continue;
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

}).call(MC, undefined, undefined);
// ---- maxboard ----
(function (module, require) {
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

}).call(MC, undefined, undefined);
// ---- main ----
// Merge Coach: main script for Scriptable. Bundled after engine.js, png.js,
// reader.js and model.js by coach/build.py (they are available as MC.*).
//
// Run from a Shortcut ("Take Screenshot" -> "Resize Image" -> "Run Script") to get a
// notification with the best move. Run inside Scriptable to check your latest screenshot.

const SETTINGS = {
  // "score": best total score (merge-coach-weights.bin).
  // "maxmerge": reach the largest possible move, 70,560, in as few turns as possible
  //             (merge-coach-maxmerge.bin); may advise starting a new game.
  objective: "score",
  weightsFile: "merge-coach-weights.bin",
  maxmergeWeightsFile: "merge-coach-maxmerge.bin",
  depth: 2,          // score mode: 1 = fast, 2 = stronger (falls back to 1 after budgetMs)
  budgetMs: 3000,
  // maxmerge mode: must match how the weights were trained (engine/goal.cpp)
  maxmerge: { beta: 0.4, power: 8, gamma: 0.9995, margin: 0.15 },
};

const B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
function base64ToBytes(s) {
  const lut = new Uint8Array(128);
  for (let i = 0; i < 64; i++) lut[B64.charCodeAt(i)] = i;
  s = s.replace(/[^A-Za-z0-9+/]/g, "");
  const out = new Uint8Array(Math.floor(s.length * 3 / 4));
  let o = 0, i = 0;
  for (; i + 4 <= s.length; i += 4) {
    const n = (lut[s.charCodeAt(i)] << 18) | (lut[s.charCodeAt(i + 1)] << 12) | (lut[s.charCodeAt(i + 2)] << 6) | lut[s.charCodeAt(i + 3)];
    out[o++] = n >> 16; out[o++] = (n >> 8) & 255; out[o++] = n & 255;
  }
  const rest = s.length - i;
  if (rest >= 2) {
    const n = (lut[s.charCodeAt(i)] << 18) | (lut[s.charCodeAt(i + 1)] << 12) | (rest === 3 ? lut[s.charCodeAt(i + 2)] << 6 : 0);
    out[o++] = n >> 16;
    if (rest === 3) out[o++] = (n >> 8) & 255;
  }
  return out.subarray(0, o);
}

async function loadWeights(name) {
  for (const fm of [FileManager.iCloud(), FileManager.local()]) {
    let path;
    try { path = fm.joinPath(fm.documentsDirectory(), name); } catch (e) { continue; }
    if (!fm.fileExists(path)) continue;
    if (fm.isFileStoredIniCloud && fm.isFileStoredIniCloud(path) && !fm.isFileDownloaded(path)) await fm.downloadFileFromiCloud(path);
    return MC.MergeModel.loadWeights(base64ToBytes(fm.read(path).toBase64String()));
  }
  return null;
}

// A group of 3+ touching equal tiles cannot exist between moves: the screenshot
// caught an animation or a misread.
function looksMidMerge(board) {
  const seen = new Uint8Array(25);
  for (let i = 0; i < 25; i++) {
    if (!board[i] || seen[i]) continue;
    const stack = [i]; let n = 0; seen[i] = 1;
    while (stack.length) {
      const q = stack.pop(); n++;
      for (const j of MC.MergeEngine.NB[q]) if (!seen[j] && board[j] === board[i]) { seen[j] = 1; stack.push(j); }
    }
    if (n >= 3) return true;
  }
  return false;
}

const KEYCAP = { 1: "1️⃣", 2: "2️⃣", 3: "3️⃣", 4: "4️⃣", 5: "5️⃣", 6: "6️⃣", 7: "Ⓜ️" };
const label = (v) => (v === 7 ? "M" : String(v));
const where = (i) => `row ${Math.floor(i / 5) + 1}, col ${(i % 5) + 1}`;

function describe(board, m) {
  const [va, vb] = m.vals;
  let title;
  if (m.b < 0) title = `${label(va)} → ${where(m.a)}`;
  else {
    const d = m.b - m.a;
    const shape = d === 1 ? `side by side, ${label(va)} on the left` : d === -1 ? `side by side, ${label(va)} on the right`
      : d === 5 ? `stacked, ${label(va)} on top` : `stacked, ${label(va)} below`;
    title = `${label(va)} → ${where(m.a)} · ${label(vb)} → ${where(m.b)}`;
    title += `\n(${shape})`;
  }
  const pts = m.points ? `+${m.points.toLocaleString("en-US")}${m.chain > 1 ? ` · ${m.chain}-step chain` : ""}` : "no merge this move";
  const grid = [];
  for (let r = 0; r < 5; r++) {
    let line = "";
    for (let c = 0; c < 5; c++) {
      const i = r * 5 + c;
      line += i === m.a ? KEYCAP[va] : i === m.b ? KEYCAP[vb] : board[i] ? "⬛" : "⬜";
    }
    grid.push(line);
  }
  return { title, body: `${pts}\n${grid.join("\n")}` };
}

// In the Scriptable app, also show what was read, to make misreads easy to spot.
let DIAG = "";
function diagnose(read) {
  const rows = [];
  for (let r = 0; r < 5; r++) rows.push(read.board.slice(r * 5, r * 5 + 5).map((v) => (v ? label(v) : "·")).join(" "));
  DIAG = `\n\nBoard I read:\n${rows.join("\n")}\nPiece: ${read.piece.map(label).join(" + ") || "none"}`;
}

async function report(title, body) {
  Script.setShortcutOutput(`${title}\n${body}`);
  if (config.runsInApp) {
    const a = new Alert();
    a.title = title; a.message = body + DIAG; a.addAction("OK");
    await a.present();
  } else {
    const n = new Notification();
    n.title = title; n.body = body;
    await n.schedule();
  }
}

async function main() {
  const t0 = Date.now();
  let image = args.images && args.images.length ? args.images[0] : null;
  if (!image && config.runsInApp) image = await Photos.latestScreenshot();
  if (!image) return report("No screenshot", "Run this from the Shortcut, or take a screenshot of the game first.");

  const img = MC.MergePNG.decode(base64ToBytes(Data.fromPNG(image).toBase64String()));
  const read = MC.MergeReader.readScreen(img);
  diagnose(read);
  if (read.alignment < 0.8) return report("Can't find the board", "This doesn't look like the game screen. Open the game and try again.");
  if (!read.piece.length) return report("Can't see the piece", "Wait until the piece is resting below the board, then try again.");
  if (read.weakCells.length || looksMidMerge(read.board)) return report("Board is moving", "The screenshot caught an animation. Try again in a moment.");

  const maxmode = SETTINGS.objective === "maxmerge";
  const file = maxmode ? SETTINGS.maxmergeWeightsFile : SETTINGS.weightsFile;
  const W = await loadWeights(file);
  if (!W) return report("Model file missing", `Put ${file} in the Scriptable folder in iCloud Drive.`);

  const unlocked = MC.MergeModel.guessUnlocked(read.board, read.piece);
  if (maxmode) {
    const g = MC.MergeModel.bestMoveGoal(W, MC.MergeMaxBoard, read.board, read.piece, unlocked, SETTINGS.maxmerge);
    const pct = `${Math.round(100 * g.progress)}% of the 70,560 setup`;
    if (g.reset) return report("Start a new game", `${g.reason === "no room" ? "This piece doesn't fit." : "A fresh game reaches the 70,560 setup sooner than this board."} (${pct})`);
    const { title, body } = describe(read.board, g);
    const status = g.points >= 70560 ? "🎯 THE MAXIMUM: 70,560" : g.ready ? "Setup complete: keep it, wait for a 2+4 domino" : pct;
    return report(title, `${status}\n${body}\n${((Date.now() - t0) / 1000).toFixed(1)} s`);
  }
  const m = MC.MergeModel.bestMove(W, read.board, read.piece, unlocked, { depth: SETTINGS.depth, budgetMs: SETTINGS.budgetMs });
  if (!m) return report("No room", "This piece doesn't fit anywhere: the game is over.");

  const { title, body } = describe(read.board, m);
  const footer = `\n${m.depth}-step search · ${((Date.now() - t0) / 1000).toFixed(1)} s`;
  return report(title, body + footer);
}

await main();
Script.complete();
