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
