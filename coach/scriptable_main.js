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
