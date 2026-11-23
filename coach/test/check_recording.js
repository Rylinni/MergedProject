// Read every frame of a screen recording and check that each change of the board
// is explained by a legal placement of the piece seen in the tray (using the engine).
// usage: node coach/test/check_recording.js VIDEO [fps] [width]
const { spawn } = require("child_process");
const path = require("path");
const R = require("../reader.js");
const E = require("../../replay/engine.js");

const [video, fpsArg, widthArg] = process.argv.slice(2);
const fps = +(fpsArg || 4), width = +(widthArg || 402), height = Math.round(width * 2622 / 1206);
const frameBytes = width * height * 4;
const ff = spawn("ffmpeg", ["-v", "error", "-i", video, "-vf", `fps=${fps},scale=${width}:${height}`, "-f", "rawvideo", "-pix_fmt", "rgba", "-"]);
let buf = Buffer.alloc(0), n = 0;
const show = (b) => b.map((v) => (v === 0 ? "." : v === 7 ? "M" : v)).join("").replace(/(.{5})(?!$)/g, "$1/");
let prevBoard = null, stable = null, lastPiece = null, pieceAtStable = null, changes = 0, explained = 0, unexplained = [];
let weakFrames = 0;
function frame(data) {
  const t = (n++ / fps).toFixed(2);
  const r = R.readScreen({ width, height, data });
  if (r.weakCells.length) weakFrames++;
  if (r.piece.length) lastPiece = r.piece;
  const key = show(r.board);
  // Only trust idle frames: piece resting in the tray, every cell clearly read,
  // and the same board as the previous frame (no animation in progress).
  if (prevBoard === key && r.weakCells.length === 0 && r.piece.length) {
    if (stable === null) { stable = r.board; pieceAtStable = lastPiece; console.log(`${t}s  ${key}  piece=${lastPiece}`); }
    else if (show(stable) !== key) {
      changes++;
      const ok = explains(stable, r.board, pieceAtStable);
      if (ok) explained++; else unexplained.push(`${t}s ${show(stable)} -> ${key} piece=${pieceAtStable}`);
      console.log(`${t}s  ${key}  piece=${lastPiece}  ${ok ? "explained by " + ok : "NOT EXPLAINED"}`);
      stable = r.board; pieceAtStable = lastPiece;
    }
  }
  prevBoard = key;
}
function explains(before, after, piece) {
  if (!piece || !piece.length) return null;
  const pieces = piece.length === 1 ? [[piece[0]]] : [[piece[0], piece[1]], [piece[1], piece[0]]];
  for (const vals of pieces) for (const [a, b] of E.legalMoves(Uint8Array.from(before), vals)) {
    const c = Uint8Array.from(before);
    const res = E.apply(c, vals, a, b, 254);
    if (show(Array.from(c)) === show(after)) return `${vals.join("+")}@${a}${b >= 0 ? "," + b : ""} (+${res.points})`;
  }
  return null;
}
ff.stdout.on("data", (d) => {
  buf = Buffer.concat([buf, d]);
  while (buf.length >= frameBytes) { frame(buf.subarray(0, frameBytes)); buf = buf.subarray(frameBytes); }
});
ff.on("close", () => {
  console.log(`\n${path.basename(video)}: ${n} frames, ${changes} board changes, ${explained} explained by a legal move of the tray piece`);
  if (unexplained.length) console.log("unexplained:\n  " + unexplained.join("\n  "));
  console.log(`frames with uncertain cells (mid-animation): ${weakFrames}`);
});
