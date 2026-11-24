// Run the bundled Scriptable script in Node with stand-ins for Scriptable's API.
// usage: node coach/test/run_scriptable.js SCREENSHOT.png [--in-app]
const fs = require("fs"), path = require("path"), vm = require("vm");
const [shot, flag] = process.argv.slice(2);
const coach = path.join(__dirname, "..");
const out = [];
class Data {
  constructor(buf) { this.buf = buf; }
  static fromPNG(img) { return new Data(img.png); }
  toBase64String() { return this.buf.toString("base64"); }
}
const fm = {
  documentsDirectory: () => coach, joinPath: (a, b) => path.join(a, b), fileExists: (p) => fs.existsSync(p),
  isFileStoredIniCloud: () => false, isFileDownloaded: () => true, read: (p) => new Data(fs.readFileSync(p)),
};
const inApp = flag === "--in-app";
const image = { png: fs.readFileSync(shot) };
const sandbox = {
  console, Date, Math, Uint8Array, Int16Array, Int32Array, Float32Array, DataView, Map, String, Array, Error,
  args: { images: inApp ? [] : [image] },
  config: { runsInApp: inApp },
  Data, FileManager: { iCloud: () => fm, local: () => fm },
  Photos: { latestScreenshot: async () => image },
  Notification: class { async schedule() { out.push(["notification", this.title, this.body]); } },
  Alert: class { addAction() {} async present() { out.push(["alert", this.title, this.message]); } },
  Script: { setShortcutOutput: () => {}, complete: () => {} },
};
const src = fs.readFileSync(process.env.COACH_SCRIPT || path.join(coach, "dist", "Merge Coach.js"), "utf8");
vm.createContext(sandbox);
vm.runInContext(`(async () => {\n${src}\n})()`, sandbox).then(() => {
  for (const [kind, t, b] of out) console.log(`[${kind}] ${t}\n${b}\n`);
}, (e) => { console.error("SCRIPT ERROR:", e); process.exit(1); });
