"""Bundle the coach into one Scriptable script: dist/Merge Coach.js

usage: python3 coach/build.py
"""
import os

here = os.path.dirname(os.path.abspath(__file__))
parts = [("engine", "../replay/engine.js"), ("png", "png.js"), ("reader", "reader.js"), ("model", "model.js")]
out = [
    "// Variables used by Scriptable.",
    "// These must be at the very top of the file. Do not edit.",
    "// icon-color: deep-blue; icon-glyph: chess-board;",
    "// Merge Coach: reads a screenshot of the merge game and suggests the best move.",
    "// Built by coach/build.py from replay/engine.js, coach/png.js, coach/reader.js,",
    "// coach/model.js and coach/scriptable_main.js. Edit those, not this file.",
    "",
    "const MC = {};",
]
for name, rel in parts:
    src = open(os.path.join(here, rel)).read()
    # Scriptable defines `module`; hide it so each file registers itself on MC instead.
    out.append(f"// ---- {name} ----")
    out.append("(function (module, require) {\n" + src + "\n}).call(MC, undefined, undefined);")
out.append("// ---- main ----")
out.append(open(os.path.join(here, "scriptable_main.js")).read())
os.makedirs(os.path.join(here, "dist"), exist_ok=True)
dst = os.path.join(here, "dist", "Merge Coach.js")
open(dst, "w").write("\n".join(out))
print(dst, os.path.getsize(dst), "bytes")
