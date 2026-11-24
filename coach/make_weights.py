"""Convert n-tuple weights (engine "MNT1" float32) to the phone format "MNQ1" (int16 x scale).

usage: python3 coach/make_weights.py SRC.bin DST.bin [SCALE]
SCALE defaults to 0.5 (points models). Pass "auto" for models whose values are small
(e.g. the max-merge goal model): the scale is then set from the largest weight.
"""
import struct, sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
scale_arg = sys.argv[3] if len(sys.argv) > 3 else "0.5"
raw = open(src, "rb").read()
magic, ntuples = struct.unpack("<II", raw[:8])
assert magic == 0x4D4E5431, "not an MNT1 file"
w = np.frombuffer(raw[8:], dtype=np.float32)
scale = float(np.abs(w).max()) / 32000 if scale_arg == "auto" else float(scale_arg)
q = np.clip(np.round(w / scale), -32768, 32767).astype("<i2")
clipped = int(np.sum(np.abs(w / scale) > 32767))
with open(dst, "wb") as f:
    f.write(struct.pack("<IIf", 0x4D4E5131, ntuples, scale))
    f.write(q.tobytes())
print(f"{dst}: {len(q)} weights, scale {scale:.3g}, max rounding error {scale/2:.3g}, clipped {clipped}, {12 + 2*len(q)} bytes")
