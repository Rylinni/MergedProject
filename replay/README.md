# Replay page

Interactive replay of a recorded game, with a "play from here" mode.

```bash
engine/build/merged_eval --agent ntuple --weights W.bin --depth 2 --games 20 --record run.json
python3 replay/build.py run.json replay.html    # single self-contained HTML file
node replay/check_engine.js run.json            # JS engine must reproduce every move's points
```

`engine.js` is a port of `engine/merged.hpp` (iOS rules). `check_engine.js` replays recorded
games move by move against the C++ results and checks the six real-game cases.
