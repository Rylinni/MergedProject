# Trained n-tuple weights

| File | Trained with | Use with `merged_eval` |
|---|---|---|
| `main_checkpoint.bin` | `--objective sum` (expected total score), ~1.4M self-play games; a checkpoint of the run still in progress | `--agent ntuple --weights models/main_checkpoint.bin` |
| `bigmerge_convex5000.bin` | started from the main checkpoint, then 60k games with `--objective convex:5000` (reward = points + points²/5000) | `--agent ntuple --weights models/bigmerge_convex5000.bin --objective convex:5000` |

Always evaluate with the same `--objective` the weights were trained with.
