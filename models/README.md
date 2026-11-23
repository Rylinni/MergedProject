# Trained n-tuple weights

| File | Trained with | Use with `merged_eval` |
|---|---|---|
| `main_final.bin` | `--objective sum` (expected total score), ~1.8M self-play games. 1-step search: mean 25,765 over 2,000 games | `--agent ntuple --weights models/main_final.bin` |
| `main_checkpoint.bin` | an earlier checkpoint of the same run (~1.4M games); the big-merge models start from it | `--agent ntuple --weights models/main_checkpoint.bin` |
| `bigmerge_convex5000.bin` | started from the main checkpoint, then 60k games with `--objective convex:5000` (reward = points + points²/5000) | `--agent ntuple --weights models/bigmerge_convex5000.bin --objective convex:5000` |

Always evaluate with the same `--objective` the weights were trained with.
