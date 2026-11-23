# Merge Coach: setup

Double-tap the back of your iPhone while playing, and a notification tells you the best move:

```
2 → row 4, col 1 · 3 → row 4, col 2
(side by side, 2 on the left)
+162 · 3-step chain
⬜⬛⬜⬜⬛
⬛⬛⬜⬜⬛
⬛⬛⬜⬜⬜
2️⃣3️⃣⬛⬜⬛
⬜⬜⬛⬜⬜
2-step search · 0.8 s
```

Everything runs on the phone, offline. Nothing is installed except the free Scriptable app.
Rows and columns count from 1 at the top left.

## What goes where

| File (in this repo) | Goes to |
|---|---|
| `coach/dist/Merge Coach.js` | iCloud Drive → **Scriptable** folder |
| `coach/merge-coach-weights.bin` (2.3 MB) | iCloud Drive → **Scriptable** folder |

## Steps on the iPhone (about 3 minutes)

1. **Install Scriptable** (free, App Store) and open it once. This creates its iCloud Drive folder.
2. **Copy the two files** into iCloud Drive → Scriptable. Any of these works:
   - From a Mac: Finder → iCloud Drive → Scriptable (or the path in the next section).
   - From Windows with iCloud for Windows: the `iCloud Drive\Scriptable` folder.
   - AirDrop or email them to yourself, then in the Files app use Share → Save to Files → iCloud Drive → Scriptable.
   Open Scriptable: **Merge Coach** should appear in the list.
3. **Make the Shortcut.** Shortcuts app → + → add these three actions, in order:
   1. **Take Screenshot**
   2. **Resize Image**: Image = Screenshot, Width = **402** (leave Height on Auto)
   3. **Run Script** (from Scriptable): Script = **Merge Coach**. Tap "Show More" and set **Images** to *Resized Image*. Leave "Run In App" off.

   Name the shortcut **Merge Coach**.
4. **Assign Back Tap.** Settings → Accessibility → Touch → Back Tap → Double Tap → **Merge Coach**.
5. **First run.** Open the game, wait until the piece is resting below the board, and double-tap the back of the phone.
   Allow the permissions it asks for (Shortcuts running Scriptable, Scriptable notifications).

## Checking that it reads your screen

Take a normal screenshot in the game, then open Scriptable and tap **Merge Coach**. It analyses your latest
screenshot and shows the suggestion **and the board it read**. If the board doesn't match the game, send that
screenshot back and the reader can be recalibrated (`coach/reader.js`).

## Messages you may see

| Message | Meaning |
|---|---|
| Can't see the piece | The screenshot was taken while a piece was being dragged or dropped. Try again when it's resting below the board. |
| Board is moving | A merge animation was still playing. Try again a moment later. |
| Can't find the board | Not the game screen, or a different screen size from the one it was calibrated on. |
| Model file missing | `merge-coach-weights.bin` isn't in the Scriptable folder. |

## Settings

At the top of `Merge Coach.js`: `depth: 2` (stronger, under a second in tests) or `1` (instant), and
`budgetMs`, the time limit after which it falls back to the quick search.

## For a Claude session on your Mac

The repo branch `claude/stoic-hypatia-kgxlm7` has everything prebuilt. On macOS, Scriptable's iCloud
folder is `~/Library/Mobile Documents/iCloud~dk~simonbs~Scriptable/Documents/`. Copy the two files there:

```bash
cp "coach/dist/Merge Coach.js" coach/merge-coach-weights.bin ~/Library/Mobile\ Documents/iCloud~dk~simonbs~Scriptable/Documents/
```

That folder exists only after Scriptable has been opened once on a device signed into the same iCloud account.
No cable is needed; iCloud syncs the files to the phone. The Shortcut and Back Tap (steps 3 and 4) have to be
set up on the phone itself.

## How it was checked

- **Reader**: run on every frame of two real screen recordings. 20 of the 22 board changes between resting
  positions are a legal move of the piece it read, according to the engine; the other 2 are one drag that
  the game bounced back.
  It gives the same reads at 201, 300, 402 and 1206 px wide, and on colours converted to Display P3.
- **Model**: makes the same choice as the C++ agent on 98% of moves; the rest are exact or near ties.
- **Whole script**: run in Node with stand-ins for Scriptable's API, on screenshots from the recordings.
  It suggests the moves the player actually made, including the +162 chain.
- Not yet tested on a real iPhone: the Scriptable and Shortcuts parts can only be checked there.

## Rebuilding

```bash
python3 coach/make_weights.py models/main_final.bin coach/merge-coach-weights.bin
python3 coach/build.py
node coach/test/check_recording.js RECORDING.mov        # reader + engine vs a screen recording
node coach/test/check_model.js RUN.json models/main_final.bin 1   # JS model vs C++ (RUN from merged_eval --record)
node coach/test/run_scriptable.js SCREENSHOT.png        # whole script, outside the phone
```
