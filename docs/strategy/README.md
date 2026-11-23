# Strategy playbook

- `playbook.html`: the human-readable field guide (published as an artifact). Open it in a browser.
- `analysis-report.md`: the analysis it is based on, with all the numbers and sample sizes.
- `verify_examples.js`: replays the six example positions with `replay/engine.js` and checks the points
  (`node docs/strategy/verify_examples.js`).
- `analysis-scripts/`: the node scripts that produced the statistics. They read recorded games from a
  scratch directory (absolute paths inside); regenerate games with `engine/build/merged_eval --record`.
