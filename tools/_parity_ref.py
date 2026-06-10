#!/usr/bin/env python3
"""Dump a reference A* route from the original Python pathfinder (server/app.py)
over two CORPUS tracks (no Spotify/cold-start), so the Rust/WASM port can be
checked for fidelity. Writes worker-core/tests/parity_ref.json.

Run:  server/.venv/bin/python tools/_parity_ref.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import server.app as A  # noqa: E402

LENGTH = 14
TOD = "afternoon"
SHUFFLE = False  # -> "linear"

g = A.load_graph()
scores = A.OnnxScorer(A.MODEL_DIR).score_graph(g)
model = A.transition_model()
ctx = model.resolve_context(tod=TOD, shuffle=SHUFFLE) if model else None

cases = []
# deterministic spread of start/end ROW indices
pairs = [(0, 5000), (100, 12000), (3, 20000), (250, 9000), (42, 17000)]
for sr, er in pairs:
    if sr >= len(g.ids) or er >= len(g.ids):
        continue
    start, end = g.ids[sr], g.ids[er]
    path = A.find_path(g, start, end, scores, length_hint=LENGTH,
                       max_expansions=A.MAX_EXPANSIONS, model=model, ctx=ctx)
    if path is None:
        continue
    if len(path) < LENGTH:
        path = A.densify(g, path, scores, LENGTH, model=model, ctx=ctx)
    cases.append({
        "start_row": sr, "end_row": er,
        "start_id": int(start), "end_id": int(end),
        "path_ids": [int(p) for p in path],
        "path_len": len(path),
    })

out = {"length": LENGTH, "tod": TOD, "shuffle": "linear", "cases": cases}
dst = ROOT / "worker-core" / "tests" / "parity_ref.json"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out, indent=2))
print(f"wrote {dst} with {len(cases)} cases")
for c in cases:
    print(f"  rows {c['start_row']}->{c['end_row']}: {c['path_len']} nodes")
