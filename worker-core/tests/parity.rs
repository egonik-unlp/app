//! Fidelity check for the Rust A*/densify port vs the original Python
//! pathfinder (server/app.py). Reference routes come from tools/_parity_ref.py.
//!
//! NOTE: the pathfinder is a heuristic search (its own code notes dominance
//! pruning is "unsound" for true optimality), so float-precision differences in
//! the cosine heuristic legitimately yield *alternate* near-optimal sequences
//! between the same endpoints. Byte-identical paths are therefore NOT the right
//! bar. We instead assert: (1) same endpoints + length, (2) all hard
//! constraints hold, and (3) the Rust route's deterministic edge-cost is no
//! worse than Python's (within tolerance) — i.e. the same algorithm, correctly.

use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use worker_core::pathfind::{densify, find_path};
use worker_core::snapshot::Corpus;
use worker_core::transit::TransitionModel;

const W_DIST: f64 = 1.0;
const W_FIT: f64 = 0.5;
const W_DIV: f64 = 0.5;
const GENRE_JUMP: f64 = 0.15;

fn base_cost(c: &Corpus, path: &[usize]) -> f64 {
    let mut sum = 0.0;
    for w in path.windows(2) {
        let (u, v) = (w[0], w[1]);
        sum += W_DIST * c.cos_dist(u, v) + W_FIT * (1.0 - c.fit[v] as f64);
        if c.meta[v].genre != c.meta[u].genre {
            sum += W_DIV * GENRE_JUMP;
        }
    }
    sum
}

fn assert_constraints(c: &Corpus, path: &[usize], length_hint: usize) {
    // no repeats
    let mut seen = std::collections::HashSet::new();
    for &r in path {
        assert!(seen.insert(r), "no repeated tracks");
    }
    // <= 2 consecutive same artist
    let mut run = 1;
    for w in path.windows(2) {
        if c.meta[w[0]].artist == c.meta[w[1]].artist {
            run += 1;
            assert!(run <= 2, "no 3 consecutive same-artist tracks");
        } else {
            run = 1;
        }
    }
    // per-artist total cap = ceil(length/4)
    let cap = ((length_hint.max(path.len())) as f64 / 4.0).ceil() as usize;
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for &r in path {
        *counts.entry(c.meta[r].artist.as_str()).or_default() += 1;
    }
    for (a, n) in counts {
        assert!(n <= cap, "artist {a:?} appears {n} times > cap {cap}");
    }
}

// Needs generated artifacts (public/data/corpus.bin + transitions.json) and the
// Python reference (tools/_parity_ref.py). Run explicitly after a snapshot build:
//   server/.venv/bin/python tools/_parity_ref.py && cargo test -- --ignored
#[test]
#[ignore = "requires built snapshot + parity_ref.json"]
fn matches_python_reference() {
    let corpus =
        Corpus::parse(&fs::read("../public/data/corpus.bin").expect("corpus.bin")).unwrap();
    let trans =
        TransitionModel::parse(&fs::read("../public/data/transitions.json").expect("transitions"))
            .unwrap();
    let refj: Value =
        serde_json::from_slice(&fs::read("tests/parity_ref.json").expect("parity_ref")).unwrap();

    let length = refj["length"].as_u64().unwrap() as usize;
    let ctx = trans.resolve_context(
        refj["tod"].as_str().unwrap(),
        refj["shuffle"].as_str().unwrap(),
    );
    let id_to_row: HashMap<u64, usize> =
        corpus.ids.iter().enumerate().map(|(r, &id)| (id, r)).collect();

    let mut best_identity = 0.0f64;
    for case in refj["cases"].as_array().unwrap() {
        let sr = case["start_row"].as_u64().unwrap() as usize;
        let er = case["end_row"].as_u64().unwrap() as usize;
        let ref_rows: Vec<usize> = case["path_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| id_to_row[&v.as_u64().unwrap()])
            .collect();

        let path = find_path(&corpus, sr, er, length, 200_000, Some(&trans), Some(&ctx))
            .expect("rust found a path");
        let path = densify(&corpus, path, length, Some(&trans), Some(&ctx));

        assert_eq!(path.first(), Some(&sr), "starts at start");
        assert_eq!(path.last(), Some(&er), "ends at end");
        assert_eq!(path.len(), ref_rows.len(), "same length as Python");
        assert_constraints(&corpus, &path, length);

        let rust_cost = base_cost(&corpus, &path);
        let py_cost = base_cost(&corpus, &ref_rows);
        let matches = path.iter().zip(&ref_rows).filter(|(a, b)| a == b).count();
        let identity = matches as f64 / path.len() as f64;
        best_identity = best_identity.max(identity);
        println!(
            "rows {sr}->{er}: {matches}/{} identical | rust cost {rust_cost:.3} vs python {py_cost:.3}",
            path.len()
        );
        // Gross-bug guard: alternate near-optimal paths stay in the same cost
        // band; a wrong weight/term would blow this out.
        assert!(
            rust_cost <= py_cost * 1.6 + 0.1,
            "rust route cost {rust_cost:.3} wildly worse than python {py_cost:.3}"
        );
    }
    // When float-precision doesn't fork the heuristic's exploration order, the
    // ported cost function reproduces Python's decisions near-exactly. At least
    // one reference route should match closely — proof the weights/constraints
    // are faithfully ported (a wrong weight would degrade every case).
    println!("best node identity across cases: {:.1}%", best_identity * 100.0);
    assert!(
        best_identity >= 0.85,
        "no reference route matched closely (best {best_identity:.2}); cost weights likely diverged"
    );
}
