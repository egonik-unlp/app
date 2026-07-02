//! Coverage for Bring-Your-Own-Playlist ordering: `order_fixed` re-orders a
//! *fixed* set of tracks into the lowest-cost open path without adding or
//! dropping any. The unit test uses synthetic nodes (no built artifacts); the
//! response-shape test drives the WASM `arrange` entry point over the real
//! snapshot and asserts the output is a permutation of the input.

use std::collections::HashSet;
use worker_core::pathfind::{order_fixed, OrderNode};
use worker_core::snapshot::Meta;

fn unit(angle: f32) -> Vec<f32> {
    let mut v = vec![angle.cos(), angle.sin(), 0.0, 0.0];
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut v {
        *x /= n;
    }
    v
}

fn meta(tag: usize) -> Meta {
    // Unique artist/genre per node keeps those terms constant across every hop,
    // so ordering is driven by cosine distance — the chain manifold's optimum.
    Meta {
        uri: format!("spotify:track:t{tag}"),
        name: format!("t{tag}"),
        artist: format!("a{tag}"),
        album: None,
        genre: format!("g{tag}"),
    }
}

// `n` unit vectors fanned across a quarter circle: the lowest-cost open path
// visits them in angle order, so the optimal ordering is monotonic.
fn chain(n: usize) -> (Vec<Vec<f32>>, Vec<f32>, Vec<Meta>) {
    let mut vecs = Vec::with_capacity(n);
    let mut angles = Vec::with_capacity(n);
    let mut metas = Vec::with_capacity(n);
    for i in 0..n {
        let ang = (i as f32) * std::f32::consts::FRAC_PI_2 / ((n.max(2) - 1) as f32);
        vecs.push(unit(ang));
        angles.push(ang);
        metas.push(meta(i));
    }
    (vecs, angles, metas)
}

#[test]
fn order_fixed_sorts_a_shuffled_chain() {
    let n = 9;
    let (vecs, angles, metas) = chain(n);
    // Present the nodes to order_fixed in a scrambled order.
    let shuffled = [4usize, 0, 7, 2, 8, 1, 5, 3, 6];
    let sv: Vec<Vec<f32>> = shuffled.iter().map(|&i| vecs[i].clone()).collect();
    let sm: Vec<Meta> = shuffled.iter().map(|&i| metas[i].clone()).collect();
    let sa: Vec<f32> = shuffled.iter().map(|&i| angles[i]).collect();

    let nodes: Vec<OrderNode> = (0..n)
        .map(|i| OrderNode {
            vec: &sv[i],
            fit: 0.5,
            meta: &sm[i],
        })
        .collect();
    let perm = order_fixed(&nodes, None, None);

    // Set preservation: a permutation of 0..n — nothing added or dropped.
    assert_eq!(perm.len(), n);
    let mut sorted = perm.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..n).collect::<Vec<_>>(), "perm covers every input exactly once");

    // The recovered angle sequence is monotonic (ascending or descending), i.e.
    // the chain was reassembled into its lowest-cost open path.
    let seq: Vec<f32> = perm.iter().map(|&i| sa[i]).collect();
    let asc = seq.windows(2).all(|w| w[0] <= w[1] + 1e-6);
    let desc = seq.windows(2).all(|w| w[0] >= w[1] - 1e-6);
    assert!(asc || desc, "ordering is monotonic along the chain: {seq:?}");
}

#[test]
fn order_fixed_trivial_sets_pass_through() {
    let (vecs, _a, metas) = chain(2);
    let nodes: Vec<OrderNode> = (0..2)
        .map(|i| OrderNode {
            vec: &vecs[i],
            fit: 0.5,
            meta: &metas[i],
        })
        .collect();
    assert_eq!(order_fixed(&nodes, None, None), vec![0, 1]);
}

// Drives the WASM `arrange` entry point over the real snapshot: a handful of
// corpus rows arrange into a path that is a permutation of exactly those tracks,
// with only "path" edges. Run after a snapshot build:
//   cargo test --test arrange -- --ignored
#[test]
#[ignore = "requires built snapshot (public/data/corpus.bin)"]
fn arrange_response_is_a_permutation_of_input() {
    use worker_core::{arrange, load_corpus, meta_at};
    let corpus = std::fs::read("../public/data/corpus.bin").expect("read corpus.bin");
    load_corpus(&corpus, &[]).unwrap_or_else(|_| panic!("load_corpus failed"));

    let input_rows = [10usize, 500, 1200, 3000, 8000, 15000];
    let want: HashSet<String> = input_rows
        .iter()
        .map(|&r| {
            let m: serde_json::Value = serde_json::from_str(&meta_at(r).unwrap()).unwrap();
            m["uri"].as_str().unwrap().to_string()
        })
        .collect();

    let tracks: Vec<serde_json::Value> = input_rows.iter().map(|&r| serde_json::json!({"row": r})).collect();
    let out = arrange(&serde_json::to_string(&tracks).unwrap(), "any", "any")
        .unwrap_or_else(|_| panic!("arrange failed"));
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();

    let path = v["path"].as_array().unwrap();
    assert_eq!(path.len(), input_rows.len(), "path keeps every input track");
    let got: HashSet<String> = path
        .iter()
        .map(|n| n["uri"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(got, want, "path is exactly the input set, re-ordered");
    for e in v["edges"].as_array().unwrap() {
        assert_eq!(e["kind"].as_str().unwrap(), "path", "arrange emits only path edges");
    }
}
