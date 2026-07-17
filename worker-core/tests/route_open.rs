//! Coverage for off-map ("open") routing: tracks that aren't in the corpus are
//! routed as *real* A* endpoints (threaded through their own nearest corpus
//! neighbors) instead of being snapped to a single anchor. The corpus-only
//! tests use a tiny synthetic corpus (no built artifacts needed); the response
//! -shape test drives the WASM entry point over the real snapshot.

use std::collections::HashSet;
use worker_core::pathfind::{find_path, find_path_open, EndSpec};
use worker_core::snapshot::{Corpus, Meta};

fn normalize(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

fn unit(angle: f32) -> [f32; 4] {
    let mut v = [angle.cos(), angle.sin(), 0.0, 0.0];
    normalize(&mut v);
    v
}

fn off_meta(tag: &str) -> Meta {
    Meta {
        uri: format!("spotify:track:off-{tag}"),
        name: format!("off {tag}"),
        artist: format!("off-artist-{tag}"),
        album: None,
        genre: format!("off-genre-{tag}"),
        release_year: None,
    }
}

/// A chain manifold: `n` unit vectors fanned across a quarter circle, so each
/// track's nearest neighbors are its index-adjacent tracks. Unique artist/genre
/// per row keeps the artist-share/consecutive constraints out of the way.
fn toy_corpus(n: usize, k: usize) -> Corpus {
    let dim = 4usize;
    let mut vecs = vec![0.0f32; n * dim];
    for i in 0..n {
        let ang = (i as f32) * std::f32::consts::FRAC_PI_2 / ((n.max(2) - 1) as f32);
        vecs[i * dim..i * dim + dim].copy_from_slice(&unit(ang));
    }
    let mut nbr_idx = vec![u32::MAX; n * k];
    let mut nbr_dist = vec![0.0f32; n * k];
    for i in 0..n {
        let vi = &vecs[i * dim..i * dim + dim];
        let mut others: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let vj = &vecs[j * dim..j * dim + dim];
                let dot: f64 = (0..dim).map(|d| (vi[d] * vj[d]) as f64).sum();
                (1.0 - dot, j)
            })
            .collect();
        others.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (slot, &(d, j)) in others.iter().take(k).enumerate() {
            nbr_idx[i * k + slot] = j as u32;
            nbr_dist[i * k + slot] = d as f32;
        }
    }
    let meta: Vec<Meta> = (0..n)
        .map(|i| Meta {
            uri: format!("spotify:track:t{i}"),
            name: format!("track{i}"),
            artist: format!("artist{i}"),
            album: None,
            genre: format!("genre{i}"),
            release_year: None,
        })
        .collect();
    let mut uri_to_row = std::collections::HashMap::new();
    for (r, m) in meta.iter().enumerate() {
        uri_to_row.insert(m.uri.clone(), r);
    }
    Corpus {
        n,
        dim,
        k,
        ids: (0..n as u64).collect(),
        vecs,
        fit: vec![0.5; n],
        nbr_idx,
        nbr_dist,
        meta,
        uri_to_row,
    }
}

#[test]
fn knn_vec_top1_matches_snap() {
    let c = toy_corpus(8, 3);
    let q = unit(2.3 * std::f32::consts::FRAC_PI_2 / 7.0);
    let knn = c.knn_vec(&q, 4);
    assert_eq!(knn.len(), 4);
    assert_eq!(knn[0].0, c.snap(&q), "knn_vec top-1 row == snap(v)");
    for w in knn.windows(2) {
        assert!(w[0].1 <= w[1].1 + 1e-9, "distances ascending");
    }
}

#[test]
fn corpus_corpus_route_still_valid() {
    let c = toy_corpus(8, 3);
    let path = find_path(&c, 0, 7, 8, 10_000, None, None).expect("corpus path");
    assert_eq!(path.first(), Some(&0), "starts at start row");
    assert_eq!(path.last(), Some(&7), "ends at end row");
    let mut seen = HashSet::new();
    for &r in &path {
        assert!(r < c.n, "corpus rows only");
        assert!(seen.insert(r), "no repeats");
    }
    // find_path delegates to find_path_open(Corpus, Corpus): same result.
    let open = find_path_open(
        &c,
        &EndSpec::Corpus(0),
        &EndSpec::Corpus(7),
        8,
        10_000,
        None,
        None,
    )
    .expect("open path");
    assert_eq!(path, open, "corpus↔corpus is identical via either entry point");
}

#[test]
fn offmap_endpoints_route_through_own_neighbors() {
    let c = toy_corpus(10, 3);
    // an off-map start just *before* row 0, and an off-map end just *after* the
    // last row — neither is in the corpus, but each sits next to the chain.
    let sv = unit(-0.15);
    let ev = unit(std::f32::consts::FRAC_PI_2 + 0.15);
    let s_knn = c.knn_vec(&sv, c.k);
    let e_knn = c.knn_vec(&ev, c.k);
    let start = EndSpec::Virtual {
        vec: &sv,
        fit: 0.5,
        meta: off_meta("start"),
        knn: s_knn.clone(),
    };
    let end = EndSpec::Virtual {
        vec: &ev,
        fit: 0.5,
        meta: off_meta("end"),
        knn: e_knn.clone(),
    };
    let path = find_path_open(&c, &start, &end, 8, 10_000, None, None).expect("open path");

    let s_set: HashSet<usize> = s_knn.iter().map(|&(r, _)| r).collect();
    let e_set: HashSet<usize> = e_knn.iter().map(|&(r, _)| r).collect();
    // the returned path is corpus-only; its first/last rows are the off-map
    // endpoints' own neighbors (the real first/closing hops), not a single anchor.
    assert!(!path.is_empty());
    let mut seen = HashSet::new();
    for &r in &path {
        assert!(r < c.n, "corpus rows only — no virtual sentinel leaked into the path");
        assert!(seen.insert(r), "no repeats");
    }
    assert!(
        s_set.contains(path.first().unwrap()),
        "first hop comes from the off-map start's own KNN"
    );
    assert!(
        e_set.contains(path.last().unwrap()),
        "closing hop reaches the off-map end's own KNN"
    );
}

// Drives the WASM entry point over the real snapshot with a synthetic off-map
// vector (no Spotify/AI needed): the path's first node IS the off-map track and
// the route carries no `snap` edge. Run after a snapshot build:
//   cargo test --test route_open -- --ignored
#[test]
#[ignore = "requires built snapshot (public/data/corpus.bin)"]
fn route_open_response_starts_at_offmap_track() {
    use worker_core::{load_corpus, route_open, vec_at};
    let corpus = std::fs::read("../public/data/corpus.bin").expect("read corpus.bin");
    load_corpus(&corpus, &[]).unwrap_or_else(|_| panic!("load_corpus failed"));

    // off-map start = corpus row 0 nudged off-lattice and renormalized.
    let v0 = vec_at(0).unwrap_or_else(|_| panic!("vec_at(0) failed"));
    let mut sv: Vec<f32> = v0.clone();
    sv[0] += 0.02;
    let n: f32 = sv.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in sv.iter_mut() {
        *x /= n;
    }
    let end_vec = vec_at(1000).unwrap_or_else(|_| panic!("vec_at(1000) failed"));
    let meta = r#"{"uri":"spotify:track:offmap","name":"Off","artist":"Nobody","album":null,"genre":"unknown"}"#;

    let out = route_open(-1, &sv, 0.5, meta, 1000, &end_vec, 0.5, "", 12, "any", "any")
        .unwrap_or_else(|_| panic!("route_open failed"));
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();

    let path = v["path"].as_array().unwrap();
    assert!(path.len() >= 3, "path threads through ≥1 corpus stop");
    assert_eq!(
        path[0]["id"].as_str().unwrap(),
        "spotify:track:offmap",
        "path[0] IS the off-map track, not an anchor"
    );
    assert_eq!(path[0]["uri"].as_str().unwrap(), "spotify:track:offmap");
    for e in v["edges"].as_array().unwrap() {
        assert_eq!(
            e["kind"].as_str().unwrap(),
            "path",
            "route_open emits no snap edges"
        );
    }
}
