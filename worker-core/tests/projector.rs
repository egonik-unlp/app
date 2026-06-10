//! Round-trips a synthetic projector.bin through the Rust parser + forward pass.
//! Guards the PFP1 byte layout (must match tools/build_snapshot.py and
//! tools/SNAPSHOT_FORMAT.md) and the feature-assembly / MLP / normalize path.

use serde_json::json;
use worker_core::project::Projector;

fn le_f32(v: &[f32], out: &mut Vec<u8>) {
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

#[test]
fn projector_roundtrip() {
    let text_dim = 4usize;
    // in_dim = text(4) + numerics(1) + acoustics(1) + flags(1) + genre_vocab(2) + album_types(1)
    let in_dim = text_dim + 1 + 1 + 1 + 2 + 1;
    let hidden = 3usize;
    let out_dim = 64usize;

    // W1=0,b1=0 -> hidden all relu(0)=0 ; W2=0 ; b2 = e0 -> output is unit vector e0
    let w1 = vec![0.0f32; hidden * in_dim];
    let b1 = vec![0.0f32; hidden];
    let w2 = vec![0.0f32; out_dim * hidden];
    let mut b2 = vec![0.0f32; out_dim];
    b2[0] = 5.0; // any magnitude; normalization should make it 1.0

    let cfg = json!({
        "text_model": "@cf/baai/bge-m3",
        "numerics": ["track_popularity"],
        "num_stats": {"track_popularity": {"mean": 30.0, "std": 20.0, "log": false}},
        "acoustics": ["af_energy"],
        "ac_stats": {"af_energy": {"mean": 0.5, "std": 0.2}},
        "genre_vocab": ["rock", "pop"],
        "album_types": ["album"]
    });
    let cfg_bytes = serde_json::to_vec(&cfg).unwrap();

    let mut buf = Vec::new();
    buf.extend_from_slice(b"PFP1");
    for v in [1u32, in_dim as u32, hidden as u32, out_dim as u32, text_dim as u32] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    le_f32(&w1, &mut buf);
    le_f32(&b1, &mut buf);
    le_f32(&w2, &mut buf);
    le_f32(&b2, &mut buf);
    buf.extend_from_slice(&(cfg_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(&cfg_bytes);

    let p = Projector::parse(&buf).expect("parse PFP1");
    assert_eq!(p.text_model(), "@cf/baai/bge-m3");

    // a track present in genre vocab + album type, with one acoustic missing
    let meta = json!({
        "track_popularity": 70,
        "af_energy": null,            // -> missing flag path
        "sp_genres": ["rock"],
        "genre_primary": "rock",
        "album_type": "album"
    });
    let text_emb = [0.1f32, 0.2, 0.3, 0.4];
    let out = p.project(&text_emb, &meta).expect("forward");

    assert_eq!(out.len(), out_dim, "output is 64-dim latent");
    let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "output is L2-normalized (got {norm})");
    assert!((out[0] - 1.0).abs() < 1e-4, "b2=e0 -> normalized unit vector e0");

    // wrong text length must error, not panic
    assert!(p.project(&[0.0, 0.0], &meta).is_err());
}
