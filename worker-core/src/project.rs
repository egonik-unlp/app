//! Cold-start projector: parses `projector.bin` (PFP1) and maps
//! [bge_text_emb ⊕ numeric ⊕ acoustic ⊕ flags ⊕ genre_multihot ⊕ album_onehot]
//! → 64-dim song-AE latent (L2-normalized). Feature assembly mirrors
//! server/app.py ColdStartEncoder.build_input_row. See tools/SNAPSHOT_FORMAT.md.

use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;

#[derive(Deserialize)]
struct Stat {
    mean: f64,
    std: f64,
    #[serde(default)]
    log: bool,
}

#[derive(Deserialize)]
struct Cfg {
    #[serde(default)]
    text_model: String,
    numerics: Vec<String>,
    num_stats: HashMap<String, Stat>,
    acoustics: Vec<String>,
    ac_stats: HashMap<String, Stat>,
    genre_vocab: Vec<String>,
    album_types: Vec<String>,
}

pub struct Projector {
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
    text_dim: usize,
    w1: Vec<f32>, // hidden * in_dim
    b1: Vec<f32>, // hidden
    w2: Vec<f32>, // out_dim * hidden
    b2: Vec<f32>, // out_dim
    cfg: Cfg,
}

fn rf32(s: &[u8]) -> Vec<f32> {
    s.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

impl Projector {
    pub fn text_model(&self) -> &str {
        &self.cfg.text_model
    }

    pub fn parse(bytes: &[u8]) -> Result<Projector, String> {
        if bytes.len() < 24 || &bytes[0..4] != b"PFP1" {
            return Err("projector.bin: bad magic".into());
        }
        let rd = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as usize;
        let _version = rd(4);
        let in_dim = rd(8);
        let hidden = rd(12);
        let out_dim = rd(16);
        let text_dim = rd(20);
        let mut p = 24;
        let w1 = rf32(&bytes[p..p + hidden * in_dim * 4]);
        p += hidden * in_dim * 4;
        let b1 = rf32(&bytes[p..p + hidden * 4]);
        p += hidden * 4;
        let w2 = rf32(&bytes[p..p + out_dim * hidden * 4]);
        p += out_dim * hidden * 4;
        let b2 = rf32(&bytes[p..p + out_dim * 4]);
        p += out_dim * 4;
        let cfg_len = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as usize;
        p += 4;
        let cfg: Cfg = serde_json::from_slice(&bytes[p..p + cfg_len])
            .map_err(|e| format!("projector cfg json: {e}"))?;
        Ok(Projector {
            in_dim,
            hidden,
            out_dim,
            text_dim,
            w1,
            b1,
            w2,
            b2,
            cfg,
        })
    }

    fn z(&self, v: Option<f64>, st: &Stat) -> f64 {
        let val = match v {
            None => st.mean,
            Some(mut x) => {
                if st.log && x >= 0.0 {
                    x = (1.0 + x).ln();
                }
                x
            }
        };
        (val - st.mean) / st.std
    }

    /// Build the in_dim feature vector and run the MLP; returns normalized 64-dim latent.
    pub fn project(&self, text_emb: &[f32], meta: &Value) -> Result<Vec<f32>, String> {
        if text_emb.len() != self.text_dim {
            return Err(format!(
                "projector: text_emb len {} != expected {}",
                text_emb.len(),
                self.text_dim
            ));
        }
        let num_f = |k: &str| -> Option<f64> { meta.get(k).and_then(num_of) };

        let mut x: Vec<f32> = Vec::with_capacity(self.in_dim);
        x.extend_from_slice(text_emb);

        for k in &self.cfg.numerics {
            let st = self.cfg.num_stats.get(k).ok_or("missing num_stat")?;
            x.push(self.z(num_f(k), st) as f32);
        }
        // acoustic values then missing flags (two separate blocks, matching app.py)
        let mut flags: Vec<f32> = Vec::with_capacity(self.cfg.acoustics.len());
        for k in &self.cfg.acoustics {
            let st = self.cfg.ac_stats.get(k).ok_or("missing ac_stat")?;
            match num_f(k) {
                Some(v) => {
                    x.push(((v - st.mean) / st.std) as f32);
                    flags.push(0.0);
                }
                None => {
                    x.push(0.0); // (mean - mean)/std == 0
                    flags.push(1.0);
                }
            }
        }
        x.extend_from_slice(&flags);

        // genre multi-hot: sp_genres ∪ {genre_primary}
        let mut genres: Vec<String> = Vec::new();
        if let Some(arr) = meta.get("sp_genres").and_then(|v| v.as_array()) {
            for g in arr {
                if let Some(s) = g.as_str() {
                    genres.push(s.to_string());
                }
            }
        }
        if let Some(gp) = meta.get("genre_primary").and_then(|v| v.as_str()) {
            genres.push(gp.to_string());
        }
        for g in &self.cfg.genre_vocab {
            x.push(if genres.iter().any(|x| x == g) { 1.0 } else { 0.0 });
        }
        // album type one-hot
        let at = meta.get("album_type").and_then(|v| v.as_str()).unwrap_or("");
        for a in &self.cfg.album_types {
            x.push(if a == at { 1.0 } else { 0.0 });
        }

        if x.len() != self.in_dim {
            return Err(format!(
                "projector: assembled {} feats != in_dim {}",
                x.len(),
                self.in_dim
            ));
        }

        // h = relu(W1 x + b1)
        let mut h = vec![0.0f32; self.hidden];
        for o in 0..self.hidden {
            let base = o * self.in_dim;
            let mut s = self.b1[o];
            for i in 0..self.in_dim {
                s += self.w1[base + i] * x[i];
            }
            h[o] = if s > 0.0 { s } else { 0.0 };
        }
        // out = W2 h + b2
        let mut out = vec![0.0f32; self.out_dim];
        for o in 0..self.out_dim {
            let base = o * self.hidden;
            let mut s = self.b2[o];
            for i in 0..self.hidden {
                s += self.w2[base + i] * h[i];
            }
            out[o] = s;
        }
        // L2 normalize
        let norm = (out.iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-9);
        for v in out.iter_mut() {
            *v /= norm;
        }
        Ok(out)
    }
}

fn num_of(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}
