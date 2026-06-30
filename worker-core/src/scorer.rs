//! Live rotation-fit scorer: parses `scorer.bin` (PFS1) and computes the raw
//! model score for a 64-dim latent + track metadata, so off-corpus (cold-start)
//! tracks get a real fit value instead of `null`. A 1:1 port of the offline
//! ONNX path — feature assembly mirrors server/app.py OnnxScorer.features and
//! the network mirrors the burn-deep-embeddings-highvocab graph:
//!
//!   cont(71) = standardize(pca64 ++ numeric7)
//!   e_genre/e_artist/e_album = onehot_block @ embed_w   (ONNX MatMul, [in,out])
//!   h = concat(cont, e_genre, e_artist, e_album)         -> 107
//!   h = relu(h @ trunk_w0 + b0); relu(h @ trunk_w1 + b1)
//!   raw = softmax(h @ head_w + head_b)[class_index]
//!
//! Returns the RAW score (after nan_to_num + clip(0,1)); the Worker rescales it
//! onto the baked [0,1] fit scale via the corpus bounds in manifest.json.
//! See tools/export_scorer.py and tools/SNAPSHOT_FORMAT.md.

use serde::Deserialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};

#[derive(Deserialize)]
struct Column {
    op: String,
    #[serde(default)]
    component: Option<usize>,
    #[serde(default)]
    field: Option<String>,
    #[serde(default)]
    group: Option<String>,
    #[serde(default)]
    value: Option<String>,
}

#[derive(Deserialize)]
struct Blocks {
    cont: usize,
    genre: usize,
    artist: usize,
    album: usize,
}

#[derive(Deserialize)]
struct Embed {
    genre: usize,
    artist: usize,
    album: usize,
}

#[derive(Deserialize)]
struct BlobSpec {
    name: String,
    shape: Vec<usize>,
}

#[derive(Deserialize)]
struct Cfg {
    blocks: Blocks,
    embed: Embed,
    trunk: Vec<usize>,
    n_class: usize,
    class_index: usize,
    pca_dims: usize,
    columns: Vec<Column>,
    blobs: Vec<BlobSpec>,
}

pub struct Scorer {
    cfg: Cfg,
    blobs: HashMap<String, Vec<f32>>,
    /// per onehot group: the set of explicit values (excludes the `__other__` catch-all)
    group_values: HashMap<String, HashSet<String>>,
}

fn rf32(s: &[u8]) -> Vec<f32> {
    s.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn num_of(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

/// matches Python `meta.get(field) not in (None, "")` (present == not absent / null / "")
fn is_missing(meta: &Value, field: &str) -> bool {
    match meta.get(field) {
        None | Some(Value::Null) => true,
        Some(Value::String(s)) => s.is_empty(),
        _ => false,
    }
}

/// matches Python `float(meta.get(field) or 0.0)`
fn num_or_zero(meta: &Value, field: &str) -> f64 {
    meta.get(field).and_then(num_of).unwrap_or(0.0)
}

impl Scorer {
    pub fn parse(bytes: &[u8]) -> Result<Scorer, String> {
        if bytes.len() < 12 || &bytes[0..4] != b"PFS1" {
            return Err("scorer.bin: bad magic".into());
        }
        let _version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let cfg_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let mut p = 12;
        let cfg: Cfg = serde_json::from_slice(&bytes[p..p + cfg_len])
            .map_err(|e| format!("scorer cfg json: {e}"))?;
        p += cfg_len;

        let mut blobs = HashMap::new();
        for b in &cfg.blobs {
            let count: usize = b.shape.iter().product();
            let need = count * 4;
            if p + need > bytes.len() {
                return Err(format!("scorer.bin: truncated blob {}", b.name));
            }
            blobs.insert(b.name.clone(), rf32(&bytes[p..p + need]));
            p += need;
        }

        let mut group_values: HashMap<String, HashSet<String>> = HashMap::new();
        for c in &cfg.columns {
            if c.op == "onehot" {
                if let (Some(g), Some(v)) = (&c.group, &c.value) {
                    if v != "__other__" {
                        group_values.entry(g.clone()).or_default().insert(v.clone());
                    }
                }
            }
        }
        let s = Scorer { cfg, blobs, group_values };
        s.validate()?;
        Ok(s)
    }

    fn blob(&self, name: &str) -> Result<&[f32], String> {
        self.blobs
            .get(name)
            .map(|v| v.as_slice())
            .ok_or_else(|| format!("scorer.bin: missing blob {name}"))
    }

    fn validate(&self) -> Result<(), String> {
        let b = &self.cfg.blocks;
        let n_cols = b.cont + b.genre + b.artist + b.album;
        if n_cols != self.cfg.columns.len() {
            return Err(format!(
                "scorer: blocks sum {} != columns {}",
                n_cols,
                self.cfg.columns.len()
            ));
        }
        if self.cfg.trunk.len() != 2 {
            return Err("scorer: expected 2 trunk layers".into());
        }
        // probe required blobs early so a malformed file fails at load, not at first score
        for name in [
            "pca_components", "pca_mean", "cont_mean", "cont_std", "embed_genre",
            "embed_artist", "embed_album", "trunk_w0", "trunk_b0", "trunk_w1",
            "trunk_b1", "head_w", "head_b",
        ] {
            self.blob(name)?;
        }
        Ok(())
    }

    /// PCA: pca[c] = Σ_k (latent[k] - mean[k]) * components[c, k]  (components is [dims, 64], row-major)
    fn pca(&self, latent: &[f32]) -> Result<Vec<f32>, String> {
        let dims = self.cfg.pca_dims;
        if latent.len() != dims {
            return Err(format!("scorer: latent len {} != pca dims {}", latent.len(), dims));
        }
        let comp = self.blob("pca_components")?;
        let mean = self.blob("pca_mean")?;
        let mut out = vec![0.0f32; dims];
        for c in 0..dims {
            let base = c * dims;
            let mut s = 0.0f32;
            for k in 0..dims {
                s += (latent[k] - mean[k]) * comp[base + k];
            }
            out[c] = s;
        }
        Ok(out)
    }

    /// Assemble the raw 497-feature vector — verbatim port of OnnxScorer.features.
    pub fn featurize(&self, latent: &[f32], meta: &Value) -> Result<Vec<f32>, String> {
        let pca = self.pca(latent)?;
        let mut x: Vec<f32> = Vec::with_capacity(self.cfg.columns.len());
        for col in &self.cfg.columns {
            let v: f32 = match col.op.as_str() {
                "pca" => {
                    let c = col.component.ok_or("pca column missing component")?;
                    *pca.get(c).ok_or("pca component out of range")?
                }
                "numeric_present_raw" | "numeric_verbatim" => {
                    let f = col.field.as_deref().ok_or("numeric column missing field")?;
                    num_or_zero(meta, f) as f32
                }
                "numeric_missing_flag" => {
                    let f = col.field.as_deref().ok_or("flag column missing field")?;
                    if is_missing(meta, f) { 1.0 } else { 0.0 }
                }
                "numeric_log1p" => {
                    let f = col.field.as_deref().ok_or("log1p column missing field")?;
                    if is_missing(meta, f) {
                        0.0
                    } else {
                        match meta.get(f).and_then(num_of) {
                            Some(x) if x >= 0.0 => (1.0 + x).ln() as f32,
                            _ => 0.0,
                        }
                    }
                }
                "onehot" => {
                    let g = col.group.as_deref().ok_or("onehot column missing group")?;
                    let val = col.value.as_deref().ok_or("onehot column missing value")?;
                    let item = meta.get(g).and_then(|v| v.as_str());
                    let fire = if val == "__other__" {
                        match item {
                            Some(s) => !self
                                .group_values
                                .get(g)
                                .map(|set| set.contains(s))
                                .unwrap_or(false),
                            None => true,
                        }
                    } else {
                        item == Some(val)
                    };
                    if fire { 1.0 } else { 0.0 }
                }
                _ => 0.0,
            };
            x.push(v);
        }
        Ok(x)
    }

    /// y[j] = b[j] + Σ_k x[k] * W[k*out + j]   (W is [in, out] row-major, ONNX X@W; bias optional)
    fn affine(x: &[f32], w: &[f32], bias: Option<&[f32]>, out: usize) -> Vec<f32> {
        let inn = x.len();
        let mut y = vec![0.0f32; out];
        for j in 0..out {
            let mut s = bias.map(|b| b[j]).unwrap_or(0.0);
            for k in 0..inn {
                s += x[k] * w[k * out + j];
            }
            y[j] = s;
        }
        y
    }

    /// Run the network over an assembled 497-vector; returns the raw class score
    /// (after nan_to_num(0.5/1/0) + clip(0,1)).
    pub fn forward(&self, feats: &[f32]) -> Result<f32, String> {
        let b = &self.cfg.blocks;
        let (o_gen, o_art, o_alb) = (b.cont, b.cont + b.genre, b.cont + b.genre + b.artist);

        let cont_mean = self.blob("cont_mean")?;
        let cont_std = self.blob("cont_std")?;
        let mut h: Vec<f32> = Vec::with_capacity(b.cont + self.cfg.embed.genre + self.cfg.embed.artist + self.cfg.embed.album);
        for i in 0..b.cont {
            h.push((feats[i] - cont_mean[i]) / cont_std[i]);
        }
        h.extend(Self::affine(&feats[o_gen..o_art], self.blob("embed_genre")?, None, self.cfg.embed.genre));
        h.extend(Self::affine(&feats[o_art..o_alb], self.blob("embed_artist")?, None, self.cfg.embed.artist));
        h.extend(Self::affine(&feats[o_alb..o_alb + b.album], self.blob("embed_album")?, None, self.cfg.embed.album));

        let mut g0 = Self::affine(&h, self.blob("trunk_w0")?, Some(self.blob("trunk_b0")?), self.cfg.trunk[0]);
        for v in g0.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        let mut g1 = Self::affine(&g0, self.blob("trunk_w1")?, Some(self.blob("trunk_b1")?), self.cfg.trunk[1]);
        for v in g1.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        let logits = Self::affine(&g1, self.blob("head_w")?, Some(self.blob("head_b")?), self.cfg.n_class);

        // numerically stable softmax, take class_index
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0f32;
        for &z in &logits {
            denom += (z - mx).exp();
        }
        let idx = self.cfg.class_index.min(self.cfg.n_class - 1);
        let raw = (logits[idx] - mx).exp() / denom;
        Ok(nan_clip(raw))
    }

    pub fn score(&self, latent: &[f32], meta: &Value) -> Result<f32, String> {
        let feats = self.featurize(latent, meta)?;
        self.forward(&feats)
    }
}

/// numpy nan_to_num(nan=0.5, posinf=1, neginf=0) then clip(0,1) — defensive; a
/// softmax output is already in [0,1].
fn nan_clip(x: f32) -> f32 {
    if x.is_nan() {
        0.5
    } else if x == f32::INFINITY {
        1.0
    } else if x == f32::NEG_INFINITY {
        0.0
    } else {
        x.clamp(0.0, 1.0)
    }
}
