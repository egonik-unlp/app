//! Parser for `corpus.bin` (magic `PFC1`). See tools/SNAPSHOT_FORMAT.md.

use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize, Clone)]
pub struct Meta {
    pub uri: String,
    pub name: String,
    pub artist: String,
    pub album: Option<String>,
    pub genre: String,
}

pub struct Corpus {
    pub n: usize,
    pub dim: usize,
    pub k: usize,
    pub ids: Vec<u64>,
    pub vecs: Vec<f32>, // n * dim, L2-normalized
    pub fit: Vec<f32>,  // n
    pub nbr_idx: Vec<u32>, // n * k (0xFFFFFFFF = empty)
    pub nbr_dist: Vec<f32>, // n * k
    pub meta: Vec<Meta>,
    pub uri_to_row: HashMap<String, usize>,
}

struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Cur<'a> {
    fn new(b: &'a [u8]) -> Self {
        Cur { b, p: 0 }
    }
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
    fn take(&mut self, len: usize) -> &'a [u8] {
        let s = &self.b[self.p..self.p + len];
        self.p += len;
        s
    }
}

fn read_u64(s: &[u8]) -> Vec<u64> {
    s.chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
fn read_u32(s: &[u8]) -> Vec<u32> {
    s.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
fn read_f32(s: &[u8]) -> Vec<f32> {
    s.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

impl Corpus {
    pub fn parse(bytes: &[u8]) -> Result<Corpus, String> {
        if bytes.len() < 20 || &bytes[0..4] != b"PFC1" {
            return Err("corpus.bin: bad magic".into());
        }
        let mut c = Cur::new(bytes);
        c.p = 4;
        let _version = c.u32();
        let n = c.u32() as usize;
        let dim = c.u32() as usize;
        let k = c.u32() as usize;

        let ids = read_u64(c.take(n * 8));
        let vecs = read_f32(c.take(n * dim * 4));
        let fit = read_f32(c.take(n * 4));
        let nbr_idx = read_u32(c.take(n * k * 4));
        let nbr_dist = read_f32(c.take(n * k * 4));
        let meta_len = c.u32() as usize;
        let meta_bytes = c.take(meta_len);
        let meta: Vec<Meta> =
            serde_json::from_slice(meta_bytes).map_err(|e| format!("corpus meta json: {e}"))?;

        let mut uri_to_row = HashMap::with_capacity(n);
        for (row, m) in meta.iter().enumerate() {
            if !m.uri.is_empty() {
                uri_to_row.insert(m.uri.clone(), row);
            }
        }

        Ok(Corpus {
            n,
            dim,
            k,
            ids,
            vecs,
            fit,
            nbr_idx,
            nbr_dist,
            meta,
            uri_to_row,
        })
    }

    #[inline]
    pub fn vec_at(&self, row: usize) -> &[f32] {
        &self.vecs[row * self.dim..(row + 1) * self.dim]
    }

    /// cosine distance (1 - cos) between two corpus rows (vectors are normalized).
    #[inline]
    pub fn cos_dist(&self, a: usize, b: usize) -> f64 {
        let (va, vb) = (self.vec_at(a), self.vec_at(b));
        let mut dot = 0.0f64;
        for i in 0..self.dim {
            dot += (va[i] * vb[i]) as f64;
        }
        1.0 - dot
    }

    /// cosine distance between a corpus row and an external normalized vector.
    #[inline]
    pub fn cos_dist_vec(&self, row: usize, v: &[f32]) -> f64 {
        let vr = self.vec_at(row);
        let mut dot = 0.0f64;
        for i in 0..self.dim {
            dot += (vr[i] * v[i]) as f64;
        }
        1.0 - dot
    }

    /// neighbors of `row` as (neighbor_row, cos_dist), ascending by distance.
    pub fn neighbors(&self, row: usize) -> Vec<(usize, f64)> {
        let base = row * self.k;
        let mut out = Vec::with_capacity(self.k);
        for j in 0..self.k {
            let idx = self.nbr_idx[base + j];
            if idx == u32::MAX {
                continue;
            }
            out.push((idx as usize, self.nbr_dist[base + j] as f64));
        }
        out
    }

    /// nearest corpus row to an external normalized vector (argmax cosine).
    pub fn snap(&self, v: &[f32]) -> usize {
        let mut best = 0usize;
        let mut best_dot = f64::NEG_INFINITY;
        for row in 0..self.n {
            let vr = self.vec_at(row);
            let mut dot = 0.0f64;
            for i in 0..self.dim {
                dot += (vr[i] * v[i]) as f64;
            }
            if dot > best_dot {
                best_dot = dot;
                best = row;
            }
        }
        best
    }

    /// Top-`k` corpus rows nearest an external normalized vector, as
    /// (row, cos_dist) ascending by distance. Generalizes `snap` (k=1 argmax) —
    /// `knn_vec(v, 1)[0].0 == snap(v)` — and costs the same full scan. Used to
    /// give an off-corpus ("off-map") track real graph edges into the corpus
    /// instead of collapsing it to a single snapped anchor.
    pub fn knn_vec(&self, v: &[f32], k: usize) -> Vec<(usize, f64)> {
        if self.n == 0 || k == 0 {
            return Vec::new();
        }
        let mut all: Vec<(f64, usize)> = (0..self.n)
            .map(|row| (self.cos_dist_vec(row, v), row))
            .collect();
        let take = k.min(all.len());
        if take < all.len() {
            all.select_nth_unstable_by(take - 1, |a, b| a.0.total_cmp(&b.0));
            all.truncate(take);
        }
        all.sort_by(|a, b| a.0.total_cmp(&b.0));
        all.into_iter().map(|(d, r)| (r, d)).collect()
    }
}
