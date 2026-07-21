//! Promoted next-track champion `blend-gru-markov-content-proj` (R'+M+C'),
//! ported to run in-isolate from the baked `public/data/champion.bin`
//! (tools/build_champion.py). No torch / numpy at runtime.
//!
//! The champion z-blends three legs over a LEARNED-PROJECTED PCA-192 item space
//! (proj = L2-normalize(item_latents @ W.T), baked verbatim):
//!   * R' — single-layer GRU (hidden 256) over the projected prefix latents ->
//!          predicted next latent (192-d) -> cosine retrieval,
//!   * M  — first-order Markov (track bigram + artist/genre back-off) via the
//!          exact `transit::TransitionModel::affinity` formula, rebuilt from the
//!          champion TRAIN sessions, plus the 1e-6*pop tie-break tail that
//!          seq_baselines.markov_scorer adds,
//!   * C' — content-kNN: max cosine of each candidate's projected latent to the
//!          prefix items' projected latents.
//! Blend = z-normalize each leg over the CANDIDATES (items not in the prefix),
//! equal-thirds average; rank descending (stable, ties -> lower index); exclude
//! prefix items (next-distinct); top-k.
//!
//! This is a faithful port of the reference predictors (seq_blend.py /
//! seq_nexttrack.py / seq_baselines.py / seq_models.py) from the sibling
//! next-track instance. The GRU forward runs in f32 (matching torch); the
//! z-blend / content / Markov arithmetic runs in f64 (matching numpy). Parity
//! against the Python reference is gated by worker-core/tests/champion.rs.

use crate::snapshot::Meta;
use crate::transit::TransitionModel;
use std::collections::HashMap;

const EPS_Z: f64 = 1e-9; // seq_blend.EPS (z-norm denominator epsilon)
const EPS_NORM_F64: f64 = 1e-12; // content-kNN f64 re-normalization epsilon
const EPS_NORM_F32: f32 = 1e-12; // torch F.normalize default eps
const POP_MIX: f64 = 1e-6; // seq_baselines popularity tie-break weight

fn u32le(b: &[u8], p: usize) -> u32 {
    u32::from_le_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]])
}

fn read_f32_vec(b: &[u8], p: usize, count: usize) -> (Vec<f32>, usize) {
    let mut out = vec![0.0f32; count];
    for (i, v) in out.iter_mut().enumerate() {
        let o = p + i * 4;
        *v = f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    }
    (out, p + count * 4)
}

pub struct ExtendItem {
    pub item_index: usize,
    pub uri: String,
    pub name: String,
    pub artist: String,
    pub genre: String,
    pub score: f64,
}

pub struct Champion {
    pub n_items: usize,
    pub dim: usize,    // projected/rank dim = 192
    pub hidden: usize, // GRU hidden width = 256
    // Projected + L2-normalized item latents, row-major (n_items * dim), f32.
    proj: Vec<f32>,
    // GRU state_dict blobs (torch row-major; gate order r,z,n).
    gru_w_ih: Vec<f32>, // (3*hidden, dim)
    gru_w_hh: Vec<f32>, // (3*hidden, hidden)
    gru_b_ih: Vec<f32>, // (3*hidden)
    gru_b_hh: Vec<f32>, // (3*hidden)
    head_w: Vec<f32>,   // (dim, hidden)
    head_b: Vec<f32>,   // (dim)
    // Markov leg.
    transitions: TransitionModel,
    metas: Vec<Meta>,      // per item index (uri/name/artist/genre)
    pop_tail: Vec<f64>,    // POP_MIX * counts[i] / (max_count + 1e-9)
    // Token resolution (mirrors seq_common.resolve_prefix).
    uri_to_idx: HashMap<String, usize>,
    id_to_idx: HashMap<String, usize>,
}

impl Champion {
    pub fn parse(bytes: &[u8]) -> Result<Champion, String> {
        if bytes.len() < 20 || &bytes[0..4] != b"PFCH" {
            return Err("champion.bin: bad magic".into());
        }
        let _version = u32le(bytes, 4);
        let n_items = u32le(bytes, 8) as usize;
        let dim = u32le(bytes, 12) as usize;
        let hidden = u32le(bytes, 16) as usize;
        let mut p = 20;

        let (proj, np_) = read_f32_vec(bytes, p, n_items * dim);
        p = np_;
        let (gru_w_ih, np_) = read_f32_vec(bytes, p, 3 * hidden * dim);
        p = np_;
        let (gru_w_hh, np_) = read_f32_vec(bytes, p, 3 * hidden * hidden);
        p = np_;
        let (gru_b_ih, np_) = read_f32_vec(bytes, p, 3 * hidden);
        p = np_;
        let (gru_b_hh, np_) = read_f32_vec(bytes, p, 3 * hidden);
        p = np_;
        let (head_w, np_) = read_f32_vec(bytes, p, dim * hidden);
        p = np_;
        let (head_b, np_) = read_f32_vec(bytes, p, dim);
        p = np_;
        let (pop_counts, np_) = read_f32_vec(bytes, p, n_items);
        p = np_;

        let trans_len = u32le(bytes, p) as usize;
        p += 4;
        let transitions = TransitionModel::parse(&bytes[p..p + trans_len])?;
        p += trans_len;

        let items_len = u32le(bytes, p) as usize;
        p += 4;
        let metas: Vec<Meta> = serde_json::from_slice(&bytes[p..p + items_len])
            .map_err(|e| format!("champion items json: {e}"))?;
        if metas.len() != n_items {
            return Err(format!(
                "champion items {} != n_items {}",
                metas.len(),
                n_items
            ));
        }

        // Precompute the popularity tie-break tail (seq_baselines.markov_scorer:
        // pop = counts / (counts.max() + 1e-9); scores += 1e-6 * pop).
        let max_count = pop_counts.iter().fold(0.0f64, |m, &c| m.max(c as f64));
        let denom = max_count + 1e-9;
        let pop_tail: Vec<f64> = pop_counts
            .iter()
            .map(|&c| POP_MIX * (c as f64) / denom)
            .collect();

        // Resolution maps (uri + bare id after the last ':').
        let mut uri_to_idx = HashMap::with_capacity(n_items);
        let mut id_to_idx = HashMap::with_capacity(n_items);
        for (i, m) in metas.iter().enumerate() {
            if !m.uri.is_empty() {
                uri_to_idx.insert(m.uri.clone(), i);
                if let Some(bare) = m.uri.rsplit(':').next() {
                    id_to_idx.insert(bare.to_string(), i);
                }
            }
        }

        Ok(Champion {
            n_items,
            dim,
            hidden,
            proj,
            gru_w_ih,
            gru_w_hh,
            gru_b_ih,
            gru_b_hh,
            head_w,
            head_b,
            transitions,
            metas,
            pop_tail,
            uri_to_idx,
            id_to_idx,
        })
    }

    #[inline]
    fn proj_row(&self, i: usize) -> &[f32] {
        &self.proj[i * self.dim..(i + 1) * self.dim]
    }

    pub fn meta(&self, i: usize) -> Option<&Meta> {
        self.metas.get(i)
    }

    /// Resolve one serving token to a vocab item index (mirrors
    /// seq_common.resolve_prefix): a bare integer index, a full Spotify URI, a
    /// bare track id, or a Spotify URL. Returns None for unknown / cold tokens.
    pub fn resolve(&self, token: &str) -> Option<usize> {
        let t = token.trim();
        // Bare integer vocab index.
        if !t.is_empty() && t.chars().all(|c| c.is_ascii_digit()) {
            let i: usize = t.parse().ok()?;
            return if i < self.n_items { Some(i) } else { None };
        }
        if let Some(&i) = self.uri_to_idx.get(t) {
            return Some(i);
        }
        // https://open.spotify.com/track/<id>?si=...  ->  <id>
        let no_query = t.split('?').next().unwrap_or(t).trim_end_matches('/');
        let bare = no_query
            .rsplit('/')
            .next()
            .unwrap_or(no_query)
            .rsplit(':')
            .next()
            .unwrap_or(no_query);
        self.id_to_idx.get(bare).copied()
    }

    // ---------------------------------------------------------------------- //
    // R' — GRU leg                                                           //
    // ---------------------------------------------------------------------- //
    /// Raw GRU predicted next-latent (head output, PRE-normalize) for a prefix
    /// of projected latents, processed left->right. f32 (matches torch).
    pub fn predict_next_latent(&self, prefix: &[usize]) -> Vec<f32> {
        let (d, hh) = (self.dim, self.hidden);
        let mut h = vec![0.0f32; hh];
        // Scratch for the three input-side gate pre-activations (r,z,n).
        let mut ih = vec![0.0f32; 3 * hh];
        let mut hg = vec![0.0f32; 3 * hh]; // hidden-side gate pre-activations
        for &item in prefix {
            let x = self.proj_row(item);
            // ih = W_ih x + b_ih ; hg = W_hh h + b_hh   (all three gates)
            for g in 0..3 * hh {
                let mut s = self.gru_b_ih[g];
                let base = g * d;
                for c in 0..d {
                    s += self.gru_w_ih[base + c] * x[c];
                }
                ih[g] = s;
                let mut s2 = self.gru_b_hh[g];
                let base2 = g * hh;
                for c in 0..hh {
                    s2 += self.gru_w_hh[base2 + c] * h[c];
                }
                hg[g] = s2;
            }
            for i in 0..hh {
                let r = sigmoid_f32(ih[i] + hg[i]);
                let z = sigmoid_f32(ih[hh + i] + hg[hh + i]);
                let n = (ih[2 * hh + i] + r * hg[2 * hh + i]).tanh();
                h[i] = (1.0 - z) * n + z * h[i];
            }
        }
        // pred = head_w @ h + head_b
        let mut pred = vec![0.0f32; d];
        for j in 0..d {
            let mut s = self.head_b[j];
            let base = j * hh;
            for c in 0..hh {
                s += self.head_w[base + c] * h[c];
            }
            pred[j] = s;
        }
        pred
    }

    /// Full-vocab GRU cosine scores: cosine(normalize(pred), normalize(proj_i)).
    /// Matches seq_blend.gru_score_fn (f32 torch), returned as f64.
    fn gru_scores(&self, prefix: &[usize]) -> Vec<f64> {
        let d = self.dim;
        let pred = self.predict_next_latent(prefix);
        // F.normalize(pred): / max(||pred||, 1e-12)
        let pn = {
            let mut s = 0.0f32;
            for &v in &pred {
                s += v * v;
            }
            s.sqrt().max(EPS_NORM_F32)
        };
        let predn: Vec<f32> = pred.iter().map(|&v| v / pn).collect();
        let mut out = vec![0.0f64; self.n_items];
        for i in 0..self.n_items {
            let row = self.proj_row(i);
            // item_norm = F.normalize(proj_i): / max(||proj_i||, 1e-12)
            let mut nrm = 0.0f32;
            let mut dot = 0.0f32;
            for c in 0..d {
                nrm += row[c] * row[c];
                dot += predn[c] * row[c];
            }
            let denom = nrm.sqrt().max(EPS_NORM_F32);
            out[i] = (dot / denom) as f64;
        }
        out
    }

    // ---------------------------------------------------------------------- //
    // C' — content-kNN leg (max cosine to prefix, f64)                       //
    // ---------------------------------------------------------------------- //
    fn content_scores(&self, prefix: &[usize]) -> Vec<f64> {
        let d = self.dim;
        if prefix.is_empty() {
            return vec![0.0f64; self.n_items];
        }
        // f64 unit vectors for the prefix items (unit = proj / (||proj||+1e-12)).
        let unit_of = |i: usize| -> Vec<f64> {
            let row = self.proj_row(i);
            let mut nrm = 0.0f64;
            for c in 0..d {
                nrm += (row[c] as f64) * (row[c] as f64);
            }
            let denom = nrm.sqrt() + EPS_NORM_F64;
            (0..d).map(|c| (row[c] as f64) / denom).collect()
        };
        let pref_units: Vec<Vec<f64>> = prefix.iter().map(|&j| unit_of(j)).collect();
        let mut out = vec![0.0f64; self.n_items];
        for i in 0..self.n_items {
            let row = self.proj_row(i);
            let mut nrm = 0.0f64;
            for c in 0..d {
                nrm += (row[c] as f64) * (row[c] as f64);
            }
            let denom = nrm.sqrt() + EPS_NORM_F64;
            let mut best = f64::NEG_INFINITY;
            for pu in &pref_units {
                let mut dot = 0.0f64;
                for c in 0..d {
                    dot += (row[c] as f64) / denom * pu[c];
                }
                if dot > best {
                    best = dot;
                }
            }
            out[i] = best;
        }
        out
    }

    // ---------------------------------------------------------------------- //
    // M — first-order Markov leg (affinity + 1e-6*pop tail, f64)             //
    // ---------------------------------------------------------------------- //
    fn markov_scores(&self, prefix: &[usize]) -> Vec<f64> {
        let mut out = vec![0.0f64; self.n_items];
        let Some(&u) = prefix.last() else {
            return out;
        };
        let mu = &self.metas[u];
        for i in 0..self.n_items {
            out[i] = self.transitions.affinity(mu, &self.metas[i]) + self.pop_tail[i];
        }
        out
    }

    // ---------------------------------------------------------------------- //
    // Equal-thirds z-blend over candidates + rank                            //
    // ---------------------------------------------------------------------- //
    /// Ranked (item_index, blended_score) best-first, excluding prefix items
    /// (next-distinct). Mirrors seq_blend.build_blend_score_fn + rank_topk.
    pub fn blend_rank(&self, prefix: &[usize], k: usize) -> Vec<(usize, f64)> {
        let n = self.n_items;
        let mut is_prefix = vec![false; n];
        for &p in prefix {
            if p < n {
                is_prefix[p] = true;
            }
        }
        let gru = self.gru_scores(prefix);
        let markov = self.markov_scores(prefix);
        let content = self.content_scores(prefix);

        let zg = zcand(&gru, &is_prefix);
        let zm = zcand(&markov, &is_prefix);
        let zc = zcand(&content, &is_prefix);

        let mut blended = vec![f64::NEG_INFINITY; n];
        for i in 0..n {
            if is_prefix[i] {
                continue; // excluded (next-distinct)
            }
            blended[i] = (zg[i] + zm[i] + zc[i]) / 3.0;
        }

        // Descending by score; ties -> lower index (matches np.argsort(-s, stable)).
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| match blended[b].partial_cmp(&blended[a]) {
            Some(std::cmp::Ordering::Equal) | None => a.cmp(&b),
            Some(o) => o,
        });
        order
            .into_iter()
            .take(k)
            .map(|i| (i, blended[i]))
            .collect()
    }

    /// Ranked next-track suggestions with identity, best-first.
    pub fn extend(&self, prefix: &[usize], k: usize) -> Vec<ExtendItem> {
        self.blend_rank(prefix, k)
            .into_iter()
            .map(|(i, score)| {
                let m = &self.metas[i];
                ExtendItem {
                    item_index: i,
                    uri: m.uri.clone(),
                    name: m.name.clone(),
                    artist: m.artist.clone(),
                    genre: m.genre.clone(),
                    score,
                }
            })
            .collect()
    }
}

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Z-normalize `v` using mean/std computed over the CANDIDATE items (indices
/// where `is_prefix` is false), applied to the whole vector. Population std
/// (ddof=0) + EPS_Z, matching seq_blend._zcand exactly.
fn zcand(v: &[f64], is_prefix: &[bool]) -> Vec<f64> {
    let mut sum = 0.0f64;
    let mut cnt = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if !is_prefix[i] {
            sum += x;
            cnt += 1;
        }
    }
    let mean = if cnt > 0 { sum / cnt as f64 } else { 0.0 };
    let mut var = 0.0f64;
    for (i, &x) in v.iter().enumerate() {
        if !is_prefix[i] {
            let d = x - mean;
            var += d * d;
        }
    }
    let std = if cnt > 0 { (var / cnt as f64).sqrt() } else { 0.0 };
    let denom = std + EPS_Z;
    v.iter().map(|&x| (x - mean) / denom).collect()
}
