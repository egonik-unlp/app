//! Port of the sibling pathfinder transition model (server/app.py TransitionModel).

use crate::snapshot::Meta;
use serde::Deserialize;
use std::collections::HashMap;

const TRANSITION_BACKOFF_K: f64 = 8.0;
const TRANSITION_ARTIST_WEIGHT: f64 = 0.6;
const TRANSITION_GENRE_WEIGHT: f64 = 0.4;

type Cond = HashMap<String, HashMap<String, f64>>;

#[derive(Deserialize)]
pub struct TransitionModel {
    pub track_bigram: Cond,
    pub artist_cond: Cond,
    pub genre_cond: Cond,
    pub ctx_genre_lift: Cond,
    pub ctx_track_lift: Cond,
    #[serde(default)]
    pub meta: serde_json::Value,
}

pub struct Context {
    pub genre_tod: HashMap<String, f64>,
    pub genre_shuf: HashMap<String, f64>,
    pub track_tod: HashMap<String, f64>,
    pub label: String,
}

impl TransitionModel {
    pub fn parse(bytes: &[u8]) -> Result<TransitionModel, String> {
        serde_json::from_slice(bytes).map_err(|e| format!("transitions.json: {e}"))
    }

    /// Next-track affinity (track bigram with artist/genre back-off), matches app.py.
    pub fn affinity(&self, mu: &Meta, mv: &Meta) -> f64 {
        let mut track_p = 0.0;
        let mut n_u = 0.0;
        if let Some(tb) = self.track_bigram.get(&mu.uri) {
            n_u = tb.values().sum();
            if n_u > 0.0 {
                track_p = tb.get(&mv.uri).copied().unwrap_or(0.0) / n_u;
            }
        }
        let a_p = self
            .artist_cond
            .get(&mu.artist)
            .and_then(|m| m.get(&mv.artist))
            .copied()
            .unwrap_or(0.0);
        let g_p = self
            .genre_cond
            .get(&mu.genre)
            .and_then(|m| m.get(&mv.genre))
            .copied()
            .unwrap_or(0.0);
        let backoff = TRANSITION_ARTIST_WEIGHT * a_p + TRANSITION_GENRE_WEIGHT * g_p;
        let trust = n_u / (n_u + TRANSITION_BACKOFF_K);
        trust * track_p + (1.0 - trust) * backoff
    }

    pub fn context_fit(&self, mv: &Meta, ctx: &Context) -> f64 {
        let mut lift = ctx.genre_tod.get(&mv.genre).copied().unwrap_or(1.0)
            * ctx.genre_shuf.get(&mv.genre).copied().unwrap_or(1.0);
        if let Some(t) = ctx.track_tod.get(&mv.uri) {
            lift *= *t;
        }
        lift
    }

    /// Build a Context. `tod` is "" (any) or one of the tod buckets — already
    /// resolved by the caller (the Worker turns "now" into a bucket via Date).
    /// `shuffle` is "", "shuffle", or "linear".
    pub fn resolve_context(&self, tod: &str, shuffle: &str) -> Context {
        let mut labels: Vec<String> = Vec::new();
        let mut genre_tod = HashMap::new();
        let mut track_tod = HashMap::new();
        if !tod.is_empty() {
            if let Some(m) = self.ctx_genre_lift.get(&format!("tod:{tod}")) {
                genre_tod = m.clone();
            }
            if let Some(m) = self.ctx_track_lift.get(tod) {
                track_tod = m.clone();
            }
            labels.push(tod.to_string());
        }
        let mut genre_shuf = HashMap::new();
        if shuffle == "shuffle" || shuffle == "linear" {
            if let Some(m) = self.ctx_genre_lift.get(&format!("shuf:{shuffle}")) {
                genre_shuf = m.clone();
            }
            labels.push(shuffle.to_string());
        }
        let label = if labels.is_empty() {
            "any".to_string()
        } else {
            labels.join(" · ")
        };
        Context {
            genre_tod,
            genre_shuf,
            track_tod,
            label,
        }
    }
}
