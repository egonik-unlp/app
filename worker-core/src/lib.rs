//! WASM compute core for the Cloudflare Worker pathfinder. Loads the static
//! corpus snapshot + transition model + optional cold-start projector, and
//! exposes search / snap / route over `wasm-bindgen`. All heavy numeric work
//! (A*, densify, PCA) lives here; the TS Worker owns I/O and bindings.

pub mod pathfind;
pub mod pca;
pub mod project;
pub mod snapshot;
pub mod transit;

use project::Projector;
use serde_json::{json, Value};
use snapshot::Corpus;
use std::cell::RefCell;
use transit::TransitionModel;
use wasm_bindgen::prelude::*;

const MAX_EXPANSIONS: usize = 200_000;
const DEFAULT_LENGTH: usize = 14;

struct State {
    corpus: Corpus,
    transitions: Option<TransitionModel>,
    projector: Option<Projector>,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

fn err(msg: impl Into<String>) -> JsValue {
    JsValue::from_str(&msg.into())
}

#[wasm_bindgen]
pub fn load_corpus(corpus_bytes: &[u8], transitions_bytes: &[u8]) -> Result<(), JsValue> {
    let corpus = Corpus::parse(corpus_bytes).map_err(err)?;
    let transitions = if transitions_bytes.is_empty() {
        None
    } else {
        Some(TransitionModel::parse(transitions_bytes).map_err(err)?)
    };
    STATE.with(|s| {
        *s.borrow_mut() = Some(State {
            corpus,
            transitions,
            projector: None,
        });
    });
    Ok(())
}

#[wasm_bindgen]
pub fn set_projector(bytes: &[u8]) -> Result<(), JsValue> {
    let p = Projector::parse(bytes).map_err(err)?;
    STATE.with(|s| {
        if let Some(st) = s.borrow_mut().as_mut() {
            st.projector = Some(p);
        }
    });
    Ok(())
}

fn with_state<R>(f: impl FnOnce(&State) -> Result<R, JsValue>) -> Result<R, JsValue> {
    STATE.with(|s| {
        let b = s.borrow();
        let st = b.as_ref().ok_or_else(|| err("core not initialized"))?;
        f(st)
    })
}

#[wasm_bindgen]
pub fn info() -> Result<String, JsValue> {
    with_state(|st| {
        Ok(json!({
            "n": st.corpus.n,
            "dim": st.corpus.dim,
            "k": st.corpus.k,
            "has_transitions": st.transitions.is_some(),
            "has_projector": st.projector.is_some(),
            "text_model": st.projector.as_ref().map(|p| p.text_model()),
        })
        .to_string())
    })
}

// i32 (not i64) so JS gets a plain number, not a BigInt — rows fit easily.
#[wasm_bindgen]
pub fn find_row(uri: &str) -> Result<i32, JsValue> {
    with_state(|st| {
        Ok(st
            .corpus
            .uri_to_row
            .get(uri)
            .map(|&r| r as i32)
            .unwrap_or(-1))
    })
}

#[wasm_bindgen]
pub fn id_at(row: usize) -> Result<String, JsValue> {
    with_state(|st| {
        st.corpus
            .ids
            .get(row)
            .map(|id| id.to_string())
            .ok_or_else(|| err("row out of range"))
    })
}

#[wasm_bindgen]
pub fn vec_at(row: usize) -> Result<Vec<f32>, JsValue> {
    with_state(|st| {
        if row >= st.corpus.n {
            return Err(err("row out of range"));
        }
        Ok(st.corpus.vec_at(row).to_vec())
    })
}

#[wasm_bindgen]
pub fn meta_at(row: usize) -> Result<String, JsValue> {
    with_state(|st| {
        let m = st.corpus.meta.get(row).ok_or_else(|| err("row out of range"))?;
        Ok(json!({
            "uri": m.uri, "name": m.name, "artist": m.artist,
            "album": m.album, "genre": m.genre,
        })
        .to_string())
    })
}

#[wasm_bindgen]
pub fn search(query: &str, limit: usize) -> Result<String, JsValue> {
    with_state(|st| {
        let q = query.trim().to_lowercase();
        if q.is_empty() {
            return Ok("[]".to_string());
        }
        let mut hits: Vec<(u8, usize, usize)> = Vec::new(); // (rank, name_len, row)
        for row in 0..st.corpus.n {
            let m = &st.corpus.meta[row];
            let name = m.name.to_lowercase();
            let artist = m.artist.to_lowercase();
            let rank = if name.starts_with(&q) {
                0
            } else if name.contains(&q) {
                1
            } else if artist.contains(&q) {
                2
            } else if format!("{artist} {name}").contains(&q) {
                3
            } else {
                continue;
            };
            hits.push((rank, name.len(), row));
        }
        hits.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        let out: Vec<Value> = hits
            .into_iter()
            .take(limit.max(1))
            .map(|(_, _, row)| {
                let m = &st.corpus.meta[row];
                json!({
                    "spotify_id": spotify_id(&m.uri),
                    "uri": m.uri, "name": m.name, "artist": m.artist,
                    "album": m.album, "genre": m.genre,
                    "in_corpus": true, "art": Value::Null,
                })
            })
            .collect();
        Ok(Value::Array(out).to_string())
    })
}

#[wasm_bindgen]
pub fn snap(v: &[f32]) -> Result<usize, JsValue> {
    with_state(|st| Ok(st.corpus.snap(v)))
}

/// Cold-start: project an arbitrary track's features into the corpus latent
/// space and snap to the nearest anchor. `meta_json` carries the raw Spotify +
/// ReccoBeats fields; `text_emb` is the bge embedding from the Worker AI binding.
#[wasm_bindgen]
pub fn project_and_snap(text_emb: &[f32], meta_json: &str) -> Result<String, JsValue> {
    with_state(|st| {
        let p = st
            .projector
            .as_ref()
            .ok_or_else(|| err("cold-start projector unavailable"))?;
        let meta: Value = serde_json::from_str(meta_json).map_err(|e| err(e.to_string()))?;
        let vec = p.project(text_emb, &meta).map_err(err)?;
        let snap_row = st.corpus.snap(&vec);
        Ok(json!({
            "snap_row": snap_row,
            "snap_id": st.corpus.ids[snap_row].to_string(),
            "vec": vec,
        })
        .to_string())
    })
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen]
pub fn route(
    start_row: usize,
    end_row: usize,
    start_vec: &[f32],
    end_vec: &[f32],
    length: i32,
    tod: &str,
    shuffle: &str,
) -> Result<String, JsValue> {
    with_state(|st| {
        let c = &st.corpus;
        if start_row >= c.n || end_row >= c.n {
            return Err(err("anchor row out of range"));
        }
        let length = (length.max(4).min(50)) as usize;
        let _ = DEFAULT_LENGTH;

        let model = st.transitions.as_ref();
        let ctx = model.map(|m| {
            let tod = if tod == "any" { "" } else { tod };
            let shuffle = if shuffle == "any" { "" } else { shuffle };
            m.resolve_context(tod, shuffle)
        });

        let path = pathfind::find_path(
            c,
            start_row,
            end_row,
            length,
            MAX_EXPANSIONS,
            model,
            ctx.as_ref(),
        )
        .ok_or_else(|| err("no_path"))?;
        let path = if path.len() < length {
            pathfind::densify(c, path, length, model, ctx.as_ref())
        } else {
            path
        };

        // cloud: up to 8 nearest neighbors per path node, capped at 90
        let mut cloud: Vec<usize> = Vec::new();
        let mut seen: std::collections::HashSet<usize> = path.iter().copied().collect();
        'outer: for &pid in &path {
            for (nbr, _) in c.neighbors(pid).into_iter().take(8) {
                if seen.insert(nbr) {
                    cloud.push(nbr);
                }
                if cloud.len() >= 90 {
                    break 'outer;
                }
            }
        }

        // PCA over path + cloud + the two requested vectors
        let mut vectors: Vec<&[f32]> = Vec::with_capacity(path.len() + cloud.len() + 2);
        for &pid in &path {
            vectors.push(c.vec_at(pid));
        }
        for &pid in &cloud {
            vectors.push(c.vec_at(pid));
        }
        vectors.push(start_vec);
        vectors.push(end_vec);
        let pos = pca::project3d(&vectors, c.dim);

        let node = |row: usize, idx: usize| -> Value {
            let m = &c.meta[row];
            let p = pos[idx];
            json!({
                "id": c.ids[row].to_string(),
                "uri": m.uri,
                "name": m.name,
                "artist": m.artist,
                "album": m.album,
                "genre": m.genre,
                "spotify_url": spotify_url(&m.uri),
                "fit": c.fit[row] as f64,
                "position": [p[0], p[1], p[2]],
                "kind": "path",
            })
        };

        let path_json: Vec<Value> = path.iter().enumerate().map(|(i, &r)| node(r, i)).collect();
        let cloud_json: Vec<Value> = cloud
            .iter()
            .enumerate()
            .map(|(i, &r)| node(r, path.len() + i))
            .collect();
        let edges: Vec<Value> = path
            .windows(2)
            .map(|w| json!({"from": c.ids[w[0]].to_string(), "to": c.ids[w[1]].to_string(), "kind": "path"}))
            .collect();
        let start_pos = pos[path.len() + cloud.len()];
        let end_pos = pos[path.len() + cloud.len() + 1];

        Ok(json!({
            "context": ctx.as_ref().map(|c| c.label.clone()),
            "path": path_json,
            "cloud": cloud_json,
            "edges": edges,
            "req_start_pos": [start_pos[0], start_pos[1], start_pos[2]],
            "req_end_pos": [end_pos[0], end_pos[1], end_pos[2]],
        })
        .to_string())
    })
}

fn spotify_id(uri: &str) -> String {
    uri.rsplit(':').next().unwrap_or("").to_string()
}

fn spotify_url(uri: &str) -> Value {
    if uri.is_empty() {
        Value::Null
    } else {
        Value::String(format!("https://open.spotify.com/track/{}", spotify_id(uri)))
    }
}
