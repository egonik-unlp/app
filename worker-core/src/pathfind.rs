//! Faithful port of pathfinder/search.py (A* + densify) and the cost weights
//! from pathfinder/config.py. Operates on corpus row indices.

use crate::snapshot::Corpus;
use crate::transit::{Context, TransitionModel};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

const W_DIST: f64 = 1.0;
const W_FIT: f64 = 0.5;
const W_DIV: f64 = 0.5;
const W_TRANS: f64 = 0.6;
const W_CTX: f64 = 0.4;
const GENRE_JUMP_PENALTY: f64 = 0.15;
const MAX_CONSECUTIVE_ARTIST: usize = 2;
const ARTIST_SHARE_DIVISOR: f64 = 4.0;

fn minmax(raw: &[(usize, f64)]) -> HashMap<usize, f64> {
    let mut out = HashMap::with_capacity(raw.len());
    if raw.is_empty() {
        return out;
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &(_, v) in raw {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let span = hi - lo;
    if span <= 0.0 {
        for &(k, _) in raw {
            out.insert(k, 0.5);
        }
    } else {
        for &(k, v) in raw {
            out.insert(k, (v - lo) / span);
        }
    }
    out
}

fn artist(c: &Corpus, row: usize) -> &str {
    &c.meta[row].artist
}

fn violates(c: &Corpus, path: &[usize], candidate: usize, length_hint: usize) -> bool {
    if path.contains(&candidate) {
        return true;
    }
    let a = artist(c, candidate);
    let mut run = 1;
    for &pid in path.iter().rev() {
        if artist(c, pid) == a {
            run += 1;
        } else {
            break;
        }
    }
    if run > MAX_CONSECUTIVE_ARTIST {
        return true;
    }
    let cap = ((length_hint.max(path.len() + 1)) as f64 / ARTIST_SHARE_DIVISOR).ceil() as usize;
    let total = path.iter().filter(|&&pid| artist(c, pid) == a).count() + 1;
    total > cap
}

fn violates_insertion(
    c: &Corpus,
    path: &[usize],
    i: usize,
    candidate: usize,
    length_hint: usize,
) -> bool {
    if path.contains(&candidate) {
        return true;
    }
    let a = artist(c, candidate);
    let mut run = 1;
    for &pid in path[..=i].iter().rev() {
        if artist(c, pid) == a {
            run += 1;
        } else {
            break;
        }
    }
    for &pid in &path[i + 1..] {
        if artist(c, pid) == a {
            run += 1;
        } else {
            break;
        }
    }
    if run > MAX_CONSECUTIVE_ARTIST {
        return true;
    }
    let cap = ((length_hint.max(path.len() + 1)) as f64 / ARTIST_SHARE_DIVISOR).ceil() as usize;
    let total = path.iter().filter(|&&pid| artist(c, pid) == a).count() + 1;
    total > cap
}

struct HeapItem {
    f: f64,
    counter: u64,
    g: f64,
    node: usize,
    path: Vec<usize>,
}
impl PartialEq for HeapItem {
    fn eq(&self, o: &Self) -> bool {
        self.f == o.f && self.counter == o.counter
    }
}
impl Eq for HeapItem {}
impl Ord for HeapItem {
    // Reversed: smaller (f, counter) is "greater" so BinaryHeap (max-heap) pops it first.
    fn cmp(&self, o: &Self) -> Ordering {
        o.f.total_cmp(&self.f)
            .then_with(|| o.counter.cmp(&self.counter))
    }
}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn find_path(
    c: &Corpus,
    start: usize,
    end: usize,
    length_hint: usize,
    max_expansions: usize,
    model: Option<&TransitionModel>,
    ctx: Option<&Context>,
) -> Option<Vec<usize>> {
    let mut counter: u64 = 0;
    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();
    heap.push(HeapItem {
        f: c.cos_dist(start, end),
        counter,
        g: 0.0,
        node: start,
        path: vec![start],
    });
    counter += 1;
    let mut best_g: HashMap<usize, f64> = HashMap::new();
    best_g.insert(start, 0.0);

    for _ in 0..max_expansions {
        let item = match heap.pop() {
            Some(x) => x,
            None => return None,
        };
        let HeapItem {
            g: g_cost,
            node,
            path,
            ..
        } = item;
        if node == end {
            return Some(path);
        }
        if g_cost > *best_g.get(&node).unwrap_or(&f64::INFINITY) {
            continue;
        }
        let prev_genre = &c.meta[node].genre;

        let cands: Vec<(usize, f64)> = c
            .neighbors(node)
            .into_iter()
            .filter(|&(nbr, _)| nbr == end || !violates(c, &path, nbr, length_hint))
            .collect();

        let mut trans_n: HashMap<usize, f64> = HashMap::new();
        let mut ctx_n: HashMap<usize, f64> = HashMap::new();
        if let Some(m) = model {
            if !cands.is_empty() {
                let raw: Vec<(usize, f64)> = cands
                    .iter()
                    .map(|&(nbr, _)| (nbr, m.affinity(&c.meta[node], &c.meta[nbr])))
                    .collect();
                trans_n = minmax(&raw);
                if let Some(cx) = ctx {
                    let raw_c: Vec<(usize, f64)> = cands
                        .iter()
                        .map(|&(nbr, _)| (nbr, m.context_fit(&c.meta[nbr], cx)))
                        .collect();
                    ctx_n = minmax(&raw_c);
                }
            }
        }

        for &(nbr, dist) in &cands {
            let mut step = W_DIST * dist;
            step += W_FIT * (1.0 - c.fit[nbr] as f64);
            if &c.meta[nbr].genre != prev_genre {
                step += W_DIV * GENRE_JUMP_PENALTY;
            }
            if !trans_n.is_empty() {
                step += W_TRANS * (1.0 - trans_n.get(&nbr).copied().unwrap_or(0.5));
            }
            if !ctx_n.is_empty() {
                step += W_CTX * (1.0 - ctx_n.get(&nbr).copied().unwrap_or(0.5));
            }
            let new_g = g_cost + step;
            if new_g >= *best_g.get(&nbr).unwrap_or(&f64::INFINITY) {
                continue;
            }
            best_g.insert(nbr, new_g);
            let mut np = path.clone();
            np.push(nbr);
            heap.push(HeapItem {
                f: new_g + c.cos_dist(nbr, end),
                counter,
                g: new_g,
                node: nbr,
                path: np,
            });
            counter += 1;
        }
    }
    None
}

pub fn densify(
    c: &Corpus,
    path: Vec<usize>,
    target_length: usize,
    model: Option<&TransitionModel>,
    ctx: Option<&Context>,
) -> Vec<usize> {
    let mut path = path;
    while path.len() < target_length {
        // widest remaining hop, descending
        let mut gaps: Vec<(f64, usize)> = path
            .windows(2)
            .enumerate()
            .map(|(i, w)| (c.cos_dist(w[0], w[1]), i))
            .collect();
        gaps.sort_by(|a, b| b.0.total_cmp(&a.0));

        let mut inserted = false;
        for &(_, i) in &gaps {
            let a = path[i];
            let b = path[i + 1];
            let gap = c.cos_dist(a, b);

            // candidate set = neighbors(a) ∪ neighbors(b)
            let mut cand_rows: Vec<usize> = Vec::new();
            for (nbr, _) in c.neighbors(a) {
                cand_rows.push(nbr);
            }
            for (nbr, _) in c.neighbors(b) {
                if !cand_rows.contains(&nbr) {
                    cand_rows.push(nbr);
                }
            }

            let mut valid: Vec<(usize, f64)> = Vec::new();
            for cc in cand_rows {
                if violates_insertion(c, &path, i, cc, target_length) {
                    continue;
                }
                let dac = c.cos_dist(a, cc);
                let dcb = c.cos_dist(cc, b);
                if dac >= gap || dcb >= gap {
                    continue;
                }
                valid.push((cc, dac + dcb - gap));
            }
            if valid.is_empty() {
                continue;
            }

            let mut trans_n: HashMap<usize, f64> = HashMap::new();
            let mut ctx_n: HashMap<usize, f64> = HashMap::new();
            if let Some(m) = model {
                let raw: Vec<(usize, f64)> = valid
                    .iter()
                    .map(|&(cc, _)| {
                        (
                            cc,
                            m.affinity(&c.meta[a], &c.meta[cc])
                                + m.affinity(&c.meta[cc], &c.meta[b]),
                        )
                    })
                    .collect();
                trans_n = minmax(&raw);
                if let Some(cx) = ctx {
                    let raw_c: Vec<(usize, f64)> =
                        valid.iter().map(|&(cc, _)| (cc, m.context_fit(&c.meta[cc], cx))).collect();
                    ctx_n = minmax(&raw_c);
                }
            }

            let mut best: Option<usize> = None;
            let mut best_cost = f64::INFINITY;
            for &(cc, detour) in &valid {
                let mut cost = W_DIST * detour + W_FIT * (1.0 - c.fit[cc] as f64);
                if !trans_n.is_empty() {
                    cost += W_TRANS * (1.0 - trans_n.get(&cc).copied().unwrap_or(0.5));
                }
                if !ctx_n.is_empty() {
                    cost += W_CTX * (1.0 - ctx_n.get(&cc).copied().unwrap_or(0.5));
                }
                if cost < best_cost {
                    best = Some(cc);
                    best_cost = cost;
                }
            }
            if let Some(cc) = best {
                path.insert(i + 1, cc);
                inserted = true;
                break;
            }
        }
        if !inserted {
            break;
        }
    }
    path
}
