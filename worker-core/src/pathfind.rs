//! Faithful port of pathfinder/search.py (A* + densify) and the cost weights
//! from pathfinder/config.py. Operates on corpus row indices.

use crate::snapshot::{Corpus, Meta};
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

// Sentinel node ids for the two off-map ("virtual") endpoints. They never enter
// a `path` Vec — the search path stays corpus-only — so densify/violates and the
// flat corpus arrays are never indexed by them. V_START is only ever the seed
// node (expands to the off-map start's KNN); V_END is only ever a goal candidate
// reachable from the off-map end's KNN rows.
const V_END: usize = usize::MAX;
const V_START: usize = usize::MAX - 1;

/// An A* endpoint: an in-corpus `row`, or an off-map ("virtual") node carrying
/// its own latent vector, fit, metadata, and precomputed KNN into the corpus.
/// `Corpus`-only on both ends reproduces the original corpus↔corpus search
/// exactly (the heuristic and costs reduce to the corpus-row forms).
pub enum EndSpec<'a> {
    Corpus(usize),
    Virtual {
        vec: &'a [f32],
        fit: f64,
        meta: Meta,
        knn: Vec<(usize, f64)>,
    },
}

/// cosine distance between two normalized vectors of equal length.
fn cos_dist_vv(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f64;
    for i in 0..n {
        dot += (a[i] * b[i]) as f64;
    }
    1.0 - dot
}

// Metadata for a node, handling the two virtual sentinels. The fallthrough is
// never reached for a sentinel unless its spec is Virtual, so c.meta is never
// indexed out of range.
fn vmeta<'e>(c: &'e Corpus, start: &'e EndSpec, end: &'e EndSpec, node: usize) -> &'e Meta {
    if node == V_START {
        if let EndSpec::Virtual { meta, .. } = start {
            return meta;
        }
    } else if node == V_END {
        if let EndSpec::Virtual { meta, .. } = end {
            return meta;
        }
    }
    &c.meta[node]
}

// Rotation-fit for a node. V_END uses the off-map end's fit; V_START is never a
// candidate so its fit is never read.
fn vfit(c: &Corpus, end: &EndSpec, node: usize) -> f64 {
    if node == V_END {
        if let EndSpec::Virtual { fit, .. } = end {
            return *fit;
        }
    }
    c.fit[node] as f64
}

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
    find_path_open(
        c,
        &EndSpec::Corpus(start),
        &EndSpec::Corpus(end),
        length_hint,
        max_expansions,
        model,
        ctx,
    )
}

/// A* with arbitrary endpoints (corpus rows or off-map virtual nodes). Returns
/// the **corpus-only** path of rows; the caller prepends/appends any virtual
/// endpoints. With two `Corpus` endpoints this is identical to the original
/// corpus↔corpus search: `end_knn` is empty (no V_END edge), the heuristic
/// `cos_dist_vec(node, vec_at(end))` equals `cos_dist(node, end)`, and the seed
/// is the single start row — so the parity test still gates this code path.
#[allow(clippy::too_many_arguments)]
pub fn find_path_open(
    c: &Corpus,
    start: &EndSpec,
    end: &EndSpec,
    length_hint: usize,
    max_expansions: usize,
    model: Option<&TransitionModel>,
    ctx: Option<&Context>,
) -> Option<Vec<usize>> {
    // Goal vector for the heuristic: the corpus row's vector, or the off-map
    // latent. For a corpus end, cos_dist_vec(node, goal) == cos_dist(node, end).
    let goal_vec: &[f32] = match end {
        EndSpec::Corpus(r) => c.vec_at(*r),
        EndSpec::Virtual { vec, .. } => vec,
    };
    let end_row: Option<usize> = match end {
        EndSpec::Corpus(r) => Some(*r),
        EndSpec::Virtual { .. } => None,
    };
    // For an off-map end: corpus rows from which it is reachable, plus the cosine
    // distance of that closing hop. Empty for a corpus end (no V_END candidate).
    let end_knn: HashMap<usize, f64> = match end {
        EndSpec::Virtual { knn, .. } => knn.iter().copied().collect(),
        EndSpec::Corpus(_) => HashMap::new(),
    };
    let is_goal = |nbr: usize| match end_row {
        Some(r) => nbr == r,
        None => nbr == V_END,
    };
    // h(node): cosine distance from a node's vector to the goal vector.
    let heur = |node: usize| -> f64 {
        match node {
            V_END => 0.0,
            V_START => match start {
                EndSpec::Virtual { vec, .. } => cos_dist_vv(vec, goal_vec),
                EndSpec::Corpus(_) => 0.0,
            },
            _ => c.cos_dist_vec(node, goal_vec),
        }
    };

    let mut counter: u64 = 0;
    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();
    let mut best_g: HashMap<usize, f64> = HashMap::new();

    let start_node = match start {
        EndSpec::Corpus(sr) => *sr,
        EndSpec::Virtual { .. } => V_START,
    };
    // The seed path is corpus-only: a corpus start seeds [start]; an off-map
    // start seeds [] and expands to corpus rows (V_START never enters the path).
    let start_path: Vec<usize> = match start {
        EndSpec::Corpus(sr) => vec![*sr],
        EndSpec::Virtual { .. } => Vec::new(),
    };
    heap.push(HeapItem {
        f: heur(start_node),
        counter,
        g: 0.0,
        node: start_node,
        path: start_path,
    });
    counter += 1;
    best_g.insert(start_node, 0.0);

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
        if is_goal(node) {
            return Some(path);
        }
        if g_cost > *best_g.get(&node).unwrap_or(&f64::INFINITY) {
            continue;
        }
        let prev_genre = &vmeta(c, start, end, node).genre;

        // candidates: the off-map start expands to its KNN; a corpus node uses
        // its baked neighbors, plus the V_END closing edge when it's one of the
        // off-map end's nearest rows. The goal is exempt from path constraints.
        let cands: Vec<(usize, f64)> = match (node, start) {
            (V_START, EndSpec::Virtual { knn, .. }) => knn.clone(),
            _ => {
                let mut v = c.neighbors(node);
                if let Some(&d) = end_knn.get(&node) {
                    v.push((V_END, d));
                }
                v
            }
        }
        .into_iter()
        .filter(|&(nbr, _)| is_goal(nbr) || !violates(c, &path, nbr, length_hint))
        .collect();

        let mut trans_n: HashMap<usize, f64> = HashMap::new();
        let mut ctx_n: HashMap<usize, f64> = HashMap::new();
        if let Some(m) = model {
            if !cands.is_empty() {
                let prev_meta = vmeta(c, start, end, node);
                let raw: Vec<(usize, f64)> = cands
                    .iter()
                    .map(|&(nbr, _)| (nbr, m.affinity(prev_meta, vmeta(c, start, end, nbr))))
                    .collect();
                trans_n = minmax(&raw);
                if let Some(cx) = ctx {
                    let raw_c: Vec<(usize, f64)> = cands
                        .iter()
                        .map(|&(nbr, _)| (nbr, m.context_fit(vmeta(c, start, end, nbr), cx)))
                        .collect();
                    ctx_n = minmax(&raw_c);
                }
            }
        }

        for &(nbr, dist) in &cands {
            let mut step = W_DIST * dist;
            step += W_FIT * (1.0 - vfit(c, end, nbr));
            if &vmeta(c, start, end, nbr).genre != prev_genre {
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
            // V_END is the goal marker only — it never enters the path.
            let mut np = path.clone();
            if nbr != V_END {
                np.push(nbr);
            }
            heap.push(HeapItem {
                f: new_g + heur(nbr),
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

/// A node in a fixed set to be re-ordered (Bring-Your-Own-Playlist). Carries
/// everything the step-cost needs. Unlike A*, no node may be dropped — the whole
/// user set must survive — so there are no hard `violates()` constraints here;
/// ordering only *minimizes* the same cost A* weighs.
pub struct OrderNode<'a> {
    pub vec: &'a [f32],
    pub fit: f64,
    pub meta: &'a Meta,
}

/// Re-order a fixed set of tracks into the lowest-cost open path — an asymmetric
/// open-TSP over the *same* step-cost as `find_path_open` (cosine distance, fit,
/// genre-jump, transition affinity, time-of-day context). Returns a permutation
/// of `0..nodes.len()`. Transition affinity and context fit are min-max
/// normalized globally (across all ordered pairs / all nodes) — the fixed-set
/// analog of A*'s per-expansion normalization. Construction is greedy
/// nearest-neighbor from the best of all seeds, then 2-opt to convergence; the
/// caller caps N (playlist length), so the O(N^2) cost matrix and
/// O(N^2)-per-pass 2-opt stay cheap.
pub fn order_fixed(
    nodes: &[OrderNode],
    model: Option<&TransitionModel>,
    ctx: Option<&Context>,
) -> Vec<usize> {
    let n = nodes.len();
    if n <= 2 {
        return (0..n).collect();
    }

    // Global min-max normalization of the transition term over all ordered pairs
    // (flat n*n; entry i*n+j is the normalized affinity of the hop i→j), and of
    // the context term over all nodes. Empty when there's no transition model.
    let mut trans: Vec<f64> = Vec::new();
    let mut ctxn: Vec<f64> = Vec::new();
    if let Some(m) = model {
        let mut raw = vec![0.0f64; n * n];
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let a = m.affinity(nodes[i].meta, nodes[j].meta);
                    raw[i * n + j] = a;
                    lo = lo.min(a);
                    hi = hi.max(a);
                }
            }
        }
        let span = hi - lo;
        trans = raw
            .iter()
            .map(|&v| if span > 0.0 { (v - lo) / span } else { 0.5 })
            .collect();
        if let Some(cx) = ctx {
            let raw_c: Vec<(usize, f64)> =
                (0..n).map(|j| (j, m.context_fit(nodes[j].meta, cx))).collect();
            let norm = minmax(&raw_c);
            ctxn = (0..n).map(|j| norm.get(&j).copied().unwrap_or(0.5)).collect();
        }
    }

    // Precompute the asymmetric cost matrix once. cm[i*n+j] = cost of placing j
    // immediately after i — the same weighted terms as the A* step cost.
    let mut cm = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let mut step = W_DIST * cos_dist_vv(nodes[i].vec, nodes[j].vec);
            step += W_FIT * (1.0 - nodes[j].fit);
            if nodes[i].meta.genre != nodes[j].meta.genre {
                step += W_DIV * GENRE_JUMP_PENALTY;
            }
            if !trans.is_empty() {
                step += W_TRANS * (1.0 - trans[i * n + j]);
            }
            if !ctxn.is_empty() {
                step += W_CTX * (1.0 - ctxn[j]);
            }
            cm[i * n + j] = step;
        }
    }
    let tour_cost = |order: &[usize]| -> f64 {
        order.windows(2).map(|w| cm[w[0] * n + w[1]]).sum()
    };

    // Greedy nearest-neighbor from every seed; keep the cheapest construction.
    let mut best_order: Vec<usize> = (0..n).collect();
    let mut best_cost = f64::INFINITY;
    for seed in 0..n {
        let mut used = vec![false; n];
        let mut order = Vec::with_capacity(n);
        order.push(seed);
        used[seed] = true;
        for _ in 1..n {
            let last = *order.last().unwrap();
            let mut nx = 0usize;
            let mut nx_cost = f64::INFINITY;
            for j in 0..n {
                if used[j] {
                    continue;
                }
                let c = cm[last * n + j];
                if c < nx_cost {
                    nx_cost = c;
                    nx = j;
                }
            }
            order.push(nx);
            used[nx] = true;
        }
        let tc = tour_cost(&order);
        if tc < best_cost {
            best_cost = tc;
            best_order = order;
        }
    }

    // 2-opt for an open path: reverse best_order[i..=j] whenever it lowers the
    // total cost. The cost is asymmetric, so re-score the whole tour per move
    // (cheap at these N). Iterate to convergence with a hard pass cap.
    let mut improved = true;
    let mut passes = 0;
    while improved && passes < 60 {
        improved = false;
        passes += 1;
        for i in 1..n - 1 {
            for j in i + 1..n {
                let mut cand = best_order.clone();
                cand[i..=j].reverse();
                let tc = tour_cost(&cand);
                if tc + 1e-12 < best_cost {
                    best_cost = tc;
                    best_order = cand;
                    improved = true;
                }
            }
        }
    }
    best_order
}
