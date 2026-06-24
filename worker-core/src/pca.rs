//! 3D projection matching server/app.py `local_positions`: center, take the
//! top-3 principal axes (SVD right singular vectors == top eigenvectors of the
//! covariance), project, then scale by the 90th-percentile radius to ±8.
//!
//! Top eigenvectors are found by power iteration with deflation on the
//! dim×dim covariance (dim=64, point count ~100 → trivially cheap). Sign of an
//! axis is arbitrary in both numpy SVD and here; it only flips view orientation.

/// Plain local PCA over all `vectors` (splash / explore fallback).
pub fn project3d(vectors: &[&[f32]], dim: usize) -> Vec<[f32; 3]> {
    project3d_anchored(vectors, dim, vectors.len())
}

/// Path-anchored PCA (3 axes for the map). See [`project_anchored4`].
pub fn project3d_anchored(vectors: &[&[f32]], dim: usize, anchor_count: usize) -> Vec<[f32; 3]> {
    project_anchored4(vectors, dim, anchor_count)
        .into_iter()
        .map(|c| [c[0], c[1], c[2]])
        .collect()
}

/// Path-anchored PCA returning the 3 mapped axes *plus* the 4th principal
/// coordinate per point — the largest slice of variance the map discards. The
/// `w` value (same world scale as x/y/z) is what the "4th-dimension shadow"
/// renders, surfacing how far each stop sits in the direction the map flattened.
pub fn project4d_anchored(
    vectors: &[&[f32]],
    dim: usize,
    anchor_count: usize,
) -> (Vec<[f32; 3]>, Vec<f32>) {
    let c4 = project_anchored4(vectors, dim, anchor_count);
    let xyz = c4.iter().map(|c| [c[0], c[1], c[2]]).collect();
    let w = c4.iter().map(|c| c[3]).collect();
    (xyz, w)
}

/// Path-anchored PCA core: the principal frame (mean, axes, scale) is derived
/// from only the first `anchor_count` vectors — the route's path — and then
/// *all* vectors are projected into it. This makes the path the thing rendered
/// faithfully (its own variance picks the axes, so it spreads out as a coherent
/// thread instead of folding into crossings); the cloud rides along as context.
/// `anchor_count == vectors.len()` reduces exactly to plain local PCA. Returns 4
/// components per point (the 4th is the shadow dimension); the scale is fixed
/// from the first 3 so x/y/z stay identical to the pure-3D projection.
fn project_anchored4(vectors: &[&[f32]], dim: usize, anchor_count: usize) -> Vec<[f32; 4]> {
    let m = vectors.len();
    if m == 0 {
        return Vec::new();
    }
    // The anchor subset defines the frame; clamp to a sane, non-empty range.
    let anchor = anchor_count.clamp(1, m);
    // mean over the anchor subset
    let mut mean = vec![0.0f64; dim];
    for v in &vectors[..anchor] {
        for i in 0..dim {
            mean[i] += v[i] as f64;
        }
    }
    for x in mean.iter_mut() {
        *x /= anchor as f64;
    }
    // centered matrix — all rows centered by the anchor mean (anchor rows first)
    let mut centered = vec![0.0f64; m * dim];
    for (r, v) in vectors.iter().enumerate() {
        for i in 0..dim {
            centered[r * dim + i] = v[i] as f64 - mean[i];
        }
    }
    // covariance C = A^T A over the anchor rows only  (dim×dim, symmetric)
    let mut cov = vec![0.0f64; dim * dim];
    for r in 0..anchor {
        let row = &centered[r * dim..(r + 1) * dim];
        for i in 0..dim {
            let ri = row[i];
            if ri == 0.0 {
                continue;
            }
            for j in i..dim {
                cov[i * dim + j] += ri * row[j];
            }
        }
    }
    for i in 0..dim {
        for j in i + 1..dim {
            cov[j * dim + i] = cov[i * dim + j];
        }
    }

    // top-4 eigenvectors via power iteration + deflation (axes 0..3 are the map,
    // axis 3 is the shadow dimension). Computing the 4th leaves the first 3
    // unchanged, so the mapped x/y/z match the pure-3D projection exactly.
    let mut axes: Vec<Vec<f64>> = Vec::new();
    let ncomp = 4.min(dim);
    for comp in 0..ncomp {
        // deterministic dense init (avoids RNG; non-orthogonal to dominant space)
        let mut v: Vec<f64> = (0..dim)
            .map(|j| ((j as f64 + 1.0) * (comp as f64 + 1.3)).sin() + 0.1)
            .collect();
        orthonormalize(&mut v, &axes);
        for _ in 0..100 {
            let mut nv = matvec(&cov, &v, dim);
            orthonormalize(&mut nv, &axes);
            let norm = l2(&nv);
            if norm < 1e-12 {
                break;
            }
            for x in nv.iter_mut() {
                *x /= norm;
            }
            v = nv;
        }
        axes.push(v);
    }

    // coords = centered @ axes^T   (m×4)
    let mut coords = vec![[0.0f32; 4]; m];
    for r in 0..m {
        let row = &centered[r * dim..(r + 1) * dim];
        for (a, ax) in axes.iter().enumerate() {
            let mut s = 0.0f64;
            for i in 0..dim {
                s += row[i] * ax[i];
            }
            coords[r][a] = s as f32;
        }
    }

    // scale by the anchor subset's 90th-percentile radius (over the 3 mapped
    // axes only) -> ±8, so the path fills the view regardless of how far the
    // cloud happens to spread. The 4th coordinate rides the same scale.
    let mut norms: Vec<f64> = coords[..anchor]
        .iter()
        .map(|c| ((c[0] * c[0] + c[1] * c[1] + c[2] * c[2]) as f64).sqrt())
        .collect();
    let mut scale = percentile90(&mut norms);
    if scale <= 1e-6 {
        scale = 1.0;
    }
    let k = (1.0 / scale * 8.0) as f32;
    for c in coords.iter_mut() {
        c[0] *= k;
        c[1] *= k;
        c[2] *= k;
        c[3] *= k;
    }
    coords
}

fn matvec(a: &[f64], v: &[f64], dim: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; dim];
    for i in 0..dim {
        let mut s = 0.0;
        let base = i * dim;
        for j in 0..dim {
            s += a[base + j] * v[j];
        }
        out[i] = s;
    }
    out
}

fn l2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn orthonormalize(v: &mut [f64], basis: &[Vec<f64>]) {
    for b in basis {
        let dot: f64 = v.iter().zip(b).map(|(x, y)| x * y).sum();
        for (x, y) in v.iter_mut().zip(b) {
            *x -= dot * y;
        }
    }
}

/// numpy-style linear-interpolated 90th percentile.
fn percentile90(values: &mut [f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return values[0];
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let rank = 0.9 * (n as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let frac = rank - lo as f64;
    values[lo] + (values[hi] - values[lo]) * frac
}
