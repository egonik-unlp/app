//! 3D projection matching server/app.py `local_positions`: center, take the
//! top-3 principal axes (SVD right singular vectors == top eigenvectors of the
//! covariance), project, then scale by the 90th-percentile radius to ±8.
//!
//! Top eigenvectors are found by power iteration with deflation on the
//! dim×dim covariance (dim=64, point count ~100 → trivially cheap). Sign of an
//! axis is arbitrary in both numpy SVD and here; it only flips view orientation.

pub fn project3d(vectors: &[&[f32]], dim: usize) -> Vec<[f32; 3]> {
    let m = vectors.len();
    if m == 0 {
        return Vec::new();
    }
    // mean
    let mut mean = vec![0.0f64; dim];
    for v in vectors {
        for i in 0..dim {
            mean[i] += v[i] as f64;
        }
    }
    for x in mean.iter_mut() {
        *x /= m as f64;
    }
    // centered matrix
    let mut centered = vec![0.0f64; m * dim];
    for (r, v) in vectors.iter().enumerate() {
        for i in 0..dim {
            centered[r * dim + i] = v[i] as f64 - mean[i];
        }
    }
    // covariance C = centered^T centered  (dim×dim, symmetric)
    let mut cov = vec![0.0f64; dim * dim];
    for r in 0..m {
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

    // top-3 eigenvectors via power iteration + deflation
    let mut axes: Vec<Vec<f64>> = Vec::new();
    let ncomp = 3.min(dim);
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

    // coords = centered @ axes^T   (m×3)
    let mut coords = vec![[0.0f32; 3]; m];
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

    // scale by 90th-percentile radius -> ±8
    let mut norms: Vec<f64> = coords
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
