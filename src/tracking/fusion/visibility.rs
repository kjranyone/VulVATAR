//! Keypoint visibility — the single answer to "is this detector keypoint
//! on the tracked person at all?".
//!
//! A top-down SimCC detector (RTMW3D) emits an argmax for every one of its
//! 133 keypoints on every frame. Keypoints that are physically outside the
//! frame (legs and hands under the desk in the bust-up envelope) or that
//! belong to nobody (an empty chair) still come out with a plausible
//! position and a mid confidence, because those joints carried no
//! supervision at training time. Downstream that used to be handled by a
//! stack of per-joint geometric gates (border clamps, leg / arm coherence,
//! reach filters). This module replaces them with two measured signals:
//!
//! 1. **SimCC peak shape** ([`VisPolicy`]): the second-peak ratio, half-max
//!    coverage and localisation σ of the raw heatmap vectors separate
//!    hallucinated joints from observed ones far better than the peak
//!    height alone (measured 2026-09-11 on 20 desk sessions + 3 frontal
//!    replays, see `diagnostics/visibility`). A logistic calibration turns
//!    them into a visibility probability.
//! 2. **Depth silhouette** ([`Silhouette`]): the aligned D435 depth gives
//!    the person's actual outline — a connected, depth-continuous surface
//!    grown from the face. A keypoint whose pixel is not within a few
//!    centimetres of that outline is not on this person, whatever the
//!    detector says. The outline also tells whether the body is truncated
//!    by the frame bottom (the hips and everything below are then out of
//!    frame by construction).

use crate::tracking::detector::DecodedJoint;

/// Logistic visibility calibration over the SimCC peak statistics.
///
/// `p = σ(w · [score, sxy, second, half, zscore, ln sxy] + b)` with
/// `sxy = max(sx, sy)`, `second = max(second_x, second_y)`,
/// `half = max(half_x, half_y)`. Coefficients come from
/// `diagnostics/visibility/analyze_vis.py` (session-split validation).
#[derive(Clone, Copy, Debug)]
pub struct VisPolicy {
    pub w: [f64; 6],
    pub bias: f64,
    /// Presence threshold: below this the keypoint does not count as
    /// evidence that a body part is there (self-track bbox, silhouette
    /// seeds, hand-crop seeding).
    pub min_p: f32,
    /// Observation threshold: below this the keypoint is not used as a
    /// residual at all. Kept separate from `min_p` so a softer policy
    /// (uncertain keypoints entering at inflated σ) can be benched;
    /// 0.15 was measured worse than 0.5 over 23 recordings (torso yaw
    /// std sum 128 vs 114, wrist snaps 152 vs 134), so both are 0.5.
    pub min_p_obs: f32,
}

impl Default for VisPolicy {
    fn default() -> Self {
        Self {
            // Full fit 2026-09-11 (analyze_vis.py): 125 984 labelled
            // keypoints, 23 recordings. Session-split validation AUC 0.985
            // (peak height alone 0.928); at 0.5: face / shoulder positives
            // kept 99.9 %, out-of-frame legs rejected 98.8 %.
            w: [49.8919, 9.6979, -11.0647, -3.5774, -26.6312, 3.5178],
            bias: 5.5525,
            min_p: 0.5,
            min_p_obs: 0.5,
        }
    }
}

impl VisPolicy {
    /// Visibility probability of one decoded keypoint.
    pub fn p_vis(&self, j: &DecodedJoint) -> f32 {
        if !(j.nx.is_finite() && j.ny.is_finite()) {
            return 0.0;
        }
        let sxy = j.sx.max(j.sy).max(1e-5) as f64;
        let second = j.second_x.max(j.second_y) as f64;
        let half = j.half_x.max(j.half_y) as f64;
        let x = [j.score as f64, sxy, second, half, j.zscore as f64, sxy.ln()];
        let z: f64 = self.w.iter().zip(x.iter()).map(|(w, x)| w * x).sum::<f64>() + self.bias;
        (1.0 / (1.0 + (-z).exp())) as f32
    }
}

/// Parameters of the depth silhouette.
#[derive(Clone, Copy, Debug)]
pub struct SilhouetteParams {
    /// Band in front of the face reference depth (m) — arms reaching
    /// toward the camera.
    pub z_front: f64,
    /// Band behind the face reference depth (m) — shoulders and back
    /// when leaning forward.
    pub z_back: f64,
    /// Maximum depth step between 4-neighbours inside the surface, per
    /// metre of reference depth (D435 noise grows with range).
    pub grad_per_m: f64,
    /// Maximum metric distance from the outline for a keypoint to count
    /// as on the person (joint centres sit inside the body; the detector
    /// jitters a few pixels).
    pub max_dist_m: f64,
    /// Same tolerance for a keypoint whose pixel has NO depth (a hole):
    /// a hole is absence of information, not evidence of absence — near
    /// hands lose depth constantly and cut holes into the shoulder they
    /// cover (measured: a shoulder 0.09–0.10 m from the outline, on a
    /// hole, dropped in 6 % of frames → one-shoulder torso yaw flips).
    pub max_dist_hole_m: f64,
    /// Minimum visible silhouette height (m, crown to lowest visible row)
    /// for the hips to be inside the frame at all. Adult crown-to-hip is
    /// ≈ 0.8 m; below this the whole leg chain is out of frame by
    /// construction (desk sessions measure 0.25–0.53 m).
    pub min_height_for_legs_m: f64,
    /// Area range (m²) of a depth-continuous blob inside the person band
    /// that is NOT connected to the main silhouette but still counts as
    /// the person's limb: a hand or hand + forearm entering from outside
    /// the frame, or held in front of the chest (the depth step to the
    /// torso breaks continuity). A hand is ≈ 0.01–0.02 m², hand + forearm
    /// ≈ 0.04; a desk plane or chair back is ≫ 0.1.
    pub limb_blob_area_m2: (f64, f64),
}

impl Default for SilhouetteParams {
    fn default() -> Self {
        Self {
            z_front: 0.60,
            z_back: 0.45,
            grad_per_m: 0.03,
            max_dist_m: 0.06,
            max_dist_hole_m: 0.20,
            min_height_for_legs_m: 0.65,
            limb_blob_area_m2: (0.003, 0.08),
        }
    }
}

/// The person's outline in the depth frame plus a chamfer distance field.
/// Built on a `scale`-times decimated grid (2 at 640x480): the outline
/// needs centimetre, not pixel, precision and the flood fill + distance
/// transform on the full 307 k pixels cost ~20 ms in the dev build.
#[derive(Clone, Debug)]
pub struct Silhouette {
    /// Decimation factor between the depth frame and the mask grid.
    pub scale: u32,
    /// Mask grid size (depth size / `scale`).
    pub width: u32,
    pub height: u32,
    /// Depth frame size.
    pub src_width: u32,
    pub src_height: u32,
    /// 1 where the pixel belongs to the person surface.
    pub mask: Vec<u8>,
    /// Chamfer (3-4) distance to the nearest mask pixel, in thirds of a
    /// pixel; 0 inside the mask.
    dist3: Vec<u32>,
    /// Candidate pixels (in band, depth-continuous) — the flood-fill
    /// domain, kept for the limb-blob query.
    cand: Vec<u8>,
    /// Lazily assigned blob label per candidate pixel (0 = unlabelled,
    /// 1 = main silhouette, ≥ 2 = detached blobs) and their pixel counts
    /// (`usize::MAX` = overflowed the query's area limit).
    blob: Vec<u32>,
    blob_px: Vec<usize>,
    fx: f64,
    /// Face reference depth (m).
    pub z_ref: f64,
    /// Pixels per metre at `z_ref`.
    pub px_per_m: f64,
    /// The surface reaches the bottom rows: the body continues below the
    /// frame.
    pub touches_bottom: bool,
    /// Approximate surface area (m²) at `z_ref`.
    pub area_m2: f64,
    /// Visible height (m) of the surface: lowest minus highest mask row
    /// at `z_ref`.
    pub height_m: f64,
}

impl Silhouette {
    /// Metric distance (m) from pixel `(u, v)` to the outline (0 inside).
    /// Pixels outside the frame are clamped to the border.
    pub fn dist_m(&self, u: f64, v: f64) -> f64 {
        let s = self.scale as f64;
        let x = ((u / s).round() as i64).clamp(0, self.width as i64 - 1) as usize;
        let y = ((v / s).round() as i64).clamp(0, self.height as i64 - 1) as usize;
        self.dist3[y * self.width as usize + x] as f64 / 3.0 / self.px_per_m
    }

    /// Mask grid index -> source-frame pixel index (top-left of the cell).
    fn src_index(&self, i: usize) -> usize {
        let (w, s) = (self.width as usize, self.scale as usize);
        let (x, y) = (i % w, i / w);
        (y * s) * self.src_width as usize + x * s
    }

    /// Whether pixel `(u, v)` is on the person (within `max_dist_m`).
    pub fn contains(&self, u: f64, v: f64, max_dist_m: f64) -> bool {
        self.dist_m(u, v) <= max_dist_m
    }

    /// Dense surface samples of the person: every `stride`-th pixel of the
    /// main silhouette (camera-space metres) with a D435-style range σ
    /// (`0.004 + 0.001·z²` m). Pixels for which `skip(u, v)` is true (hand
    /// crops) are left out. This is the primary metric observation of the
    /// visible body: tens of thousands of surface points pin the trunk,
    /// head and upper arms every frame, where a handful of 2-D keypoints
    /// could not.
    pub fn sample_points(
        &self,
        points: &[[f32; 3]],
        stride: usize,
        skip: &dyn Fn(f64, f64) -> bool,
    ) -> Vec<([f64; 3], f64)> {
        let (w, h) = (self.width as usize, self.height as usize);
        let s = self.scale as usize;
        // `stride` is in source pixels; walk the mask grid accordingly.
        let stride = (stride / s).max(1);
        let mut out = Vec::with_capacity(w * h / (stride * stride) / 3);
        // Offset the grid by half a stride so the samples are not biased
        // to the mask's top-left.
        let off = stride / 2;
        let mut y = off;
        while y < h {
            let mut x = off;
            while x < w {
                let i = y * w + x;
                if self.mask[i] == 1 && !skip((x * s) as f64, (y * s) as f64) {
                    let p = points[self.src_index(i)];
                    let z = p[2] as f64;
                    if z.is_finite() && z > 0.0 {
                        let sigma = 0.004 + 0.001 * z * z;
                        out.push(([p[0] as f64, p[1] as f64, z], sigma));
                    }
                }
                x += stride;
            }
            y += stride;
        }
        out
    }

    /// Area (m², at the blob's own depth) of the in-band depth-continuous
    /// blob under pixel `(u, v)` (nearest candidate pixel within 4 px), or
    /// `None` when there is no in-band depth there. The main silhouette
    /// reports its own area. Fills stop at `max_area_m2`: a larger blob
    /// reports `f64::INFINITY`.
    pub fn blob_area_m2(
        &mut self,
        points: &[[f32; 3]],
        u: f64,
        v: f64,
        max_area_m2: f64,
    ) -> Option<f64> {
        let (w, h) = (self.width as usize, self.height as usize);
        let s = self.scale as f64;
        let (cu, cv) = ((u / s).round() as i64, (v / s).round() as i64);
        let mut start = None;
        'outer: for r in 0..=2i64 {
            for dy in -r..=r {
                for dx in -r..=r {
                    if dx.abs() != r && dy.abs() != r {
                        continue;
                    }
                    let (x, y) = (cu + dx, cv + dy);
                    if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
                        continue;
                    }
                    let i = y as usize * w + x as usize;
                    if self.cand[i] == 1 {
                        start = Some(i);
                        break 'outer;
                    }
                }
            }
        }
        let start = start?;
        let z_blob = points[self.src_index(start)][2] as f64;
        let px_per_m = self.fx / z_blob.max(0.1) / self.scale as f64;
        let px2 = px_per_m * px_per_m;
        let max_px = (max_area_m2 * px2).ceil() as usize;
        if self.blob[start] != 0 {
            let n = self.blob_px[self.blob[start] as usize];
            return Some(if n == usize::MAX {
                f64::INFINITY
            } else {
                n as f64 / px2
            });
        }
        let label = self.blob_px.len() as u32;
        let mut queue: Vec<usize> = vec![start];
        self.blob[start] = label;
        let mut head = 0;
        let mut overflow = false;
        while head < queue.len() {
            if queue.len() > max_px {
                overflow = true;
                break;
            }
            let i = queue[head];
            head += 1;
            let (x, y) = (i % w, i / w);
            let mut nb = [usize::MAX; 4];
            if x > 0 {
                nb[0] = i - 1;
            }
            if x + 1 < w {
                nb[1] = i + 1;
            }
            if y > 0 {
                nb[2] = i - w;
            }
            if y + 1 < h {
                nb[3] = i + w;
            }
            for j in nb {
                if j != usize::MAX && self.cand[j] == 1 && self.blob[j] == 0 {
                    self.blob[j] = label;
                    queue.push(j);
                }
            }
        }
        if overflow {
            self.blob_px.push(usize::MAX);
            Some(f64::INFINITY)
        } else {
            self.blob_px.push(queue.len());
            Some(queue.len() as f64 / px2)
        }
    }
}

/// Median of the valid depths in a `(2r+1)²` window; `None` if fewer than
/// three valid pixels.
pub fn window_median_z(
    points: &[[f32; 3]],
    w: usize,
    h: usize,
    u: i64,
    v: i64,
    r: i64,
) -> Option<f64> {
    let mut zs: Vec<f32> = Vec::with_capacity(((2 * r + 1) * (2 * r + 1)) as usize);
    for dy in -r..=r {
        let y = v + dy;
        if y < 0 || y >= h as i64 {
            continue;
        }
        for dx in -r..=r {
            let x = u + dx;
            if x < 0 || x >= w as i64 {
                continue;
            }
            let z = points[y as usize * w + x as usize][2];
            if z.is_finite() && z > 0.0 {
                zs.push(z);
            }
        }
    }
    if zs.len() < 3 {
        return None;
    }
    zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(zs[zs.len() / 2] as f64)
}

/// Build the person silhouette from the aligned point cloud (`points[y*w+x]`
/// = camera-space metres, NaN / ≤0 for holes), the focal length `fx` and
/// the face seed pixels `(u, v)`. Returns `None` when no seed has depth.
pub fn build_silhouette(
    points: &[[f32; 3]],
    width: u32,
    height: u32,
    fx: f64,
    seeds: &[(f64, f64)],
    p: &SilhouetteParams,
) -> Option<Silhouette> {
    let (sw, sh) = (width as usize, height as usize);
    if points.len() != sw * sh || sw == 0 || sh == 0 {
        return None;
    }
    let scale: usize = if sw >= 1000 { 3 } else { 2 };
    let (w, h) = (sw / scale, sh / scale);
    // Reference depth: median over the seeds' local medians (source res).
    let mut refs: Vec<f64> = Vec::with_capacity(seeds.len());
    let mut seed_px: Vec<(usize, usize)> = Vec::with_capacity(seeds.len());
    for &(u, v) in seeds {
        let (ui, vi) = (u.round() as i64, v.round() as i64);
        if ui < 0 || vi < 0 || ui >= sw as i64 || vi >= sh as i64 {
            continue;
        }
        if let Some(z) = window_median_z(points, sw, sh, ui, vi, 2) {
            refs.push(z);
            seed_px.push((ui as usize / scale, vi as usize / scale));
        }
    }
    // Decimated depth: one sample per cell (top-left pixel).
    let zs: Vec<f32> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            points[(y * scale) * sw + x * scale][2]
        })
        .collect();
    if refs.is_empty() {
        return None;
    }
    refs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let z_ref = refs[refs.len() / 2];
    let z_lo = (z_ref - p.z_front) as f32;
    let z_hi = (z_ref + p.z_back) as f32;
    let step = (p.grad_per_m * z_ref.max(0.3)) as f32;

    // Candidate cells: in band and depth-continuous with left / up. The
    // step is per source pixel; across a `scale`-cell it scales up.
    let step = step * scale as f32;
    let mut cand = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let z = zs[i];
            if !(z.is_finite() && z > z_lo && z < z_hi) {
                continue;
            }
            let mut ok = true;
            if x > 0 {
                let zl = zs[i - 1];
                if zl.is_finite() && zl > 0.0 && (z - zl).abs() >= step {
                    ok = false;
                }
            }
            if ok && y > 0 {
                let zu = zs[i - w];
                if zu.is_finite() && zu > 0.0 && (z - zu).abs() >= step {
                    ok = false;
                }
            }
            cand[i] = ok as u8;
        }
    }
    // Flood fill (4-connected) from every seed that lands on a candidate
    // pixel, or on one within a 5×5 neighbourhood (the seed itself may be
    // a hole on glasses / hair).
    let mut mask = vec![0u8; w * h];
    let mut queue: Vec<usize> = Vec::with_capacity(w * h / 8);
    for &(su, sv) in &seed_px {
        let mut start = None;
        'outer: for r in 0..=2i64 {
            for dy in -r..=r {
                for dx in -r..=r {
                    let (x, y) = (su as i64 + dx, sv as i64 + dy);
                    if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
                        continue;
                    }
                    let i = y as usize * w + x as usize;
                    if cand[i] == 1 {
                        start = Some(i);
                        break 'outer;
                    }
                }
            }
        }
        if let Some(i) = start {
            if mask[i] == 0 {
                mask[i] = 1;
                queue.push(i);
            }
        }
    }
    if queue.is_empty() {
        return None;
    }
    let mut head = 0;
    while head < queue.len() {
        let i = queue[head];
        head += 1;
        let (x, y) = (i % w, i / w);
        let mut push = |j: usize| {
            if cand[j] == 1 && mask[j] == 0 {
                mask[j] = 1;
                queue.push(j);
            }
        };
        if x > 0 {
            push(i - 1);
        }
        if x + 1 < w {
            push(i + 1);
        }
        if y > 0 {
            push(i - w);
        }
        if y + 1 < h {
            push(i + w);
        }
    }
    let n_mask = queue.len();
    let touches_bottom = mask[(h - 2) * w..].iter().any(|&m| m == 1);
    // Pixels per metre on the MASK grid.
    let px_per_m = fx / z_ref / scale as f64;
    let area_m2 = n_mask as f64 / (px_per_m * px_per_m);
    let (mut top, mut bottom) = (usize::MAX, 0usize);
    for &i in &queue {
        let y = i / w;
        top = top.min(y);
        bottom = bottom.max(y);
    }
    let height_m = (bottom - top) as f64 / px_per_m;

    // Chamfer 3-4 distance transform (two passes).
    const INF: u32 = u32::MAX / 4;
    let mut d: Vec<u32> = mask.iter().map(|&m| if m == 1 { 0 } else { INF }).collect();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if d[i] == 0 {
                continue;
            }
            let mut best = d[i];
            if x > 0 {
                best = best.min(d[i - 1] + 3);
            }
            if y > 0 {
                best = best.min(d[i - w] + 3);
                if x > 0 {
                    best = best.min(d[i - w - 1] + 4);
                }
                if x + 1 < w {
                    best = best.min(d[i - w + 1] + 4);
                }
            }
            d[i] = best;
        }
    }
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let i = y * w + x;
            if d[i] == 0 {
                continue;
            }
            let mut best = d[i];
            if x + 1 < w {
                best = best.min(d[i + 1] + 3);
            }
            if y + 1 < h {
                best = best.min(d[i + w] + 3);
                if x + 1 < w {
                    best = best.min(d[i + w + 1] + 4);
                }
                if x > 0 {
                    best = best.min(d[i + w - 1] + 4);
                }
            }
            d[i] = best;
        }
    }
    // Blob labels: the main silhouette is label 1 (already filled).
    let mut blob = vec![0u32; w * h];
    for &i in &queue {
        blob[i] = 1;
    }
    Some(Silhouette {
        scale: scale as u32,
        width: w as u32,
        height: h as u32,
        src_width: width,
        src_height: height,
        mask,
        dist3: d,
        cand,
        blob,
        blob_px: vec![0, n_mask],
        fx,
        z_ref,
        px_per_m,
        touches_bottom,
        area_m2,
        height_m,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic scene: a wall at 1.2 m filling the frame, a "person"
    /// rectangle at 0.6 m from x 40..100, y 20..120 (touching the bottom),
    /// and a "desk" at 0.9 m in the bottom-left corner connected to
    /// nothing at the person's depth.
    fn scene() -> (Vec<[f32; 3]>, u32, u32) {
        let (w, h) = (160usize, 120usize);
        let fx = 300.0f32;
        let mut pts = vec![[0.0f32; 3]; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut z = 1.2f32;
                if (40..100).contains(&x) && y >= 20 {
                    z = 0.6;
                }
                if x < 30 && y >= 100 {
                    z = 0.9;
                }
                // a hole strip on the person's "glasses"
                if (60..80).contains(&x) && (30..33).contains(&y) {
                    pts[y * w + x] = [f32::NAN; 3];
                    continue;
                }
                pts[y * w + x] = [(x as f32 - 80.0) / fx * z, (y as f32 - 60.0) / fx * z, z];
            }
        }
        (pts, w as u32, h as u32)
    }

    #[test]
    fn silhouette_is_the_person_only() {
        let (pts, w, h) = scene();
        let s = build_silhouette(
            &pts,
            w,
            h,
            300.0,
            &[(70.0, 31.0), (66.0, 40.0)],
            &SilhouetteParams::default(),
        )
        .expect("seeds have depth");
        assert!((s.z_ref - 0.6).abs() < 1e-3);
        assert!(s.touches_bottom);
        // Inside the person.
        assert!(s.contains(70.0, 80.0, 0.0));
        // Wall pixel next to the person: outside, ~1 px away → dist ≈ 1/500 m.
        assert!(!s.contains(101.0, 80.0, 0.0));
        assert!(s.contains(101.0, 80.0, 0.06));
        // Far wall and desk: outside even with the tolerance.
        assert!(!s.contains(140.0, 60.0, 0.06));
        assert!(!s.contains(10.0, 110.0, 0.06));
        // Area: 60×100 px at 500 px/m → 0.024 m² (minus the hole strip).
        assert!((s.area_m2 - 0.024).abs() < 0.002, "area {}", s.area_m2);
        // 100 px tall at 500 px/m.
        assert!((s.height_m - 0.2).abs() < 0.01, "height {}", s.height_m);
    }

    #[test]
    fn detached_hand_blob_is_accepted_but_a_desk_plane_is_not() {
        let (mut pts, w, h) = scene();
        let fx = 300.0f32;
        let wu = w as usize;
        // A "hand" at 0.45 m (in band, in front of the chest) detached
        // from the person: 30×40 px at 667 px/m → 0.0027 m².
        for y in 40..80 {
            for x in 110..140 {
                let z = 0.45f32;
                pts[y * wu + x] = [(x as f32 - 80.0) / fx * z, (y as f32 - 60.0) / fx * z, z];
            }
        }
        // A "desk" plane at 0.7 m across the whole bottom: y 100..120.
        for y in 100..120 {
            for x in 0..wu {
                let z = 0.7f32;
                pts[y * wu + x] = [(x as f32 - 80.0) / fx * z, (y as f32 - 60.0) / fx * z, z];
            }
        }
        // The synthetic scene is tiny (160×120 px): scale the limb range
        // so the 0.0027 m² hand passes and the 0.0175 m² desk overflows.
        let p = SilhouetteParams {
            limb_blob_area_m2: (0.002, 0.01),
            ..Default::default()
        };
        let mut s = build_silhouette(&pts, w, h, 300.0, &[(70.0, 40.0)], &p).unwrap();
        assert!(!s.contains(125.0, 60.0, 0.03));
        let a = s
            .blob_area_m2(&pts, 125.0, 60.0, p.limb_blob_area_m2.1)
            .unwrap();
        assert!(a > 0.002 && a < 0.004, "hand blob {a}");
        // Same blob queried again: cached, same answer.
        assert_eq!(
            s.blob_area_m2(&pts, 130.0, 70.0, p.limb_blob_area_m2.1)
                .unwrap(),
            a
        );
        // Desk: overflows the limit.
        let d = s
            .blob_area_m2(&pts, 20.0, 110.0, p.limb_blob_area_m2.1)
            .unwrap();
        assert!(d.is_infinite(), "desk {d}");
        // Main silhouette reports its own area.
        let m = s.blob_area_m2(&pts, 70.0, 80.0, 1.0).unwrap();
        assert!((m - s.area_m2).abs() < 1e-6);
        // Wall (out of band): no blob.
        assert!(s.blob_area_m2(&pts, 20.0, 30.0, 1.0).is_none());
    }

    #[test]
    fn wall_in_band_but_behind_a_step_is_excluded() {
        // Wall at 0.9 m is inside z_ref + z_back but separated by a 0.3 m step.
        let (mut pts, w, h) = scene();
        for p in pts.iter_mut() {
            if (p[2] - 1.2).abs() < 1e-3 {
                let s = 0.9 / 1.2;
                *p = [p[0] * s, p[1] * s, 0.9];
            }
        }
        let s = build_silhouette(
            &pts,
            w,
            h,
            300.0,
            &[(70.0, 40.0)],
            &SilhouetteParams::default(),
        )
        .unwrap();
        assert!(!s.contains(140.0, 60.0, 0.03));
        assert!(s.contains(70.0, 80.0, 0.0));
    }

    #[test]
    fn no_depth_under_seeds_gives_none() {
        let (pts, w, h) = scene();
        assert!(build_silhouette(
            &pts,
            w,
            h,
            300.0,
            &[(70.0, 31.0)],
            &SilhouetteParams::default()
        )
        .is_some());
        let mut holes = pts.clone();
        for p in holes.iter_mut() {
            *p = [f32::NAN; 3];
        }
        assert!(build_silhouette(
            &holes,
            w,
            h,
            300.0,
            &[(70.0, 40.0)],
            &SilhouetteParams::default()
        )
        .is_none());
    }

    #[test]
    fn p_vis_orders_sharp_over_flat_peaks() {
        let pol = VisPolicy::default();
        let sharp = DecodedJoint {
            score: 0.71,
            sx: 0.005,
            sy: 0.007,
            second_x: 0.1,
            second_y: 0.1,
            half_x: 0.035,
            half_y: 0.03,
            zscore: 0.63,
            ..Default::default()
        };
        let flat = DecodedJoint {
            score: 0.56,
            sx: 0.013,
            sy: 0.010,
            second_x: 0.7,
            second_y: 0.65,
            half_x: 0.16,
            half_y: 0.13,
            zscore: 0.55,
            ..Default::default()
        };
        assert!(pol.p_vis(&sharp) > 0.9, "{}", pol.p_vis(&sharp));
        assert!(pol.p_vis(&flat) < 0.1, "{}", pol.p_vis(&flat));
    }
}
