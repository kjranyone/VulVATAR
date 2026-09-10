//! Offline analysis of a `VULVATAR_RECORD` session — turns the per-frame
//! `pose.jsonl` written by `tracking::session_record` into a verdict about
//! *which stage* produced a misbehaving body.
//!
//!   cargo run --bin analyze_session -- diagnostics/session_<unix>
//!   cargo run --bin analyze_session                # newest session in diagnostics/
//!
//! Writes `summary.md` next to the recording and prints the same text.
//!
//! The report is built around the one question a "the avatar went crazy"
//! report cannot answer on its own: *was the bad joint measured or
//! invented?* Every recorded joint carries its [`JointOrigin`] tag, so the
//! jump table can separate
//!
//!   * `E` (Extrapolated) — the depth sample was a hole and the joint was
//!     pinned one anthropometric bone-length along its 2D ray. A cluster of
//!     jumps here means the *bone lengths* are wrong, which points at
//!     `reference_span_m` (calibration scale), not at the detector.
//!   * `O` (Observed) — the sensor really returned that depth. Jumps here
//!     are depth-sampling / person-masking failures.
//!   * `S` (Synthesized) — a fabricated canonical pair; should never move
//!     fast at all.
//!
//! plus two whole-body failure modes that masquerade as "wild motion":
//! a left/right block transposition (shoulder x-order flipping frame to
//! frame) and a moving global scale (`reference_span_m` / `mpsu` drifting,
//! which zooms and re-proportions the entire skeleton at once).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Joints whose per-frame motion is worth tabulating. The finger chain is
/// excluded — it is dense, noisy, and never the thing someone means by
/// "the body moved in an incredible direction".
const TRACKED: &[&str] = &[
    "Hips",
    "Spine",
    "Chest",
    "UpperChest",
    "Neck",
    "Head",
    "LeftShoulder",
    "RightShoulder",
    "LeftUpperArm",
    "RightUpperArm",
    "LeftLowerArm",
    "RightLowerArm",
    "LeftHand",
    "RightHand",
    "LeftUpperLeg",
    "RightUpperLeg",
    "LeftLowerLeg",
    "RightLowerLeg",
];

/// Plausible human shoulder span in metres — mirrors
/// `vulvatar_lib::tracking::SHOULDER_SPAN_MIN_M` / `_MAX_M`. Duplicated as
/// plain numbers so this bin stays a pure text tool with no feature-gated
/// dependency on the tracking stack.
const SPAN_MIN_M: f64 = 0.20;
const SPAN_MAX_M: f64 = 0.60;

struct Frame {
    index: u64,
    t_ms: Option<f64>,
    conf: f64,
    ref_span_m: Option<f64>,
    mpsu: Option<f64>,
    joints: BTreeMap<String, ([f64; 3], f64, String)>,
    jumps: Vec<(String, f64)>,
    dump: Option<String>,
}

fn f3(v: &serde_json::Value) -> Option<[f64; 3]> {
    let a = v.as_array()?;
    Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?, a.get(2)?.as_f64()?])
}

fn parse(line: &str) -> Option<Frame> {
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    let mut joints = BTreeMap::new();
    if let Some(map) = v.get("j").and_then(|j| j.as_object()) {
        for (name, jv) in map {
            let Some(p) = jv.get("p").and_then(f3) else {
                continue;
            };
            let c = jv.get("c").and_then(|c| c.as_f64()).unwrap_or(0.0);
            let o = jv
                .get("o")
                .and_then(|o| o.as_str())
                .unwrap_or("?")
                .to_string();
            joints.insert(name.clone(), (p, c, o));
        }
    }
    let jumps = v
        .get("jump")
        .and_then(|j| j.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    let a = e.as_array()?;
                    Some((a.first()?.as_str()?.to_string(), a.get(1)?.as_f64()?))
                })
                .collect()
        })
        .unwrap_or_default();
    Some(Frame {
        index: v.get("f").and_then(|f| f.as_u64()).unwrap_or(0),
        t_ms: v.get("t_ms").and_then(|t| t.as_f64()),
        conf: v.get("conf").and_then(|c| c.as_f64()).unwrap_or(0.0),
        ref_span_m: v.pointer("/metric/ref_span_m").and_then(|x| x.as_f64()),
        mpsu: v.pointer("/metric/mpsu").and_then(|x| x.as_f64()),
        joints,
        jumps,
        dump: v
            .get("dump")
            .and_then(|d| d.as_str())
            .map(|s| s.to_string()),
    })
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i]
}

fn newest_session() -> Option<PathBuf> {
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir("diagnostics").ok()?.flatten() {
        let p = entry.path();
        if !p.join("pose.jsonl").is_file() {
            continue;
        }
        let t = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        if best.as_ref().is_none_or(|(bt, _)| t > *bt) {
            best = Some((t, p));
        }
    }
    best.map(|(_, p)| p)
}

fn main() {
    let dir = match std::env::args().nth(1) {
        Some(a) => PathBuf::from(a),
        None => match newest_session() {
            Some(d) => {
                eprintln!("no path given — using newest session {}", d.display());
                d
            }
            None => {
                eprintln!(
                    "usage: analyze_session <session-dir>\n\
                     no diagnostics/*/pose.jsonl found. Record one first:\n\n  \
                     PowerShell: $env:VULVATAR_RECORD=\"1\"; cargo run --release\n  \
                     then reproduce the problem and stop tracking."
                );
                std::process::exit(2);
            }
        },
    };
    let path = dir.join("pose.jsonl");
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read {}: {e}", path.display());
            std::process::exit(2);
        }
    };
    let frames: Vec<Frame> = text.lines().filter_map(parse).collect();
    if frames.is_empty() {
        eprintln!("{}: no parseable frames", path.display());
        std::process::exit(2);
    }

    let report = analyze(&dir, &frames);
    print!("{report}");
    let out = dir.join("summary.md");
    if let Err(e) = std::fs::write(&out, &report) {
        eprintln!("could not write {}: {e}", out.display());
    } else {
        eprintln!("\nwrote {}", out.display());
    }
}

fn analyze(dir: &Path, frames: &[Frame]) -> String {
    let mut r = String::new();
    let _ = writeln!(r, "# Session analysis — `{}`\n", dir.display());

    // --- coverage ----------------------------------------------------
    let span_ms = match (frames.first().and_then(|f| f.t_ms), frames.last().and_then(|f| f.t_ms)) {
        (Some(a), Some(b)) if b > a => Some(b - a),
        _ => None,
    };
    let gaps = frames
        .windows(2)
        .filter(|w| w[1].index != w[0].index + 1)
        .count();
    let _ = writeln!(r, "## Coverage\n");
    let _ = writeln!(r, "- frames: **{}**", frames.len());
    if let Some(ms) = span_ms {
        let _ = writeln!(
            r,
            "- capture span: {:.1} s ({:.1} fps by device clock)",
            ms / 1000.0,
            (frames.len() - 1) as f64 / (ms / 1000.0)
        );
    }
    let _ = writeln!(
        r,
        "- index gaps (dropped lines): **{gaps}**{}",
        if gaps > 0 {
            " — the writer fell behind; the series below is thinned"
        } else {
            ""
        }
    );
    let mean_conf = frames.iter().map(|f| f.conf).sum::<f64>() / frames.len() as f64;
    let _ = writeln!(r, "- mean overall confidence: {mean_conf:.2}\n");

    // --- global scale ------------------------------------------------
    // Wrong here and nothing downstream can be right: reference_span_m
    // multiplies every anthropometric bone length AND divides the whole
    // skeleton at publish.
    let _ = writeln!(r, "## Global scale\n");
    let spans: Vec<f64> = frames.iter().filter_map(|f| f.ref_span_m).collect();
    if spans.is_empty() {
        let _ = writeln!(
            r,
            "No `metric_frame_info` on any frame — this recording is not the depth path.\n"
        );
    } else {
        let mut s = spans.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (lo, med, hi) = (s[0], pct(&s, 0.5), s[s.len() - 1]);
        let constant = (hi - lo).abs() < 1e-6;
        let _ = writeln!(
            r,
            "- `reference_span_m`: min {lo:.4} / median {med:.4} / max {hi:.4}{}",
            if constant {
                "  — **constant ⇒ a stored calibration is driving the scale**"
            } else {
                "  (varying ⇒ the provider's auto/stabilised span)"
            }
        );
        if let Some(m) = frames.iter().filter_map(|f| f.mpsu).next() {
            let _ = writeln!(r, "- `mpsu` (first frame): {m:.4} m/source-unit");
        }
        if med < SPAN_MIN_M || med > SPAN_MAX_M {
            let _ = writeln!(
                r,
                "\n> **IMPLAUSIBLE.** A human shoulder span is {SPAN_MIN_M:.2}–{SPAN_MAX_M:.2} m. \
                 At {med:.3} every anthropometric bone length is off by ×{:.2}, so every \
                 depth-hole joint is pinned that far along its ray, and the published skeleton \
                 is scaled to {:.0}% of the range the solver is tuned for.",
                med / 0.38,
                100.0 * 0.38 / med
            );
        } else if !constant {
            let jitter = hi / lo;
            let _ = writeln!(
                r,
                "- span jitter (max/min): ×{jitter:.3}{}",
                if jitter > 1.10 {
                    "  — **high**: the whole avatar is zooming frame to frame"
                } else {
                    ""
                }
            );
        }
        let _ = writeln!(r);
    }

    // --- left/right block transposition ------------------------------
    // A whole-body L/R swap reads as the torso yaw snapping ±180°, which
    // is exactly "the body moved in an incredible direction" — and it is
    // invisible in a per-joint jump table, because every joint jumps at
    // once and each one looks like a plausible mirror.
    let _ = writeln!(r, "## Left/right block transposition\n");
    let mut sign_flips = 0usize;
    let mut prev_sign: Option<bool> = None;
    let mut reversed_frames = 0usize;
    let mut have_pair = 0usize;
    for f in frames {
        let (Some(l), Some(rj)) = (f.joints.get("LeftUpperArm"), f.joints.get("RightUpperArm"))
        else {
            continue;
        };
        have_pair += 1;
        // Source +x is the subject's anatomical left; a front-facing hold
        // puts LeftUpperArm at the larger x.
        let normal = l.0[0] > rj.0[0];
        if !normal {
            reversed_frames += 1;
        }
        if prev_sign.is_some_and(|p| p != normal) {
            sign_flips += 1;
        }
        prev_sign = Some(normal);
    }
    if have_pair == 0 {
        let _ = writeln!(r, "No frame carried both upper arms.\n");
    } else {
        let _ = writeln!(
            r,
            "- frames with both upper arms: {have_pair}\n\
             - frames with shoulders x-reversed: {reversed_frames} ({:.1}%)\n\
             - order flips between consecutive frames: **{sign_flips}**{}\n",
            100.0 * reversed_frames as f64 / have_pair as f64,
            if sign_flips > have_pair / 20 {
                "  — **the L/R correction is oscillating**; the whole skeleton mirrors back \
                 and forth at frame rate"
            } else {
                ""
            }
        );
    }

    // --- per-joint provenance + motion -------------------------------
    let _ = writeln!(r, "## Per-joint provenance and inter-frame motion\n");
    let _ = writeln!(
        r,
        "`O` = observed depth sample · `E` = extrapolated (depth hole → pinned one \
         bone-length along the 2D ray) · `S` = synthesized\n"
    );
    let _ = writeln!(
        r,
        "| joint | frames | O% | E% | S% | move p50 | p95 | max |"
    );
    let _ = writeln!(r, "|---|---:|---:|---:|---:|---:|---:|---:|");

    let mut worst_e_share = 0.0f64;
    let mut worst_e_joint = String::new();
    for name in TRACKED {
        let mut deltas: Vec<f64> = Vec::new();
        let (mut n, mut o, mut e, mut s) = (0usize, 0usize, 0usize, 0usize);
        let mut prev: Option<[f64; 3]> = None;
        for f in frames {
            match f.joints.get(*name) {
                Some((p, _, tag)) => {
                    n += 1;
                    match tag.as_str() {
                        "O" => o += 1,
                        "E" => e += 1,
                        "S" => s += 1,
                        _ => {}
                    }
                    if let Some(q) = prev {
                        deltas.push(
                            ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2))
                                .sqrt(),
                        );
                    }
                    prev = Some(*p);
                }
                // A joint that drops out and returns must not contribute a
                // fake "jump" across the gap.
                None => prev = None,
            }
        }
        if n == 0 {
            continue;
        }
        deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let e_share = 100.0 * e as f64 / n as f64;
        if e_share > worst_e_share {
            worst_e_share = e_share;
            worst_e_joint = (*name).to_string();
        }
        let _ = writeln!(
            r,
            "| {name} | {n} | {:.0} | {:.0} | {:.0} | {:.3} | {:.3} | {:.3} |",
            100.0 * o as f64 / n as f64,
            e_share,
            100.0 * s as f64 / n as f64,
            pct(&deltas, 0.5),
            pct(&deltas, 0.95),
            deltas.last().copied().unwrap_or(f64::NAN),
        );
    }
    let _ = writeln!(r);

    // --- jump events ---------------------------------------------------
    let _ = writeln!(r, "## Flagged jumps\n");
    let total_jumps: usize = frames.iter().map(|f| f.jumps.len()).sum();
    let jump_frames = frames.iter().filter(|f| !f.jumps.is_empty()).count();
    if total_jumps == 0 {
        let _ = writeln!(
            r,
            "None — no tracked joint exceeded the trigger. If the avatar still \
             misbehaved during this recording, the fault is **downstream of the source \
             skeleton** (solver / IK / retarget), not in tracking. Lower the trigger with \
             `VULVATAR_RECORD_JUMP` to see smaller events.\n"
        );
    } else {
        let _ = writeln!(
            r,
            "**{total_jumps}** jumps across **{jump_frames}** frames ({:.1}% of the session).\n",
            100.0 * jump_frames as f64 / frames.len() as f64
        );
        // Classify each jump by the provenance of BOTH its endpoints, not
        // just the frame it landed on. A transient excursion is flagged
        // twice — once going out, once coming back — so "what was the tag
        // on the landing frame" splits ~50/50 for every teleport and
        // discriminates nothing. The question that does discriminate is
        // whether either end of the move was a position the sensor never
        // measured.
        let mut prev_tag: BTreeMap<String, String> = BTreeMap::new();
        let mut by_joint: BTreeMap<String, (usize, f64, usize, usize)> = BTreeMap::new();
        for f in frames {
            for (name, d) in &f.jumps {
                let here = f.joints.get(name).map(|j| j.2.as_str()).unwrap_or("?");
                let there = prev_tag.get(name).map(|s| s.as_str()).unwrap_or("?");
                let e = by_joint.entry(name.clone()).or_insert((0, 0.0, 0, 0));
                e.0 += 1;
                e.1 = e.1.max(*d);
                if here == "E" || there == "E" {
                    e.2 += 1;
                } else if here == "O" && there == "O" {
                    e.3 += 1;
                }
            }
            // Refreshed after the jump scan so `there` is genuinely the
            // previous frame's tag for this joint.
            for (name, (_, _, tag)) in &f.joints {
                prev_tag.insert(name.clone(), tag.clone());
            }
        }
        let _ = writeln!(
            r,
            "| joint | jumps | largest | extrapolated end | measured both ends |"
        );
        let _ = writeln!(r, "|---|---:|---:|---:|---:|");
        let mut rows: Vec<_> = by_joint.iter().collect();
        rows.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        for (name, (n, max_d, on_e, on_o)) in rows {
            let _ = writeln!(r, "| {name} | {n} | {max_d:.3} | {on_e} | {on_o} |");
        }
        let _ = writeln!(r);

        let dumps: Vec<&String> = frames.iter().filter_map(|f| f.dump.as_ref()).collect();
        if dumps.is_empty() {
            let _ = writeln!(
                r,
                "_No colour/depth pairs were captured for these frames._\n"
            );
        } else {
            let _ = writeln!(
                r,
                "Replayable frames captured ({}). Re-run one through the real provider with:\n\n\
                 ```\ncargo run --features realsense --bin diagnose_depth_replay -- \\\n  \
                 {}/{}_color.bmp\n```\n\n\
                 For the whole session (continuous temporal state, the live pipeline minus \
                 the camera):\n\n```\ncargo run --release --bin diagnose_video_replay -- {}\n```\n",
                dumps.len(),
                dir.display(),
                dumps[0],
                dir.display()
            );
        }

        // --- verdict ---------------------------------------------------
        let on_e: usize = by_joint.values().map(|v| v.2).sum();
        let on_o: usize = by_joint.values().map(|v| v.3).sum();
        let _ = writeln!(r, "## Verdict\n");
        if on_e * 2 > total_jumps {
            let _ = writeln!(
                r,
                "**Most jumps have an extrapolated endpoint** ({on_e}/{total_jumps}). Those \
                 positions were never measured — the depth sample was a hole and the joint was \
                 pinned one anthropometric bone-length along its 2D ray. That makes the \
                 *bone lengths* the suspect, and they come from `reference_span_m` in the \
                 Global scale section above. Check it is a real metric span before looking \
                 at the detector."
            );
        } else if on_o * 2 > total_jumps {
            let _ = writeln!(
                r,
                "**Most jumps are measured at both ends** ({on_o}/{total_jumps}) — the sensor \
                 really returned both depths, so the bone-length fallback is not involved. \
                 Look at depth sampling / person masking (an occluder pixel read as the \
                 joint), or at a genuine detector keypoint flip."
            );
        } else {
            let _ = writeln!(
                r,
                "Jumps split between measured-both-ends and extrapolated-endpoint — no single \
                 stage dominates. Replay the captured frames above."
            );
        }
        if sign_flips > have_pair / 20 {
            let _ = writeln!(
                r,
                "\n**Also: the left/right block is oscillating** ({sign_flips} order flips). \
                 That alone produces whole-body thrashing and will mask everything else — \
                 fix it first."
            );
        }
        if let Some(med) = {
            let mut s = spans.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (!s.is_empty()).then(|| pct(&s, 0.5))
        } {
            if med < SPAN_MIN_M || med > SPAN_MAX_M {
                let _ = writeln!(
                    r,
                    "\n**Also: the global scale is implausible** ({med:.3} m). Every \
                     conclusion above is drawn from a mis-scaled skeleton."
                );
            }
        }
        if worst_e_share > 30.0 {
            let _ = writeln!(
                r,
                "\nHighest extrapolation rate: `{worst_e_joint}` at {worst_e_share:.0}% of its \
                 frames — that joint is mostly *inferred*, not measured, however smooth it looks."
            );
        }
        let _ = writeln!(r);
    }

    r
}
