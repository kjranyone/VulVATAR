//! Articulated body model for the fusion estimator.
//!
//! * **Body frame** = VRM humanoid convention: +Y up, +Z forward (the
//!   subject faces +Z), +X = the subject's LEFT. Rest pose is the VRM
//!   T-pose (arms straight out along ±X, palms down, fingers extended).
//! * The root joint is the pelvis; `State::root_r/root_t` map body → camera
//!   (x-right / y-down / z-forward). A subject squarely facing the camera
//!   has `root_r ≈ Rx(180°)`.
//! * Joints are either 3-DoF balls (rotation stored as a matrix, perturbed
//!   on the left in the parent frame) or 1-DoF hinges about a fixed axis in
//!   the parent frame. Bone lengths and capsule radii are the shape vector
//!   `β` (log-multipliers over an anthropometric template).
//! * Every model point (joint origin, site, capsule end) exposes an
//!   analytic Jacobian w.r.t. the full parameter vector through
//!   [`Model::point_jacobian`], which is what every residual block uses.

use crate::asset::HumanoidBone;

use super::math::*;

/// Kinematic joint type.
#[derive(Clone, Copy, Debug)]
pub enum JointKind {
    /// 3-DoF rotation. Limits are on the rotation-vector components
    /// (soft, per axis) in the parent frame.
    Ball { lo: V3, hi: V3 },
    /// 1-DoF rotation about `axis` (unit, parent frame). Positive angle =
    /// anatomical flexion (or the documented convention per joint).
    Hinge { axis: V3, lo: f64, hi: f64 },
}

/// Bone-length shape groups (index into `β_len`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum LenGroup {
    Shoulder = 0,
    UpperArm = 1,
    Forearm = 2,
    Hand = 3,
    Spine = 4,
    Head = 5,
    Hips = 6,
    Thigh = 7,
    Shin = 8,
    Foot = 9,
}
pub const NUM_LEN_GROUPS: usize = 10;

/// Capsule radius groups (index into `β_rad`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum RadGroup {
    Torso = 0,
    Head = 1,
    UpperArm = 2,
    Forearm = 3,
    Hand = 4,
    Thigh = 5,
    Shin = 6,
    Neck = 7,
}
pub const NUM_RAD_GROUPS: usize = 8;

#[derive(Clone, Debug)]
pub struct JointDef {
    pub name: &'static str,
    pub parent: Option<usize>,
    /// Unit direction of the offset from the parent joint, in the parent
    /// joint's frame (rest pose).
    pub offset_dir: V3,
    /// Template length of that offset (metres).
    pub base_len: f64,
    pub len_group: LenGroup,
    pub kind: JointKind,
    pub bone: Option<HumanoidBone>,
    /// Prior mean of the joint rotation (rotation vector / hinge angle) in
    /// the parent frame — the "relaxed" pose the estimator falls back to
    /// when a joint is unobserved.
    pub prior_mean: V3,
    /// Prior standard deviation (rad) per component (hinge uses [0]).
    pub prior_sigma: V3,
}

/// A point rigidly attached to a joint frame (keypoint anchor / capsule end).
#[derive(Clone, Copy, Debug)]
pub struct SiteDef {
    pub name: &'static str,
    pub joint: usize,
    /// Offset direction in the joint frame (need not be unit).
    pub offset: V3,
    /// Multiplied by `exp(β_scale + β_len[group])`.
    pub len_group: LenGroup,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointRef {
    Joint(usize),
    Site(usize),
}

#[derive(Clone, Copy, Debug)]
pub struct CapsuleDef {
    pub a: PointRef,
    pub b: PointRef,
    /// Lateral radius (the only radius for a round capsule).
    pub base_radius: f64,
    pub rad_group: RadGroup,
    /// Elliptic cross-section: `lateral` is a reference point whose
    /// direction from the axis defines the wide semi-axis (`base_radius`);
    /// the perpendicular (depth) semi-axis is `base_radius × aspect`.
    /// `None` / 1.0 for a round capsule. The trunk uses this: a chest is a
    /// flat slab (width 0.34 m, depth 0.21 m), and a flat surface patch
    /// pins its yaw where any arrangement of round tubes cannot
    /// (measured: two round trunk capsules fitted the chest plane equally
    /// well 30° off).
    pub lateral: Option<PointRef>,
    pub aspect: f64,
    /// Which limb this capsule belongs to (for association bookkeeping and
    /// visibility duty accounting).
    pub part: Part,
}

/// Coarse body part label used for association and per-part statistics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Part {
    Torso,
    Head,
    LeftArm,
    RightArm,
    LeftHand,
    RightHand,
    LeftLeg,
    RightLeg,
}

/// The kinematic + shape template. Immutable after construction.
pub struct Model {
    pub joints: Vec<JointDef>,
    pub sites: Vec<SiteDef>,
    pub capsules: Vec<CapsuleDef>,
    /// Parameter offset of each joint in the flat vector.
    pub joint_param: Vec<usize>,
    /// Total parameter count.
    pub num_params: usize,
    /// Offsets of the shape parameters.
    pub beta_scale: usize,
    pub beta_len: usize,
    pub beta_rad: usize,
    /// Depth of each joint (root children = 1).
    pub depth: Vec<usize>,
}

pub const ROOT_ROT: usize = 0;
pub const ROOT_T: usize = 3;
pub const NUM_ROOT_PARAMS: usize = 6;

/// Named joint indices, filled in by [`Model::humanoid`] construction and
/// looked up by name at build time so the table stays the single source.
#[derive(Clone, Copy, Debug, Default)]
pub struct JointIdx {
    pub pelvis: usize,
    pub spine1: usize,
    pub spine2: usize,
    pub spine3: usize,
    pub neck: usize,
    pub head: usize,
    pub l_clav: usize,
    pub r_clav: usize,
    pub l_shoulder: usize,
    pub r_shoulder: usize,
    pub l_elbow: usize,
    pub r_elbow: usize,
    pub l_elbow_twist: usize,
    pub r_elbow_twist: usize,
    pub l_wrist: usize,
    pub r_wrist: usize,
    pub l_hip: usize,
    pub r_hip: usize,
    pub l_knee: usize,
    pub r_knee: usize,
    pub l_ankle: usize,
    pub r_ankle: usize,
    /// `[hand][finger][segment]` — segment 0 = MCP flex (thumb: CMC
    /// flex), 1 = MCP abduction (thumb: CMC abduction), 2 = PIP (thumb
    /// MCP), 3 = DIP (thumb IP). Hand 0 = left.
    pub finger: [[[usize; 4]; 5]; 2],
}

/// Named site indices.
#[derive(Clone, Copy, Debug, Default)]
pub struct SiteIdx {
    pub head_top: usize,
    pub nose: usize,
    pub l_eye: usize,
    pub r_eye: usize,
    pub l_ear: usize,
    pub r_ear: usize,
    /// `[hand][finger]` fingertip.
    pub tip: [[usize; 5]; 2],
    /// `[leg]` toe tip.
    pub toe: [usize; 2],
    pub torso_l_lo: usize,
    /// Trunk axis ends (pelvis / upper chest centre line).
    pub torso_lo: usize,
    pub torso_hi: usize,
    pub torso_r_lo: usize,
    pub torso_l_hi: usize,
    pub torso_r_hi: usize,
    pub head_center: usize,
}

pub struct Humanoid {
    pub model: Model,
    pub j: JointIdx,
    pub s: SiteIdx,
}

const BALL_WIDE: JointKind = JointKind::Ball {
    lo: [-2.8, -2.8, -2.8],
    hi: [2.8, 2.8, 2.8],
};

fn ball(lo: V3, hi: V3) -> JointKind {
    JointKind::Ball { lo, hi }
}
fn hinge(axis: V3, lo: f64, hi: f64) -> JointKind {
    JointKind::Hinge {
        axis: normalize(axis),
        lo,
        hi,
    }
}

/// Mirror a definition across the YZ plane (left → right). Offsets flip x;
/// hinge axes become `(ax, −ay, −az)` so a positive angle stays flexion;
/// ball limits and priors flip the y/z components (a rotation vector
/// mirrored across YZ is `(−wx, wy, wz)`... expressed so the *anatomical*
/// motion mirrors: swap sign of the y and z components).
fn mirror_kind(k: JointKind) -> JointKind {
    match k {
        JointKind::Ball { lo, hi } => JointKind::Ball {
            lo: [lo[0], -hi[1], -hi[2]],
            hi: [hi[0], -lo[1], -lo[2]],
        },
        JointKind::Hinge { axis, lo, hi } => JointKind::Hinge {
            axis: [axis[0], -axis[1], -axis[2]],
            lo,
            hi,
        },
    }
}
fn mirror_v(v: V3) -> V3 {
    [-v[0], v[1], v[2]]
}
fn mirror_rotvec(w: V3) -> V3 {
    [w[0], -w[1], -w[2]]
}

struct Builder {
    joints: Vec<JointDef>,
    sites: Vec<SiteDef>,
    capsules: Vec<CapsuleDef>,
}

impl Builder {
    fn joint(
        &mut self,
        name: &'static str,
        parent: Option<usize>,
        offset_dir: V3,
        base_len: f64,
        len_group: LenGroup,
        kind: JointKind,
        bone: Option<HumanoidBone>,
        prior_mean: V3,
        prior_sigma: V3,
    ) -> usize {
        self.joints.push(JointDef {
            name,
            parent,
            offset_dir: if norm(offset_dir) > 0.0 {
                normalize(offset_dir)
            } else {
                offset_dir
            },
            base_len,
            len_group,
            kind,
            bone,
            prior_mean,
            prior_sigma,
        });
        self.joints.len() - 1
    }
    fn site(&mut self, name: &'static str, joint: usize, offset: V3, len_group: LenGroup) -> usize {
        self.sites.push(SiteDef {
            name,
            joint,
            offset,
            len_group,
        });
        self.sites.len() - 1
    }
    fn capsule(&mut self, a: PointRef, b: PointRef, r: f64, g: RadGroup, part: Part) {
        self.capsules.push(CapsuleDef {
            a,
            b,
            base_radius: r,
            rad_group: g,
            part,
            lateral: None,
            aspect: 1.0,
        });
    }
    #[allow(clippy::too_many_arguments)]
    fn capsule_ellipse(
        &mut self,
        a: PointRef,
        b: PointRef,
        lateral: PointRef,
        r_lat: f64,
        aspect: f64,
        g: RadGroup,
        part: Part,
    ) {
        self.capsules.push(CapsuleDef {
            a,
            b,
            base_radius: r_lat,
            rad_group: g,
            part,
            lateral: Some(lateral),
            aspect,
        });
    }
}

impl Humanoid {
    /// Build the full-body template (average adult, ~1.70 m).
    pub fn new() -> Self {
        use HumanoidBone as B;
        use LenGroup as L;
        use RadGroup as R;
        let mut b = Builder {
            joints: Vec::new(),
            sites: Vec::new(),
            capsules: Vec::new(),
        };
        let mut j = JointIdx::default();
        let mut s = SiteIdx::default();
        let z3 = [0.0, 0.0, 0.0];
        let sig = |v: f64| [v, v, v];

        // ---- torso chain -------------------------------------------------
        j.pelvis = b.joint(
            "pelvis",
            None,
            z3,
            0.0,
            L::Spine,
            BALL_WIDE, // root rotation lives in root params; this ball is locked by the estimator
            Some(B::Hips),
            z3,
            sig(1.0),
        );
        j.spine1 = b.joint(
            "spine1",
            Some(j.pelvis),
            [0.0, 1.0, 0.0],
            0.10,
            L::Spine,
            ball([-0.5, -0.5, -0.4], [0.7, 0.5, 0.4]),
            Some(B::Spine),
            z3,
            sig(0.25),
        );
        j.spine2 = b.joint(
            "spine2",
            Some(j.spine1),
            [0.0, 1.0, 0.0],
            0.13,
            L::Spine,
            ball([-0.5, -0.5, -0.4], [0.6, 0.5, 0.4]),
            Some(B::Chest),
            z3,
            sig(0.25),
        );
        j.spine3 = b.joint(
            "spine3",
            Some(j.spine2),
            [0.0, 1.0, 0.0],
            0.13,
            L::Spine,
            ball([-0.4, -0.5, -0.3], [0.5, 0.5, 0.3]),
            Some(B::UpperChest),
            z3,
            sig(0.25),
        );
        j.neck = b.joint(
            "neck",
            Some(j.spine3),
            [0.0, 1.0, 0.0],
            0.12,
            L::Spine,
            ball([-0.6, -0.9, -0.5], [0.8, 0.9, 0.5]),
            Some(B::Neck),
            z3,
            sig(0.35),
        );
        j.head = b.joint(
            "head",
            Some(j.neck),
            [0.0, 1.0, 0.0],
            0.10,
            L::Head,
            ball([-0.7, -1.2, -0.6], [0.7, 1.2, 0.6]),
            Some(B::Head),
            z3,
            sig(0.5),
        );
        s.head_top = b.site("head_top", j.head, [0.0, 0.20, 0.0], L::Head);
        s.head_center = b.site("head_center", j.head, [0.0, 0.10, 0.0], L::Head);
        s.nose = b.site("nose", j.head, [0.0, 0.055, 0.105], L::Head);
        s.l_eye = b.site("l_eye", j.head, [0.032, 0.09, 0.085], L::Head);
        s.r_eye = b.site("r_eye", j.head, [-0.032, 0.09, 0.085], L::Head);
        s.l_ear = b.site("l_ear", j.head, [0.075, 0.07, 0.0], L::Head);
        s.r_ear = b.site("r_ear", j.head, [-0.075, 0.07, 0.0], L::Head);
        // torso capsule ends (two vertical capsules give the torso a
        // 2:1 elliptical section instead of a round tube)
        s.torso_l_lo = b.site("torso_l_lo", j.pelvis, [0.06, 0.02, 0.0], L::Hips);
        s.torso_r_lo = b.site("torso_r_lo", j.pelvis, [-0.06, 0.02, 0.0], L::Hips);
        s.torso_l_hi = b.site("torso_l_hi", j.spine3, [0.07, 0.10, 0.0], L::Shoulder);
        s.torso_r_hi = b.site("torso_r_hi", j.spine3, [-0.07, 0.10, 0.0], L::Shoulder);
        s.torso_lo = b.site("torso_lo", j.pelvis, [0.0, 0.02, 0.0], L::Hips);
        s.torso_hi = b.site("torso_hi", j.spine3, [0.0, 0.10, 0.0], L::Shoulder);
        // One elliptic trunk (lateral semi-axis 0.17 m, depth 0.105 m):
        // the same 2:1 section the former capsule pair approximated, as a
        // single surface whose flat front carries yaw.
        b.capsule_ellipse(
            PointRef::Site(s.torso_lo),
            PointRef::Site(s.torso_hi),
            PointRef::Site(s.torso_l_hi),
            0.17,
            0.105 / 0.17,
            R::Torso,
            Part::Torso,
        );
        b.capsule(
            PointRef::Joint(j.spine3),
            PointRef::Joint(j.neck),
            0.06,
            R::Neck,
            Part::Torso,
        );
        b.capsule(
            PointRef::Joint(j.head),
            PointRef::Site(s.head_top),
            0.085,
            R::Head,
            Part::Head,
        );

        // ---- arms (left, then mirrored) ------------------------------------
        // Clavicle: from upper chest, mostly lateral+up. Shoulder joint sits
        // at the acromion. Elbow = flexion hinge + forearm twist hinge.
        let l_arm = |b: &mut Builder, s: &mut SiteIdx, mirror: bool| -> ([usize; 5], [[usize; 4]; 5]) {
            let m = |v: V3| if mirror { mirror_v(v) } else { v };
            let mk = |k: JointKind| if mirror { mirror_kind(k) } else { k };
            let mr = |w: V3| if mirror { mirror_rotvec(w) } else { w };
            let hand = if mirror { 1 } else { 0 };
            let (b_clav, b_up, b_lo, b_hand) = if mirror {
                (B::RightShoulder, B::RightUpperArm, B::RightLowerArm, B::RightHand)
            } else {
                (B::LeftShoulder, B::LeftUpperArm, B::LeftLowerArm, B::LeftHand)
            };
            let clav = b.joint(
                if mirror { "r_clav" } else { "l_clav" },
                Some(3), // spine3
                m([0.25, 1.0, 0.0]),
                0.10,
                L::Shoulder,
                mk(ball([-0.4, -0.5, -0.4], [0.4, 0.5, 0.6])),
                Some(b_clav),
                z3,
                sig(0.15),
            );
            let sh = b.joint(
                if mirror { "r_shoulder" } else { "l_shoulder" },
                Some(clav),
                m([1.0, 0.05, 0.0]),
                0.16,
                L::Shoulder,
                mk(BALL_WIDE),
                Some(b_up),
                // relaxed: arm hanging down ≈ rotate −80° about +Z for the
                // left arm (T-pose +X → down)
                mr([0.0, 0.0, -1.4]),
                sig(0.9),
            );
            let el = b.joint(
                if mirror { "r_elbow" } else { "l_elbow" },
                Some(sh),
                m([1.0, 0.0, 0.0]),
                0.28,
                L::UpperArm,
                // flexion brings the forearm forward (+Z) from +X: rotate
                // about −Y for the left arm.
                mk(hinge([0.0, -1.0, 0.0], -0.05, 2.6)),
                None,
                [0.3, 0.0, 0.0],
                sig(0.7),
            );
            let tw = b.joint(
                if mirror { "r_elbow_twist" } else { "l_elbow_twist" },
                Some(el),
                z3,
                0.0,
                L::Forearm,
                // pronation/supination about the forearm axis (+X left)
                mk(hinge([1.0, 0.0, 0.0], -1.6, 1.6)),
                Some(b_lo),
                z3,
                sig(0.6),
            );
            let wr = b.joint(
                if mirror { "r_wrist" } else { "l_wrist" },
                Some(tw),
                m([1.0, 0.0, 0.0]),
                0.26,
                L::Forearm,
                mk(ball([-0.5, -0.6, -1.2], [0.5, 0.6, 1.2])),
                Some(b_hand),
                z3,
                sig(0.35),
            );
            b.capsule(
                PointRef::Joint(sh),
                PointRef::Joint(el),
                0.045,
                R::UpperArm,
                if mirror { Part::RightArm } else { Part::LeftArm },
            );
            b.capsule(
                PointRef::Joint(el),
                PointRef::Joint(wr),
                0.040,
                R::Forearm,
                if mirror { Part::RightArm } else { Part::LeftArm },
            );
            // Hand: MCPs as joints (children of wrist), fingers as chains.
            // Palm-down T-pose: fingers along +X (left), thumb toward +Z.
            let mut mcp_pos = [
                ("thumb", [0.030, -0.010, 0.030], 0.045),
                ("index", [0.090, 0.0, 0.028], 0.040),
                ("middle", [0.095, 0.0, 0.008], 0.045),
                ("ring", [0.090, 0.0, -0.012], 0.040),
                ("little", [0.082, 0.0, -0.030], 0.033),
            ];
            let seg_len = [
                [0.045, 0.030, 0.025], // thumb: MCP→IP, IP→tip after CMC→MCP (prox len above)
                [0.040, 0.025, 0.020],
                [0.045, 0.028, 0.022],
                [0.040, 0.026, 0.020],
                [0.033, 0.020, 0.018],
            ];
            let bones: [[B; 3]; 5] = if mirror {
                [
                    [B::RightThumbProximal, B::RightThumbIntermediate, B::RightThumbDistal],
                    [B::RightIndexProximal, B::RightIndexIntermediate, B::RightIndexDistal],
                    [B::RightMiddleProximal, B::RightMiddleIntermediate, B::RightMiddleDistal],
                    [B::RightRingProximal, B::RightRingIntermediate, B::RightRingDistal],
                    [B::RightLittleProximal, B::RightLittleIntermediate, B::RightLittleDistal],
                ]
            } else {
                [
                    [B::LeftThumbProximal, B::LeftThumbIntermediate, B::LeftThumbDistal],
                    [B::LeftIndexProximal, B::LeftIndexIntermediate, B::LeftIndexDistal],
                    [B::LeftMiddleProximal, B::LeftMiddleIntermediate, B::LeftMiddleDistal],
                    [B::LeftRingProximal, B::LeftRingIntermediate, B::LeftRingDistal],
                    [B::LeftLittleProximal, B::LeftLittleIntermediate, B::LeftLittleDistal],
                ]
            };
            let hand_part = if mirror { Part::RightHand } else { Part::LeftHand };
            let mut finger_idx = [[0usize; 4]; 5];
            for (f, (_name, pos, prox_len)) in mcp_pos.iter_mut().enumerate() {
                let is_thumb = f == 0;
                // Finger direction in the hand frame at rest.
                let dir: V3 = if is_thumb {
                    normalize([0.7, -0.15, 0.7])
                } else {
                    [1.0, 0.0, 0.0]
                };
                // Flexion axis: curl toward the palm (−Y). For a finger
                // along +X that is rotation about −Z. Thumb curls across
                // the palm: about an axis ⟂ to its dir and to the palm
                // normal.
                let flex_axis: V3 = if is_thumb {
                    normalize(cross(dir, [0.0, -1.0, 0.0]))
                } else {
                    [0.0, 0.0, -1.0]
                };
                let abd_axis: V3 = if is_thumb {
                    [0.0, -1.0, 0.0]
                } else {
                    [0.0, 1.0, 0.0]
                };
                // Metacarpal head as an offset joint (MCP flex hinge), with
                // the abduction hinge stacked at zero offset.
                let mcp_len = norm(*pos);
                let mcp_flex = b.joint(
                    "mcp_flex",
                    Some(wr),
                    m(*pos),
                    mcp_len,
                    L::Hand,
                    mk(hinge(flex_axis, if is_thumb { -0.6 } else { -0.3 }, 1.6)),
                    Some(bones[f][0]),
                    [0.1, 0.0, 0.0],
                    sig(0.5),
                );
                let mcp_abd = b.joint(
                    "mcp_abd",
                    Some(mcp_flex),
                    z3,
                    0.0,
                    L::Hand,
                    mk(hinge(abd_axis, -0.5, 0.5)),
                    None,
                    z3,
                    sig(0.25),
                );
                let pip = b.joint(
                    "pip",
                    Some(mcp_abd),
                    m(dir),
                    *prox_len,
                    L::Hand,
                    mk(hinge(flex_axis, -0.1, 1.9)),
                    Some(bones[f][1]),
                    [0.15, 0.0, 0.0],
                    sig(0.6),
                );
                let dip = b.joint(
                    "dip",
                    Some(pip),
                    m(dir),
                    seg_len[f][1],
                    L::Hand,
                    mk(hinge(flex_axis, -0.1, 1.6)),
                    Some(bones[f][2]),
                    [0.1, 0.0, 0.0],
                    sig(0.6),
                );
                let tip = b.site(
                    "tip",
                    dip,
                    scale(m(dir), seg_len[f][2]),
                    L::Hand,
                );
                s.tip[hand][f] = tip;
                finger_idx[f] = [mcp_flex, mcp_abd, pip, dip];
                b.capsule(
                    PointRef::Joint(mcp_flex),
                    PointRef::Site(tip),
                    0.009,
                    R::Hand,
                    hand_part,
                );
                if f == 2 {
                    // palm capsule wrist → middle MCP
                    b.capsule(
                        PointRef::Joint(wr),
                        PointRef::Joint(mcp_flex),
                        0.028,
                        R::Hand,
                        hand_part,
                    );
                }
            }
            ([clav, sh, el, tw, wr], finger_idx)
        };
        let (l, lf) = l_arm(&mut b, &mut s, false);
        j.finger[0] = lf;
        j.l_clav = l[0];
        j.l_shoulder = l[1];
        j.l_elbow = l[2];
        j.l_elbow_twist = l[3];
        j.l_wrist = l[4];
        let (r, rf) = l_arm(&mut b, &mut s, true);
        j.finger[1] = rf;
        j.r_clav = r[0];
        j.r_shoulder = r[1];
        j.r_elbow = r[2];
        j.r_elbow_twist = r[3];
        j.r_wrist = r[4];

        // ---- legs ---------------------------------------------------------
        let leg = |b: &mut Builder, s: &mut SiteIdx, mirror: bool| -> [usize; 3] {
            let m = |v: V3| if mirror { mirror_v(v) } else { v };
            let mk = |k: JointKind| if mirror { mirror_kind(k) } else { k };
            let (b_up, b_lo, b_ft) = if mirror {
                (B::RightUpperLeg, B::RightLowerLeg, B::RightFoot)
            } else {
                (B::LeftUpperLeg, B::LeftLowerLeg, B::LeftFoot)
            };
            let hip = b.joint(
                if mirror { "r_hip" } else { "l_hip" },
                Some(0),
                m([1.0, -0.35, 0.0]),
                0.10,
                L::Hips,
                mk(ball([-1.8, -0.8, -0.6], [0.6, 0.8, 0.9])),
                Some(b_up),
                z3,
                sig(0.4),
            );
            let knee = b.joint(
                if mirror { "r_knee" } else { "l_knee" },
                Some(hip),
                [0.0, -1.0, 0.0],
                0.42,
                L::Thigh,
                // flexion swings the shin backward (−Z) from −Y: about +X
                mk(hinge([1.0, 0.0, 0.0], -0.05, 2.5)),
                Some(b_lo),
                [0.2, 0.0, 0.0],
                sig(0.6),
            );
            let ankle = b.joint(
                if mirror { "r_ankle" } else { "l_ankle" },
                Some(knee),
                [0.0, -1.0, 0.0],
                0.41,
                L::Shin,
                mk(ball([-0.6, -0.5, -0.4], [0.6, 0.5, 0.4])),
                Some(b_ft),
                z3,
                sig(0.3),
            );
            let toe = b.site("toe", ankle, [0.0, -0.06, 0.18], L::Foot);
            s.toe[if mirror { 1 } else { 0 }] = toe;
            let part = if mirror { Part::RightLeg } else { Part::LeftLeg };
            b.capsule(PointRef::Joint(hip), PointRef::Joint(knee), 0.075, R::Thigh, part);
            b.capsule(PointRef::Joint(knee), PointRef::Joint(ankle), 0.055, R::Shin, part);
            b.capsule(PointRef::Joint(ankle), PointRef::Site(toe), 0.035, R::Shin, part);
            [hip, knee, ankle]
        };
        let ll = leg(&mut b, &mut s, false);
        j.l_hip = ll[0];
        j.l_knee = ll[1];
        j.l_ankle = ll[2];
        let rl = leg(&mut b, &mut s, true);
        j.r_hip = rl[0];
        j.r_knee = rl[1];
        j.r_ankle = rl[2];

        // ---- parameter layout ---------------------------------------------
        let mut joint_param = Vec::with_capacity(b.joints.len());
        let mut off = NUM_ROOT_PARAMS;
        for jd in &b.joints {
            joint_param.push(off);
            off += match jd.kind {
                JointKind::Ball { .. } => 3,
                JointKind::Hinge { .. } => 1,
            };
        }
        let beta_scale = off;
        let beta_len = off + 1;
        let beta_rad = beta_len + NUM_LEN_GROUPS;
        let num_params = beta_rad + NUM_RAD_GROUPS;
        let mut depth = vec![0usize; b.joints.len()];
        for (i, jd) in b.joints.iter().enumerate() {
            depth[i] = jd.parent.map(|p| depth[p] + 1).unwrap_or(0);
        }
        Self {
            model: Model {
                joints: b.joints,
                sites: b.sites,
                capsules: b.capsules,
                joint_param,
                num_params,
                beta_scale,
                beta_len,
                beta_rad,
                depth,
            },
            j,
            s,
        }
    }
}

impl Default for Humanoid {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// State + forward kinematics
// ---------------------------------------------------------------------------

/// Full estimator state: root pose, per-joint local rotations (balls as
/// matrices, hinges as angles), and shape log-multipliers.
#[derive(Clone, Debug)]
pub struct State {
    pub root_r: M3,
    pub root_t: V3,
    /// Local rotation of every joint (hinges also keep the matrix form,
    /// rebuilt from `angle`).
    pub rot: Vec<M3>,
    /// Hinge angles (unused entries for balls).
    pub angle: Vec<f64>,
    /// `β_scale`.
    pub scale: f64,
    pub len: [f64; NUM_LEN_GROUPS],
    pub rad: [f64; NUM_RAD_GROUPS],
}

impl State {
    pub fn rest(model: &Model) -> Self {
        Self {
            root_r: I3,
            root_t: [0.0, 0.0, 0.0],
            rot: vec![I3; model.joints.len()],
            angle: vec![0.0; model.joints.len()],
            scale: 0.0,
            len: [0.0; NUM_LEN_GROUPS],
            rad: [0.0; NUM_RAD_GROUPS],
        }
    }

    /// Length multiplier of a group.
    #[inline]
    pub fn len_mul(&self, g: LenGroup) -> f64 {
        (self.scale + self.len[g as usize]).exp()
    }
    #[inline]
    /// Capsule radius multiplier. Deliberately NOT tied to the global
    /// length scale: a person's girth is not proportional to their height,
    /// and with the dense surface term the radii are measured directly —
    /// coupling them to `scale` let the surface inflate every bone length
    /// by 13 % to widen the chest (measured).
    pub fn rad_mul(&self, g: RadGroup) -> f64 {
        self.rad[g as usize].exp()
    }

    /// Rotation vector of a joint's local rotation (ball) or its hinge
    /// angle in slot 0.
    pub fn joint_rotvec(&self, model: &Model, j: usize) -> V3 {
        match model.joints[j].kind {
            JointKind::Ball { .. } => so3_log(&self.rot[j]),
            JointKind::Hinge { .. } => [self.angle[j], 0.0, 0.0],
        }
    }

    /// Apply a parameter increment `δ` (length `model.num_params`).
    pub fn apply_delta(&mut self, model: &Model, delta: &[f64]) {
        let dr = [delta[ROOT_ROT], delta[ROOT_ROT + 1], delta[ROOT_ROT + 2]];
        self.root_r = orthonormalize(&mat_mul(&so3_exp(dr), &self.root_r));
        self.root_t = add(
            self.root_t,
            [delta[ROOT_T], delta[ROOT_T + 1], delta[ROOT_T + 2]],
        );
        for (j, jd) in model.joints.iter().enumerate() {
            let p = model.joint_param[j];
            match jd.kind {
                JointKind::Ball { .. } => {
                    let d = [delta[p], delta[p + 1], delta[p + 2]];
                    self.rot[j] = orthonormalize(&mat_mul(&so3_exp(d), &self.rot[j]));
                }
                JointKind::Hinge { axis, .. } => {
                    self.angle[j] += delta[p];
                    self.rot[j] = so3_exp(scale(axis, self.angle[j]));
                }
            }
        }
        self.scale += delta[model.beta_scale];
        for g in 0..NUM_LEN_GROUPS {
            self.len[g] += delta[model.beta_len + g];
        }
        for g in 0..NUM_RAD_GROUPS {
            self.rad[g] += delta[model.beta_rad + g];
        }
    }

    /// Set a hinge angle (keeps the matrix in sync).
    pub fn set_hinge(&mut self, model: &Model, j: usize, angle: f64) {
        if let JointKind::Hinge { axis, .. } = model.joints[j].kind {
            self.angle[j] = angle;
            self.rot[j] = so3_exp(scale(axis, angle));
        }
    }

    /// Set a ball joint from a rotation vector.
    pub fn set_ball(&mut self, j: usize, w: V3) {
        self.rot[j] = so3_exp(w);
    }

    /// Set every joint to its prior mean (the relaxed pose).
    pub fn set_relaxed(&mut self, model: &Model) {
        for (j, jd) in model.joints.iter().enumerate() {
            match jd.kind {
                JointKind::Ball { .. } => self.rot[j] = so3_exp(jd.prior_mean),
                JointKind::Hinge { axis, .. } => {
                    self.angle[j] = jd.prior_mean[0];
                    self.rot[j] = so3_exp(scale(axis, jd.prior_mean[0]));
                }
            }
        }
    }
}

/// Forward-kinematics result: world (camera-frame) rotation and origin of
/// every joint, plus the world offset vector of each joint from its parent
/// (needed by the shape Jacobians) and every site position.
#[derive(Clone, Debug)]
pub struct Fk {
    pub r: Vec<M3>,
    pub t: Vec<V3>,
    /// World-space offset `t[j] − t[parent]` (scaled by β).
    pub off: Vec<V3>,
    pub site: Vec<V3>,
    /// World-space site offset from its joint origin (scaled by β).
    pub site_off: Vec<V3>,
}

impl Fk {
    pub fn point(&self, p: PointRef) -> V3 {
        match p {
            PointRef::Joint(j) => self.t[j],
            PointRef::Site(s) => self.site[s],
        }
    }
}

impl Model {
    pub fn fk(&self, st: &State) -> Fk {
        let n = self.joints.len();
        let mut r = vec![I3; n];
        let mut t = vec![[0.0; 3]; n];
        let mut off = vec![[0.0; 3]; n];
        for (j, jd) in self.joints.iter().enumerate() {
            let (pr, pt) = match jd.parent {
                Some(p) => (r[p], t[p]),
                None => (st.root_r, st.root_t),
            };
            let len = jd.base_len * st.len_mul(jd.len_group);
            let o = mat_vec(&pr, scale(jd.offset_dir, len));
            off[j] = o;
            t[j] = add(pt, o);
            r[j] = mat_mul(&pr, &st.rot[j]);
        }
        let mut site = Vec::with_capacity(self.sites.len());
        let mut site_off = Vec::with_capacity(self.sites.len());
        for sd in &self.sites {
            let o = mat_vec(&r[sd.joint], scale(sd.offset, st.len_mul(sd.len_group)));
            site_off.push(o);
            site.push(add(t[sd.joint], o));
        }
        Fk {
            r,
            t,
            off,
            site,
            site_off,
        }
    }

    /// Jacobian of the world position of a model point w.r.t. every
    /// parameter it depends on, appended to `out` as `(param_index, ∂p/∂x)`.
    pub fn point_jacobian(&self, st: &State, fk: &Fk, pref: PointRef, out: &mut Vec<(usize, V3)>) {
        match pref {
            PointRef::Joint(j) => self.jac_chain(st, fk, j, fk.t[j], false, None, out),
            PointRef::Site(s) => {
                let sd = &self.sites[s];
                self.jac_chain(
                    st,
                    fk,
                    sd.joint,
                    fk.site[s],
                    true,
                    Some((fk.site_off[s], sd.len_group)),
                    out,
                )
            }
        }
    }

    /// Jacobian of a point rigidly attached to `joint` (moves with the
    /// joint's rotation, no shape dependence of its own).
    pub fn attached_point_jacobian(
        &self,
        st: &State,
        fk: &Fk,
        joint: usize,
        p_world: V3,
        out: &mut Vec<(usize, V3)>,
    ) {
        self.jac_chain(st, fk, joint, p_world, true, None, out)
    }

    /// Shared chain walk. `own_rot_moves`: whether `joint`'s own rotation
    /// moves the point (true for anything except the joint origin itself).
    /// `own_off`: world-space offset from the joint origin that scales with
    /// a length group (template sites).
    #[allow(clippy::too_many_arguments)]
    fn jac_chain(
        &self,
        st: &State,
        fk: &Fk,
        mut j: usize,
        p: V3,
        own_rot_moves: bool,
        own_off: Option<(V3, LenGroup)>,
        out: &mut Vec<(usize, V3)>,
    ) {
        out.clear();
        let mut dlen = [[0.0f64; 3]; NUM_LEN_GROUPS];
        let mut dscale = [0.0f64; 3];
        if let Some((o, g)) = own_off {
            dlen[g as usize] = add(dlen[g as usize], o);
            dscale = add(dscale, o);
        }
        let mut first = true;
        loop {
            let jd = &self.joints[j];
            let pidx = self.joint_param[j];
            let pr = match jd.parent {
                Some(pp) => fk.r[pp],
                None => st.root_r,
            };
            let rot_moves = !first || own_rot_moves;
            if rot_moves {
                let lever = sub(p, fk.t[j]);
                match jd.kind {
                    JointKind::Ball { .. } => {
                        for k in 0..3 {
                            let axis = col(&pr, k);
                            out.push((pidx + k, cross(axis, lever)));
                        }
                    }
                    JointKind::Hinge { axis, .. } => {
                        let a = mat_vec(&pr, axis);
                        out.push((pidx, cross(a, lever)));
                    }
                }
            }
            let o = fk.off[j];
            let g = jd.len_group as usize;
            dlen[g] = add(dlen[g], o);
            dscale = add(dscale, o);
            first = false;
            match jd.parent {
                Some(pp) => j = pp,
                None => break,
            }
        }
        let lever = sub(p, st.root_t);
        for k in 0..3 {
            let mut e = [0.0; 3];
            e[k] = 1.0;
            out.push((ROOT_ROT + k, cross(e, lever)));
            out.push((ROOT_T + k, e));
        }
        if norm(dscale) > 0.0 {
            out.push((self.beta_scale, dscale));
        }
        for g in 0..NUM_LEN_GROUPS {
            if norm(dlen[g]) > 0.0 {
                out.push((self.beta_len + g, dlen[g]));
            }
        }
    }

    /// Effective capsule radius under the current shape.
    #[inline]
    pub fn capsule_radius(&self, st: &State, c: &CapsuleDef) -> f64 {
        c.base_radius * st.rad_mul(c.rad_group)
    }

    /// Depth (camera-facing) semi-axis: the lateral radius for a round
    /// capsule, `radius × aspect` for an elliptic one. Occlusion tests use
    /// this — the conservative extent toward the camera.
    pub fn capsule_depth_radius(&self, st: &State, c: &CapsuleDef) -> f64 {
        self.capsule_radius(st, c) * c.aspect
    }

    /// Joint index whose `bone` equals `b`, if any.
    pub fn joint_for_bone(&self, b: HumanoidBone) -> Option<usize> {
        self.joints.iter().position(|j| j.bone == Some(b))
    }
}

/// Camera-frame rotation of a subject squarely facing the camera in the
/// body-frame convention (+Y up / +Z toward camera ⇒ Rx(180°)).
pub const FACING_CAMERA: M3 = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];

#[cfg(test)]
mod tests {
    use super::*;

    fn numeric_point_jacobian(h: &Humanoid, st: &State, pref: PointRef) -> Vec<V3> {
        let m = &h.model;
        let eps = 1e-6;
        let mut cols = Vec::with_capacity(m.num_params);
        for k in 0..m.num_params {
            let mut d = vec![0.0; m.num_params];
            d[k] = eps;
            let mut sp = st.clone();
            sp.apply_delta(m, &d);
            let mut sm = st.clone();
            d[k] = -eps;
            sm.apply_delta(m, &d);
            let pp = m.fk(&sp).point(pref);
            let pm = m.fk(&sm).point(pref);
            cols.push(scale(sub(pp, pm), 0.5 / eps));
        }
        cols
    }

    fn random_state(h: &Humanoid, seed: u64) -> State {
        let m = &h.model;
        let mut st = State::rest(m);
        let mut x = seed;
        let mut rnd = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x % 10000) as f64 / 10000.0 - 0.5) * 1.2
        };
        st.root_r = so3_exp([rnd(), rnd(), rnd()]);
        st.root_t = [rnd(), rnd(), 1.5 + rnd()];
        for j in 0..m.joints.len() {
            match m.joints[j].kind {
                JointKind::Ball { .. } => st.set_ball(j, [rnd(), rnd(), rnd()]),
                JointKind::Hinge { .. } => st.set_hinge(m, j, rnd()),
            }
        }
        st.scale = rnd() * 0.2;
        for g in 0..NUM_LEN_GROUPS {
            st.len[g] = rnd() * 0.2;
        }
        st
    }

    #[test]
    fn relaxed_pose_hangs_the_arms() {
        // The bootstrap pose is what a viewer sees before the arms are
        // ever observed (the desk envelope: hands under the desk for
        // minutes), so it has to be a natural rest, not a T-pose.
        let h = Humanoid::new();
        let mut st = State::rest(&h.model);
        st.root_r = FACING_CAMERA;
        st.set_relaxed(&h.model);
        let fk = h.model.fk(&st);
        for (sh, wr, name) in [
            (h.j.l_shoulder, h.j.l_wrist, "left"),
            (h.j.r_shoulder, h.j.r_wrist, "right"),
        ] {
            let d = sub(fk.t[wr], fk.t[sh]);
            // Camera +y is down: the wrist must hang well below the
            // shoulder, close to the trunk, and not behind the body.
            assert!(d[1] > 0.30, "{name} wrist not below shoulder: {d:?}");
            assert!(d[0].abs() < 0.25, "{name} wrist too far out: {d:?}");
            assert!(d[2].abs() < 0.25, "{name} wrist too far fore/aft: {d:?}");
        }
    }

    #[test]
    fn point_jacobian_matches_finite_difference() {
        let h = Humanoid::new();
        let m = &h.model;
        let st = random_state(&h, 0x9e3779b97f4a7c15);
        let fk = m.fk(&st);
        let mut jac = Vec::new();
        let probes = [
            PointRef::Joint(h.j.l_wrist),
            PointRef::Joint(h.j.r_ankle),
            PointRef::Site(h.s.tip[0][1]),
            PointRef::Site(h.s.tip[1][0]),
            PointRef::Site(h.s.nose),
            PointRef::Site(h.s.torso_l_hi),
            PointRef::Joint(h.j.pelvis),
        ];
        for pref in probes {
            m.point_jacobian(&st, &fk, pref, &mut jac);
            let num = numeric_point_jacobian(&h, &st, pref);
            let mut analytic = vec![[0.0; 3]; m.num_params];
            for &(i, v) in &jac {
                analytic[i] = add(analytic[i], v);
            }
            for k in 0..m.num_params {
                for c in 0..3 {
                    let a = analytic[k][c];
                    let n = num[k][c];
                    assert!(
                        (a - n).abs() < 1e-5 * (1.0 + n.abs()),
                        "{pref:?} param {k} comp {c}: analytic {a} numeric {n}"
                    );
                }
            }
        }
    }

    #[test]
    fn rest_pose_is_t_pose_facing_plus_z() {
        let h = Humanoid::new();
        let m = &h.model;
        let st = State::rest(m);
        let fk = m.fk(&st);
        // Left wrist is at +X, right at −X, both roughly shoulder height.
        assert!(fk.t[h.j.l_wrist][0] > 0.6);
        assert!(fk.t[h.j.r_wrist][0] < -0.6);
        assert!((fk.t[h.j.l_wrist][1] - fk.t[h.j.r_wrist][1]).abs() < 1e-9);
        // Nose is in front (+Z) of the ears.
        assert!(fk.site[h.s.nose][2] > fk.site[h.s.l_ear][2] + 0.05);
        // Feet below the pelvis.
        assert!(fk.t[h.j.l_ankle][1] < -0.7);
    }

    #[test]
    fn hinge_conventions_flex_the_expected_way() {
        let h = Humanoid::new();
        let m = &h.model;
        let mut st = State::rest(m);
        // Elbow flexion brings both wrists forward (+Z).
        st.set_hinge(m, h.j.l_elbow, 1.2);
        st.set_hinge(m, h.j.r_elbow, 1.2);
        let fk = m.fk(&st);
        assert!(fk.t[h.j.l_wrist][2] > 0.15, "left wrist z {}", fk.t[h.j.l_wrist][2]);
        assert!(fk.t[h.j.r_wrist][2] > 0.15, "right wrist z {}", fk.t[h.j.r_wrist][2]);
        // Knee flexion sends the ankles backward (−Z).
        let mut st = State::rest(m);
        st.set_hinge(m, h.j.l_knee, 1.0);
        st.set_hinge(m, h.j.r_knee, 1.0);
        let fk = m.fk(&st);
        assert!(fk.t[h.j.l_ankle][2] < -0.2);
        assert!(fk.t[h.j.r_ankle][2] < -0.2);
        // Finger flexion curls toward the palm (−Y) on both hands.
        let mut st = State::rest(m);
        for hand in 0..2 {
            for f in 1..5 {
                st.set_hinge(m, h.j.finger[hand][f][0], 1.2);
                st.set_hinge(m, h.j.finger[hand][f][2], 1.2);
            }
        }
        let fk = m.fk(&st);
        for hand in 0..2 {
            let wr = if hand == 0 { h.j.l_wrist } else { h.j.r_wrist };
            let tip = fk.site[h.s.tip[hand][1]];
            assert!(tip[1] < fk.t[wr][1] - 0.03, "hand {hand} tip y {} wrist y {}", tip[1], fk.t[wr][1]);
        }
        // Relaxed pose hangs the arms down.
        let mut st = State::rest(m);
        st.set_relaxed(m);
        let fk = m.fk(&st);
        assert!(fk.t[h.j.l_wrist][1] < fk.t[h.j.l_shoulder][1] - 0.3);
        assert!(fk.t[h.j.r_wrist][1] < fk.t[h.j.r_shoulder][1] - 0.3);
        assert!(fk.t[h.j.l_wrist][0].abs() < 0.45);
    }
}
