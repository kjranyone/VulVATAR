use crate::asset::{identity_matrix, Mat4, SkeletonAsset, Transform};
use crate::math_utils::mat4_mul;

pub type FrameTimestamp = u64;

#[derive(Clone, Debug)]
pub struct AvatarPose {
    pub local_transforms: Vec<Transform>,
    pub global_transforms: Vec<Mat4>,
    pub skinning_matrices: Vec<Mat4>,
    pub pose_timestamp: FrameTimestamp,
}

impl AvatarPose {
    pub fn identity(node_count: usize) -> Self {
        Self {
            local_transforms: vec![Transform::default(); node_count],
            global_transforms: vec![identity_matrix(); node_count],
            skinning_matrices: vec![identity_matrix(); node_count],
            pose_timestamp: 0,
        }
    }
}

/// Compute world-space transforms for every skeleton node from
/// per-node local transforms. Stack-based hierarchy traversal so
/// deep rigs don't blow the call stack.
///
/// Writes one `Mat4` per node into `out`. Roots take their local
/// matrix directly; every other node multiplies its local against
/// its parent's already-computed world. Nodes whose index is past
/// the end of `locals` or `out` are silently skipped.
///
/// This is the pure-function form of [`AvatarInstance::compute_global_pose`];
/// the snapshot path (`gui::calibration::target_pose`) uses it
/// without needing a `&mut AvatarInstance`.
pub fn compute_global_transforms(
    skeleton: &SkeletonAsset,
    locals: &[Transform],
    out: &mut [Mat4],
) {
    let mut stack: Vec<(usize, Option<usize>)> = Vec::new();
    for root in skeleton.root_nodes.iter().rev() {
        stack.push((root.0 as usize, None));
    }
    while let Some((idx, parent_idx)) = stack.pop() {
        if idx >= locals.len() || idx >= out.len() {
            continue;
        }
        let local_mat = locals[idx].to_matrix();
        out[idx] = match parent_idx {
            Some(pi) if pi < out.len() => mat4_mul(&out[pi], &local_mat),
            _ => local_mat,
        };
        for child in skeleton.nodes[idx].children.iter().rev() {
            stack.push((child.0 as usize, Some(idx)));
        }
    }
}

/// Compute per-node skinning matrices: `out[i] = globals[i] * inverse_bind[i]`
/// for each node that has both a global transform and an
/// inverse-bind matrix.
///
/// Used by `AvatarInstance::build_skinning_matrices` and the
/// calibration target-pose snapshot path. Both paths previously
/// duplicated this loop almost verbatim.
pub fn build_skinning_matrices(skeleton: &SkeletonAsset, globals: &[Mat4], out: &mut [Mat4]) {
    let n = out.len();
    for i in 0..n {
        let ibm = if i < skeleton.inverse_bind_matrices.len() {
            &skeleton.inverse_bind_matrices[i]
        } else {
            continue;
        };
        if i < globals.len() {
            out[i] = mat4_mul(&globals[i], ibm);
        }
    }
}
