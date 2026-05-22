# SOTA algorithm upgrades

Date: 2026-05-22

This plan tracks the state-of-the-art algorithm upgrades applied to
VulVATAR's runtime. The first slice (XPBD on the CPU cloth solver)
landed in commit `fa82bbc`; subsequent slices and remaining
candidates are catalogued below for future-me / future maintainers.

## Landed

### Cloth constraint solver — PBD → XPBD (CPU + GPU)

**Reference**: Macklin, Müller, Chentanez. "XPBD: Position-Based
Simulation of Compliant Constrained Dynamics", MIG 2016.

Replaced the classic Jakobsen-style "fraction of error to correct"
PBD with the XPBD compliance-based formulation:

```
α̃_j   = α_j / dt²
Δλ_j  = -(C_j + α̃_j · λ_j) / (w_a + w_b + α̃_j)
λ_j  += Δλ_j
Δx_a  = +w_a · d · Δλ_j     (d = (x_a − x_b) / |x_a − x_b|)
Δx_b  = −w_b · d · Δλ_j
```

**Why this is SOTA over the prior PBD**:

| Property | Classic PBD | XPBD |
|---|---|---|
| Stiffness depends on iteration count | yes (more iters → stiffer) | no |
| Stiffness depends on substep `dt` | yes | no (α̃ = α/dt² absorbs dt²) |
| Author authority over material | "stiffness" knob is solver-dependent | compliance α is a real material parameter |
| Equilibrium under load | drifts with solver settings | converges to load-dependent rest |

**Backward compatibility**: the legacy `stiffness ∈ (0, 1]` knob is
mapped onto an XPBD compliance via `compliance_from_pbd_stiffness`:
`stiffness = 1` → `α = 0` (rigid limit; XPBD is identical to PBD
with full correction). The conservative base scale (`1e-7 m/N`) keeps
existing assets visually identical to the prior PBD output.
`stiffness = 0` short-circuits to "constraint disabled" so the
legacy "soft-hinge-or-disable" assets keep working.

**Files (CPU side, commit `fa82bbc`)**:
- `src/simulation/cloth_solver/constraints.rs` — XPBD update rule
- `src/simulation/cloth_solver/mod.rs` — `reset_lambda` per substep
- `src/simulation/cloth.rs` — `ClothSimTempBuffers::lambda_distance` +
  `reset_lambda(count)` helper

**Files (GPU side, commit `3911428`)**:
- `src/renderer/pipeline.rs` —
  `cloth_constraint_lambda_update_cs` (new GLSL),
  `cloth_constraint_accumulate_cs` (rewritten to read `dlambda`),
  `cloth_constraint_apply_cs` (UBO layout update only),
  `ClothConstraintControl` gains `constraint_count` + `dt` fields
- `src/renderer/cloth_cache.rs` — `lambda_ssbo` + `dlambda_ssbo`
  allocation, `lambda_update_set` descriptor set
- `src/renderer/mod.rs` —
  `cloth_constraint_lambda_update_pipeline` field on `VulkanRenderer`,
  3-pass dispatch loop (lambda update → accumulate → apply),
  per-substep `cmd.fill_buffer(lambda_ssbo, 0)` reset

The GPU side uses the same lambda → dlambda buffer split that the
CPU side does (lambda is persistent across iterations within a
substep, dlambda is per-iteration scratch). The split makes the
per-particle Δx accumulate **race-free without atomics**: lambda is
mutated exactly once per constraint per iteration (the lambda-update
dispatch); the accumulate pass reads dlambda but never writes
lambda, so the two endpoint invocations of a constraint consume the
same `Δλ_j` and produce consistent Δx contributions.

**Tests**:
- `xpbd_rigid_stiffness_satisfies_constraint_in_one_pass`
- `xpbd_lambda_accumulates_within_substep_and_resets_across_substeps`
- `xpbd_disabled_constraint_leaves_lambda_zero`
- existing `cloth_constraint_jacobi_satisfies_rest_lengths_like_cpu_pbd`
  parity test stays correct because XPBD@α=0 reduces to PBD@stiffness=1
  algebraically — covers the CPU XPBD ↔ GPU XPBD parity at stiffness=1

**Stiffness sweep parity** (landed in `aa6532c`):
`cpu_mirror_of_cloth_constraint_iter` was rewritten to mirror the
three-pass GLSL XPBD dispatch with `lambda` + `dt` parameters, and
`cloth_constraint_jacobi_satisfies_rest_lengths_like_cpu_pbd` now
sweeps `stiffness ∈ {1.0, 0.7, 0.3}`. CPU XPBD ↔ GPU XPBD parity
is therefore exercised at non-trivial compliance values, not just
the `α = 0` rigid limit.

### Cloth normal recomputation — angle-weighted (Max 1999)

**Reference**: Max, "Weights for Computing Vertex Normals from
Facet Normals", JGT 1999.

Replaced area-weighted face-normal accumulation (raw cross product
summed per vertex) with angle-weighted (each incident triangle
contributes its *unit* face normal scaled by the incident angle at
the vertex). On regular meshes the two converge to the same result;
on irregular triangulations the angle-weighted version is bounded
by `[0, π]` per triangle rather than by triangle area, so the
normal at each vertex follows the local 1-ring shape instead of
the area of the largest incident triangle.

**Files (commit `70bcfdc`)**:
- `src/simulation/cloth_solver/output.rs::compute_normals` — CPU
  reference. Adds `safe_angle(u, v)` helper for degenerate
  protection.
- `src/renderer/pipeline.rs::cloth_normal_cs` — GLSL mirror.
- `src/simulation/cloth_gpu_boundary.rs::cpu_mirror_of_cloth_normal_cs`
  — formula-parity reference for the GLSL shader.

### Skinning — LBS → DQS (Kavan 2007)

**Reference**: Kavan, Collins, Žára, O'Sullivan. "Skinning with
Dual Quaternions", I3D 2007.

Replaced Linear Blend Skinning with Dual Quaternion Skinning in
`transform_cs`. DQS eliminates the candy-wrapper twist artefact at
joints with non-trivial rotational delta (forearm pronation, hip
yaw under skirts, neck twist).

**Algorithm** (in-shader, per vertex):

1. Extract rotation quaternion from each joint's skinning matrix
   via Mike Day's robust mat3→quat conversion.
2. Pick the first non-zero-weight joint as the antipodal reference;
   flip subsequent contributing quaternions when `dot(qi, ref) < 0`
   so the linear blend stays on one hemisphere.
3. Weighted-blend the (flipped) quaternions and renormalise.
4. Recover translation as the weighted average of each joint's
   `m[3].xyz` (kept separate from the quaternion blend because the
   source matrices are `global · inverse_bind` products with
   already-world-space translations).
5. Apply: `pos' = quat_rotate(q, pos) + t`,
   `nrm' = quat_rotate(q, nrm)`.

**Files (commit `198d876`, hardened in `2daf611`)**:
- `src/renderer/pipeline.rs::transform_cs` — GLSL DQS dispatch.
  The `SkinningData.matrices` SSBO layout is **unchanged**; the
  matrix → quaternion conversion happens entirely in-shader so no
  CPU code needed to be touched.

Cost: 4× mat3→quat extractions per vertex plus a 4-way quaternion
blend + normalise + two `quat_rotate` calls. On VRM avatars
(~10k–50k skinned vertices) the cost lands well below the morph /
cloth solver and graphics pipeline budget.

The `total_w < 0.001` early-return preserves the prior LBS path's
"no skinning → pass through" behaviour for non-bone-bound
accessories.

## Open SOTA slices

### Body twist / yaw — torso 4-point plane fit OR elbow Δz with arm-pose gate

See [`body-twist-yaw-investigation.md`](./body-twist-yaw-investigation.md).
Current implementation: shoulder pair sequential fallback. Two SOTA-ish
upgrades are documented as Option B (SVD plane fit) and Option C
(elbow Δz with arm-pose gate); per the measurement table, neither
recovers the 45° three-quarter rotation when shoulder Δz is at noise
floor — the root cause is the RTMW3D depth model's degenerate Z output
on close-together points, which only a backend swap (Option D: Sapiens
/ BlazePose GHUM 3D) cleanly fixes. Backend swap is multi-day work and
out of scope for this slice.

### Tracking — RTMW3D → Sapiens or BlazePose GHUM 3D

Multi-day backend swap. The body-twist-yaw 45° issue, the depth
noise floor on close-together points, and the shoulder-vs-hip
disagreement at three-quarter views are all RTMW3D model
limitations rather than algorithmic bugs. A SOTA upgrade here is
"swap the model"; the existing pose solver / depth pipeline would
adapt cleanly because the `SourceSkeleton` boundary already
abstracts the backend.

Deferred.

## What this file is NOT

- A roadmap of GPU runtime work — see
  [`gpu-runtime-roadmap-tasks.md`](./gpu-runtime-roadmap-tasks.md).
- An architecture critique — see
  [`architecture-pipeline-gui-critical-review.md`](./architecture-pipeline-gui-critical-review.md).
- A list of bugs — fixes flow through the regular commit history; this
  file only tracks SOTA *direction* upgrades.
