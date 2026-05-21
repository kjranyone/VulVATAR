# SOTA algorithm upgrades

Date: 2026-05-22

This plan tracks the state-of-the-art algorithm upgrades applied to
VulVATAR's runtime. The first slice (XPBD on the CPU cloth solver)
landed in commit `fa82bbc`; subsequent slices and remaining
candidates are catalogued below for future-me / future maintainers.

## Landed

### Cloth constraint solver — PBD → XPBD (CPU)

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

**Files**:
- `src/simulation/cloth_solver/constraints.rs` — XPBD update rule
- `src/simulation/cloth_solver/mod.rs` — `reset_lambda` per substep
- `src/simulation/cloth.rs` — `ClothSimTempBuffers::lambda_distance` +
  `reset_lambda(count)` helper

**Tests**:
- `xpbd_rigid_stiffness_satisfies_constraint_in_one_pass`
- `xpbd_lambda_accumulates_within_substep_and_resets_across_substeps`
- `xpbd_disabled_constraint_leaves_lambda_zero`
- existing `cloth_constraint_jacobi_satisfies_rest_lengths_like_cpu_pbd`
  parity test stays correct because XPBD@α=0 reduces to PBD@stiffness=1
  algebraically

## Open SOTA slices

### Cloth constraint solver — XPBD on GPU

**Scope**: migrate `pipeline::cloth_constraint_accumulate_cs` and
`cloth_constraint_apply_cs` to XPBD too. The shader needs:

- A new lambda SSBO sized to the constraint count, bound at set 0
- The `dt` and per-constraint `α` (or shared α from material) added
  to `ClothConstraintControl` UBO
- `ensure_cloth_gpu_slot` (in `renderer/cloth_cache.rs`) allocating
  and resetting the lambda SSBO at slot construction + each frame
- The shader applying the XPBD update rule per-constraint via the
  existing Jacobi adjacency

**Why deferred**: the CPU↔GPU parity test passes at the current
stiffness (1.0) because XPBD@α=0 ≡ PBD@stiffness=1, so the GPU side
can stay on PBD until cloth authoring exposes compliance < 1.0. The
migration is mechanically similar to the CPU side but lives in GLSL
and adds a buffer.

**Done when**: the GPU constraint dispatch produces the same Δx as
the CPU XPBD reference at all stiffness values, and the parity test
covers stiffness ∈ {1.0, 0.7, 0.3} not just 1.0.

### Cloth normal recomputation — area-weighted vs angle-weighted

Current implementation: area-weighted face-normal accumulation
(triangle area × face normal summed per vertex, then normalised).

SOTA alternative: angle-weighted (Max 1999) — each face contributes
its normal scaled by the incident angle at the vertex. Less prone to
biased normals on irregular triangulations. Cost: cosine per triangle
edge.

Defer until a measurable visual issue motivates it.

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

### Skinning — Linear blend → Dual quaternion

Current: linear blend skinning (LBS) on the compute prepass output.
SOTA: dual quaternion skinning (Kavan 2007) eliminates the
"candy-wrapper" twist artefact at elbows / knees. The change is
local to `transform_cs`:

- Add dual-quaternion derivation from each bone's matrix
- Blend dual-quats per vertex (weighted, then normalised), apply

Cost: ~30% more arithmetic per vertex. Visual win is most
noticeable on extreme bone twist (forearm pronation, hip rotation).

Deferred. Comparable scope to GPU XPBD.

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
