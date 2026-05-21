# GPU cloth solver compute migration (P3-02)

Date: 2026-05-20
Roadmap reference: [`gpu-runtime-roadmap-tasks.md` P3-02](./gpu-runtime-roadmap-tasks.md)

## Goal

Move the cloth particle solver from CPU (Verlet + PBD constraints in
`simulation::cloth_solver`) onto the GPU as a compute pipeline that writes
directly into the per-primitive cloth SSBOs already consumed by
`renderer::pipeline::transform_cs`.

End state: the CPU only ships compact control data (rest-length tables,
collider transforms, wind / gravity, pin targets, LoD switch) per frame; all
per-particle integration, constraint projection, self-collision, and normal
recomputation happen on the GPU. The current `ClothDeformSnapshot` Vec round
trip through `RenderAvatarInstance::cloth_deforms` collapses to a "the GPU
buffer is fresh, render it" version counter.

## Why

The roadmap rule is "large frame-shaped data stays on the GPU; the CPU moves
compact control data and lifetime tokens." Cloth is the largest remaining
violator: every frame the CPU integrates particles, projects constraints,
and copies `Vec<Vec3>` payloads into `ClothDeformSnapshot.deformed_positions`
/ `deformed_normals`, which the renderer then re-copies into the per-primitive
SSBO. At 60 Hz with a moderate skirt mesh that is several MB/s of pure
shuffling for data that the GPU could have produced in place.

It also unblocks higher-fidelity cloth: more particles, more substeps, and
self-collision become viable when the cost is GPU-side.

## Status

- [x] CPU PBD solver (`simulation::cloth_solver`) — production
- [x] Per-primitive cloth SSBO bound by `transform_cs` — production
- [x] Render-consumable boundary type
      ([`ClothRenderConsumableDeform`](../src/simulation/cloth_gpu_boundary.rs))
      and placeholder [`ClothGpuSimulationState`] — landed in P3-01
- [x] **S0 — particle state representation locked: Verlet (pos +
      prev_pos)**. Documented in `cloth_gpu_boundary.rs` alongside the
      lifecycle and SSBO slot layout the compute landing will use
- [x] `ClothGpuSimulationState` fleshed out with counts +
      `iterations` + `version`; `from_authoring` constructor in place
- [x] `ClothSolverBackend::{Cpu, Gpu}` with `Default = Cpu`; intent
      switch available at attach time even though the renderer still
      runs the CPU snapshot path for both variants
- [ ] Compute pipelines for the three solver stages (S1.2, S2.1, S3.1)
- [ ] CPU → compact-control SSBO upload path (S1.1 Vulkan side)
- [ ] Direct-draw consumption suppressing the per-primitive
      `cloth_pos_ssbo.write` calls (S4)
- [ ] Runtime switch consulting `RuntimeGpuBudget`
      (see [`runtime-gpu-budget.md`]; depends on B4 cross-thread
      plumbing landing first)

## Architecture target

```text
Frame:
  upload cloth control data (small per-frame UBO)
  compute pass: integrate particles      [SSBO read/write per primitive]
  compute pass: project constraints      [SSBO read/write per primitive, N iters]
  compute pass: recompute normals        [SSBO read; SSBO write normals]
  transform_cs (existing)                [reads cloth_pos / cloth_norm SSBO]
  graphics passes (existing)
```

No CPU readback of particle positions on the steady-state path. The CPU
holds only a "version increased" boolean per primitive so it knows whether to
flip the `has_cloth` flag in the transform control UBO.

## Solver stages (incremental landing order)

### Stage 1 — particle integration (Verlet)

Inputs (per primitive):
- particle position SSBO (current)
- particle previous position SSBO (or velocity SSBO; pick one and document)
- constants UBO: dt, gravity vec3, wind vec3, damping
- pin mask + pin target SSBO

Output:
- new position SSBO
- new previous position SSBO (or velocity)

CPU writes constants and pin targets; everything else stays GPU-side. One
dispatch per primitive, `local_size_x = 64` to match `transform_cs`.

### Stage 2 — constraint projection (PBD)

Inputs:
- particle position SSBO
- constraint table SSBO: `(particle_a, particle_b, rest_length, stiffness)`
- collider list SSBO: sphere / capsule transforms (mirrors `cloth.rs`
  `resolve_scene_colliders`)

Output:
- particle position SSBO (in-place; subsequent iterations re-read)

Constraint indices are uploaded once per cloth asset, not per frame. Multiple
iterations (currently 8 on CPU) are achieved by re-dispatching the same
pipeline N times — no per-iteration CPU bookkeeping.

Self-collision is **out of scope for stage 2**; land distance + collider
constraints first, then iterate.

### Stage 3 — normal recomputation

Inputs:
- particle position SSBO
- triangle index SSBO (per cloth asset)

Output:
- vertex normal SSBO

Reads triangle indices, accumulates face normals, normalizes per vertex.

### Stage 4 — direct draw consumption

Once stages 1-3 produce GPU-resident positions and normals, the
per-primitive `cloth_pos_ssbo` / `cloth_norm_ssbo` writes in
`renderer::mod.rs` (lines around `cloth_snap_opt = instance.cloth_deforms
.iter().find(...)`) become unnecessary for `ClothSolverBackend::Gpu`. The
control UBO still flips `has_cloth = 1`, but the buffer contents come
directly from the compute output.

## Scope

In:
- `src/renderer/pipeline.rs` — three new compute shaders + pipeline creation
- `src/renderer/mod.rs` — cloth SSBO ownership transition; dispatch wiring
- `src/simulation/cloth_gpu_boundary.rs` — fill in
  `ClothGpuSimulationState` with real SSBO handles
- `src/simulation/cloth_solver/` — split CPU path behind
  `ClothSolverBackend::Cpu`; keep tests green

Out:
- self-collision (defer; CPU path keeps it, GPU path documents the gap)
- cloth authoring changes — the GPU path consumes the same
  `ClothAuthoring` (ex-`ClothAsset`)
- runtime-budget integration — only consume the enum once
  [`runtime-gpu-budget.md`] lands

## Tasks

- [x] **S0** Verlet (pos + prev_pos) chosen; rationale in
      `cloth_gpu_boundary.rs` (mirrors CPU PBD so its reference tests
      double as compute-side smoke values, halves per-particle SSBO
      writes during constraint iteration)
- [x] **S1.1 (CPU side)** `ClothGpuSimulationState` fleshed out with
      `particle_count`, `constraint_count`, `iterations`, `version`,
      `from_authoring` constructor, `bump_version` helper. SSBO slot
      shape documented in module rustdoc
- [ ] **S1.1 (Vulkan side)** allocate per-primitive SSBOs sized by
      `ClothGpuSimulationState.particle_count` /
      `constraint_count` at cloth-attach time
- [~] **S1.2** GLSL compute shader for Verlet integration; Vulkano
      pipeline; CPU writes control UBO each frame
  - [x] `cloth_verlet_cs` module landed in `src/renderer/pipeline.rs`
        with the Verlet formula mirroring `cloth_solver::integrator`
        bit-for-bit (pos / prev_pos SSBOs + `Control` UBO)
  - [x] `create_cloth_verlet_compute_pipeline` factory + Rust mirror
        `ClothVerletControl` (std140 layout, 48 B, asserted by test)
  - [ ] Per-frame CPU UBO upload + dispatch (waits on S1.1 Vulkan side
        SSBO handles)
- [ ] **S1.3** unit test: 4-particle synthetic cloth under gravity
      matches CPU reference (`verlet_gravity_pulls_down`) within
      `1e-3` over 60 frames
- [ ] **S2.1** GLSL compute shader for PBD constraint projection;
      constraint table uploaded once at attach time
- [ ] **S2.2** integration test: 32-particle sheet drape matches CPU
      reference within tolerance
- [~] **S3.1** GLSL compute shader for vertex normal recomputation
      (face-normal accumulate + per-vertex normalise)
  - [x] `cloth_normal_cs` module landed in `src/renderer/pipeline.rs`
        — vertex-per-thread, CSR-adjacency driven, area-weighted
        accumulation matching `cloth_solver::output::compute_normals`
  - [x] `create_cloth_normal_compute_pipeline` factory + Rust mirror
        `ClothNormalControl` (std140 layout, 16 B, asserted by test)
  - [x] CPU adjacency builder `build_vertex_triangle_adjacency` in
        `cloth_gpu_boundary.rs` (CSR; out-of-range indices skipped;
        5 unit tests covering single tri / shared vertex / isolated
        vertex / corrupt index / empty input)
  - [ ] Per-frame dispatch wiring (waits on S1.1 Vulkan side SSBO
        handles)
- [ ] **S3.2** rendered-cloth snapshot diff vs CPU path
- [ ] **S4** suppress the CPU snapshot write for
      `ClothSolverBackend::Gpu` cloths in `renderer::mod.rs`; flip
      `has_cloth = 1` based on `ClothGpuSimulationState.version`
- [x] **S5 (scaffolding)** `ClothSolverBackend::{Cpu, Gpu}` available
      at the avatar attach boundary with `Default = Cpu`. Production
      attach paths still default to `Cpu` (no `Gpu` consumer until
      `RuntimeGpuBudget` B4 ships and the compute pipelines land)

## What this slice ships

A structural scaffold for the migration: the boundary types are
production-shaped, the backend enum is selectable, the particle state
representation is decided, and the SSBO slot layout is documented. The
actual compute shaders, Vulkano pipelines, and round-trip parity tests
are deferred to follow-up slices.

The deferred work splits cleanly along Vulkano boundaries: each
compute stage is its own pipeline + dispatch site, and each stage's
test compares against the existing CPU PBD solver's outputs (no
external reference data needed). Picking up where this slice left off
means filling in the SSBO handles on `ClothGpuSimulationState`,
wiring the (already-landed) `cloth_verlet_cs` pipeline into the
renderer dispatch path, and adding the 4-particle test (S1.3).

The Verlet integration compute shader and Vulkano pipeline factory
themselves have already landed alongside this plan as additive prep
(see `pipeline.rs::cloth_verlet_cs` + `create_cloth_verlet_compute_pipeline`).

## Verification

| Scenario | Must prove |
|---|---|
| 32-particle sheet, gravity only | GPU and CPU paths agree within 1e-3 |
| sphere collider, sheet drape | both paths produce sane drape silhouette |
| pin-cloth (single anchor) | pinned particles do not move; rest swings |
| 60 fps render with cloth | no per-frame `cloth_pos_ssbo.write()` calls in profile log |
| disable GPU path mid-run | CPU fallback resumes without artifact |

## Risks

- **Vulkano SSBO ownership**: per-primitive SSBOs already exist; risk is
  ownership-during-dispatch when multiple cloths share a frame. The
  per-primitive `TransformGpuData` slot model from compute-prepass-migration
  applies cleanly.
- **Constraint table churn**: re-uploading the constraint table on every
  attach is fine; re-uploading per frame would defeat the migration.
- **Self-collision parity gap**: CPU path keeps self-collision; document the
  GPU gap explicitly in the inspector so it isn't perceived as a
  GPU-path regression.

## Done when

- a cloth marked `ClothSolverBackend::Gpu` produces the same steady-state
  shape as the CPU path (within tolerance) on a smoke test
- the profile log shows zero per-frame `cloth_pos_ssbo.write()` calls for
  GPU-path cloths after warm-up
- CPU path remains a first-class fallback for `ClothSolverBackend::Cpu`
- the inspector tells the user which backend is active
