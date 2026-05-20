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
- [ ] Compute pipelines for the three solver stages
- [ ] CPU → compact-control SSBO upload path
- [ ] CPU PBD path retained behind `ClothSolverBackend::Cpu` for fallback /
      diagnostics
- [ ] Runtime switch consulting `RuntimeGpuBudget` (see [`runtime-gpu-budget.md`])

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

- [ ] **S0** decide particle state representation: Verlet (pos + prev_pos)
      vs explicit velocity. Document the choice in a comment in
      `cloth_gpu_boundary.rs`
- [ ] **S1.1** add `ClothGpuSimulationState` SSBO fields (particles,
      constraints, colliders, normals, control UBO)
- [ ] **S1.2** implement particle-integration compute shader; CPU writes
      control UBO each frame
- [ ] **S1.3** unit test on a 4-particle synthetic cloth: gravity over N
      frames matches CPU reference within tolerance
- [ ] **S2.1** implement constraint-projection compute; upload constraint
      table once at attach time
- [ ] **S2.2** integration test: hang a 32-particle sheet, compare
      steady-state height vs CPU reference
- [ ] **S3.1** implement normal recomputation compute
- [ ] **S3.2** snapshot test: rendered cloth visually matches CPU path
      (within tolerance) under no wind
- [ ] **S4** suppress the CPU snapshot write for `ClothSolverBackend::Gpu`
      cloths in `renderer::mod.rs`; flip `has_cloth` based on GPU version
- [ ] **S5** wire `ClothSolverBackend` selection into the cloth attach
      path (default `Cpu` until `RuntimeGpuBudget` decides otherwise)

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
