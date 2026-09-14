# Body-SDF hair collision

Spring-bone (hair) chains resolve against a **body-surface distance
field** instead of per-collider capsules. The capsules were heuristic
stand-ins sized for a generic 1.6 m humanoid — on yumeka v1.0.3
(~1.34 m) the synthesized chest capsule's envelope reached ~12 cm above
the shoulder line, so hair visually collided with an invisible obstacle
above the shoulders. The field replaces that with the actual posed body
surface. As-Is only; no capsule fallback exists on the spring path.

## Data flow (per frame, one frame stale by construction)

The splat is expensive — ~21k body/face triangles each walking their
shell-expanded voxel AABB with `atomicMin` (hundreds of ms of GPU time
on the models measured). Two gates keep it off frames that don't need
it (`app/render.rs`):

- **Pose-unchanged skip**: the field is a pure function of the posed
  body, so `run_frame` hashes a quantized pose key
  (`splat_pose_key`: non-spring-bone local transforms + expression
  weights; spring nodes excluded — hair jitter never skins the
  splatted primitives) and drops the plan when it is unchanged. The
  renderer keeps last frame's field slot and the app keeps the last
  readback. An idle session (no tracking sample / no clip / settled
  hair) never re-splats.
- **30 Hz cadence gate**: while the pose is moving, the splat runs at
  most every 33 ms — the tracking cadence, so no field information is
  lost; intervening frames reuse the stale field (the pipeline is
  already one-frame-stale by construction).
- **`VULVATAR_NO_BODY_SDF=1`** drops the splat request entirely for
  A/B GPU-cost attribution (hair falls back to no collision).

The splat itself atomically writes a **device-local** field; a
field → host-visible staging copy at the end of the submission feeds
the readback. Do not move the field back into host-visible memory: the
~10⁷ scattered `atomicMin`s per splat are PCIe/BAR round-trips there
(measured collapse to ~1.5 fps with the splat on).

1. **App (`app/render.rs` `build_frame_input_multi`)** — per avatar
   instance, when spring is enabled and any chain has
   `SpringBoneAsset::body_collision`, the frame input carries
   `BodySdfPlan { prims, grid }`:
   - `prims` = body primitive (`clearance::find_body_primitive`, falling
     back to `asset.body_primitive_id`) **plus any face/head-named
     primitive** (the head is a separate primitive on many models; bangs
     and twintails collide against it).
   - `grid` = `SdfGrid::for_aabb(union of those prims' rest AABBs)` —
     NOT the asset root AABB, which a T-pose bind and hair prims bloat
     (coarser voxels for everyone). Strand joints outside the grid
     sample "no collision", which is correct: contacts only happen
     within the splat shell of these surfaces.
2. **Renderer compute prepass** (`renderer::sdf_field`,
   `pipeline::body_sdf_splat_cs`) — after every transform dispatch of
   the instance, sentinel-fill (`fill_buffer` u32::MAX) then one
   dispatch per splatted primitive. Thread = triangle; each thread
   walks its voxel AABB expanded by `SHELL_METRES` (60 mm) and
   `atomicMin`s the exact point-to-triangle distance as
   `floatBitsToUint` (valid for non-negative floats). Values are
   **node samples** (distance at the node position), which is what the
   CPU sampler's trilinear interpolation assumes.
3. **Readback** — the field lives device-local; the splat block ends
   with a `field → staging` copy into a host-visible buffer, which
   `render()` maps in the same window as the cloth readback (previous
   frame's fence already waited) and ships
   `RenderResult::sdf_fields` (`SdfFieldReadback { instance_id, grid,
   data }`). ~2.4 MB per instance at the 10 mm voxels yumeka gets.
4. **App fold** (`app/render.rs` `apply_sdf_readback`) — the field
   lands on `AvatarInstance::body_sdf`, keyed by `AvatarInstanceId`.
5. **Solve** (`simulation::spring::step_spring_bones`) — chains with
   `body_collision` project each joint out of
   `radius + SDF_CONTACT_MARGIN` along the sampled gradient, then
   re-enforce bone length; the correction folds into the Verlet
   implicit velocity (inelastic contact, same as the old collider
   projection).

## Field contract (`simulation/sdf.rs`)

- **Unsigned, truncated**: exact distance within `SHELL_METRES` of a
  splatted surface, `SENTINEL` (f32::MAX, `u32::MAX` bits) beyond.
- **No signing**: a joint pushed inside the body projects to the
  nearest surface either way, which is the behaviour hair needs.
- **Sampling**: trilinear over node values; unsplatted corners
  interpolate as `OUTSIDE_VALUE` (2× shell) so gradients stay finite at
  the band boundary. `gradient` is central differences; `resolve`
  returns None at a gradient null (field minimum) instead of pushing in
  a random direction.
- Voxel size: finest candidate (20 → 6 mm) whose cell count fits
  `MAX_CELLS` (≈1.05 M ≈ 4.2 MB). yumeka: 10 mm.

## Chain opt-in (`SpringBoneAsset::body_collision`, cache v22)

Hair/tail/wing categories collide against the field. Skirt chains opt
out (the GPU clearance field owns skirt anti-penetration) and so do
surface-hugging decorative chains (breast, belt) the field would float
off the skin. VRM chains default in and are filtered by chain-root name
(`vrm::extensions::apply_body_collision_flags`). `collider_refs` remain
on the asset for **cloth** only — the spring solver never reads them.

## Debug / verification

- `cargo run --bin diagnose_sdf_hair -- [<fbx|vrm>] [out_dir]` —
  headless end-to-end run (real renderer + readback + solver): prints
  the splat prim list, grid spec, per-chain min gap / tail height per
  checkpoint frame, renders back-view PNGs to
  `diagnostics/sdf_hair/`. Acceptance: no `PENETRATING` past the first
  frame, no contact above `shoulder_y`, back-hair tails hanging below
  the shoulder line.
- `cargo run --bin diagnose_hair_collider` — static capsule-envelope
  dump of the cached asset (kept for auditing what cloth still sees).

## Known limits

- One frame stale (body pose from frame N applied to hair in frame N+1)
  — invisible at hair dynamics timescales.
- Per-triangle splat cost scales with triangle AABB volume; a body mesh
  with degenerate macro-triangles would burn the loop (bounded by the
  grid, but wasteful). yumeka/musette body meshes are fine.
- Prim list caps at 4 splatted primitives per instance.
