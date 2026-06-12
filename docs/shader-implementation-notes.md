# Shader Implementation Notes

## Purpose

This document captures implementation-facing shader notes for VulVATAR.

It is intentionally more technical and lower-level than
[vulkano-renderer-design.md](/C:/lib/github/kjranyone/VulVATAR/docs/vulkano-renderer-design.md)
(which also owns the material strategy and MToon compatibility status).

It focuses on:

- vertex and fragment shader responsibilities
- normalized data flow into shaders
- implementation notes for toon-like shading
- outline handling
- alpha handling

## Core Rule

Shader code should implement renderer-normalized intent.

Shader code should not decode raw VRM schema conventions directly.

## Shader Families

Recommended first shader families:

- skinned unlit
- skinned simple lit
- skinned toon-like
- outline
- depth only if needed later

Each family should stay narrow in purpose.

Avoid giant feature-flag shaders as the default architecture.

## Vertex Shader Responsibilities

After the compute prepass migration (landed on
`feature/compute-prepass-migration`, 2026-05-19), the main avatar
vertex shader is intentionally tiny:

- clip-space projection (`view * proj` applied to the world-space input)
- passing normalized material inputs (UV, world position, world normal) to
  the fragment stage

Everything that used to live in the vertex stage — vertex skinning, morph
blending, cloth deformation, normal/tangent transformation — has moved to
the per-primitive **transform compute prepass** (`pipeline::transform_cs`)
that runs ahead of the graphics passes each frame. The prepass writes
world-space vertices into a persistent SSBO bound as the graphics vertex
buffer.

The vertex shader should not contain:

- material-schema interpretation
- output sink logic
- tracking logic
- skinning, morph, or cloth math (those belong to the compute prepass)

## Fragment Shader Responsibilities

The fragment shader should handle:

- base color sampling
- alpha evaluation
- lit or toon-like response
- outline color use in the outline pass only
- final color output in normalized renderer space

It should not:

- decide ownership policy
- perform scene graph logic
- special-case imported schema names

## Skinning Notes

Skinning lives in the transform compute prepass, not in the graphics
vertex shader. The shape is unchanged on the input side:

- up to four weights per vertex
- skinning matrices uploaded per instance (one SSBO per `AvatarInstance`,
  bound at compute set 1)

The blend itself is Dual Quaternion Skinning (DQS, Kavan et al. I3D
2007), not linear blend:

1. extract a rotation quaternion from each joint's skinning matrix
   (Mike Day's mat3→quat case-split — see `pipeline::transform_cs`)
2. pick the first non-zero-weight joint as the antipodal reference;
   flip subsequent contributing quaternions when `dot(qi, ref) < 0`
3. weighted-blend the (flipped) quaternions and renormalise
4. recover translation as the weighted average of each joint matrix's
   `m[3].xyz` (kept separate from the quaternion blend because the
   source matrices are `global · inverse_bind` products with
   already-world-space translations)
5. apply: `pos' = quat_rotate(q, pos) + t`,
   `nrm' = quat_rotate(q, nrm)`

`source_position` is the morph- and (optionally) cloth-resolved position
described below; the prepass applies skinning on top so cloth-bearing
primitives still ride the avatar's root transform. DQS eliminates the
candy-wrapper twist artefact at joints with non-trivial rotational
delta (forearm pronation, hip yaw, neck twist) that classic LBS shows.

Assumption: skinning matrices are rigid (rotation + translation only).
Non-uniform scale in inverse-bind matrices is silently lost by the
mat3→quat extraction.

## Cloth Deformation Notes

Cloth deformation runs inside the same transform compute prepass. The
CPU solver writes per-frame `vec4` positions (and optionally normals)
into the primitive's `cloth_pos_ssbo` / `cloth_norm_ssbo`, and the
prepass switches its position / normal source from the morph-blended
base to the cloth buffer when the per-primitive `has_cloth` /
`has_cloth_normals` flags in `TransformControl` are non-zero. Skinning
is applied after the cloth override so cloth-bearing primitives still
ride the avatar's root transform — consistent with cloth reacting to
already-driven body motion.

Cloth is **scoped per primitive** (`ClothDeformSnapshot::target_primitive_id`)
and per vertex subset (`vertex_offset` / `vertex_count`). The renderer
matches each snapshot against the primitive it targets and writes the
deform only into the subset range; vertices outside that range — and
all vertices of primitives the snapshot does not target — stay at
their morph-blended / skinning rest pose. Primitives with no matching
snapshot bind the shared stub SSBO and pay no cloth cost.

Target resolution: `ClothState::target_primitive_id` is populated from
the attached `ClothAsset::render_bindings[0]` at `init_cloth_sim` /
`init_cloth_overlay` time. A cloth that has been authored but not yet
bound to a render target (no `render_bindings`) simulates but does
not contribute to any draw — that's a deliberate "well-defined no
cloth" state instead of the old "broadcasts to every primitive in
the instance" footgun.

## Toon-Like Lighting Notes

The first toon-like path should be intentionally simple and stable.

Recommended components:

- base lit intensity term
- thresholded or ramped diffuse term
- optional ambient floor
- optional face-friendly clamp rules later

### Basic Diffuse Term

Start from a stable diffuse intensity:

```text
ndotl = max(dot(normal_ws, light_dir_ws), 0.0)
```

### Thresholded Toon Term

Simple first-pass toon response:

```text
toon_term = step(shadow_threshold, ndotl)
```

This is visually useful but harsh.

### Soft Toon Threshold

More practical first-pass alternative:

```text
toon_term = smoothstep(
    shadow_threshold - shadow_softness,
    shadow_threshold + shadow_softness,
    ndotl
)
```

This is usually a better starting point for faces than a hard step.

### Ramp Texture Option

Later extension:

```text
toon_term = sample(ramp_texture, vec2(ndotl, 0.5))
```

Keep the renderer contract broad enough to allow either threshold or ramp implementation.

## Face Stability Notes

Faces are unusually sensitive to shading errors.

Implementation guidance:

- avoid aggressive darkening from noisy normals
- prefer stable threshold ranges over extreme contrast
- test face materials under front and slight side lighting early

If face results are unstable, do not spend time on low-priority parameter parity first.

## Ambient Floor

For toon-like shading, an ambient floor is often useful:

```text
final_lit = mix(ambient_floor, 1.0, toon_term)
```

This can prevent fully crushed shadows on faces and hair in early implementations.

## Color Composition

A simple first-pass composition:

```text
final_rgb = base_color.rgb * final_lit
final_a   = base_color.a
```

This should stay explicit and easy to debug before adding more stylistic layers.

## Alpha Notes

The shader contract must support:

- opaque
- alpha test
- alpha blend

### Alpha Test

Recommended first-pass behavior:

```text
if base_alpha < alpha_cutoff:
    discard
```

Use this intentionally for cutout-style materials.

### Alpha Blend

For blended materials:

- fragment shader outputs alpha explicitly
- pipeline state decides blending behavior
- draw ordering is handled by renderer policy, not shader code

## Double-Sided Notes

If a material is double-sided:

- raster state should expose that choice
- shader code should not become responsible for policy-level cull decisions

If normal flipping for backfaces is needed later, add it explicitly and only where visually justified.

## Outline Shader Notes

The outline pass should stay separate.

Recommended first outline idea:

- expand along vertex normal in clip-relevant space or object/view space
- render with explicit outline color
- use pass-specific culling

Pseudo flow:

```text
outline_position = base_position + normal * outline_width
```

This is not the final formula, but it captures the responsibility split.

The outline path should not depend on the main fragment lighting model.

### Outline Stencil Masking (Implemented)

The outline pass **must** use stencil-buffer masking to prevent overwriting main geometry.

Without stencil masking, screen-space XY offset outlines will overdraw the model interior because back faces share the same depth as front faces (only XY is modified, not Z). This causes the entire body to render as dark maroon/black, especially visible on body/clothing meshes. Depth bias workarounds are insufficient because thin geometry (neck, fingers) has back faces closer than adjacent front faces at any camera distance.

Current implementation:

- Depth format: `D32_SFLOAT_S8_UINT` (includes stencil bits)
- Main pass: `stencil compare_op=Always, pass_op=Replace, reference=1` (writes stencil=1 on all drawn fragments)
- Outline pass: `stencil compare_op=NotEqual, reference=1, write_mask=0` (draws only where stencil != 1, i.e. silhouette edges)
- Outline pass: `depth compare_op=LessOrEqual, write_enable=false`
- Outline vertex shader: front-face culling, screen-space normal expansion

This guarantees the outline never overwrites interior pixels regardless of camera distance or mesh topology.

## Normal Handling

Normal quality strongly affects toon visuals.

Implementation notes:

- normalize after transformation
- keep tangent-space normal mapping optional
- test with and without normal maps before assuming they help

Poor normal handling will show up immediately on faces and hair.

## Debugging Aids

Useful material-debug shader modes:

- flat base color
- normal visualization
- `ndotl` visualization
- toon term visualization
- alpha visualization
- outline mask visualization

These should be easy to enable without rewriting shader logic.

### Implemented Debug Views

The fragment shader supports `material.debug_view`:

- `0`: normal rendering
- `1`: UV visualization (`fract(frag_uv)` as RG)
- `2`: base texture only (`tex_color.rgb` without base_color multiplication or lighting)

These are controlled via `MaterialDebugView` enum and `MaterialUniform.debug_view`.

### diagnose_vrm Binary

`src/bin/diagnose_vrm.rs` renders a VRM to a PNG for offline inspection.

```
diagnose_vrm <input.vrm> [output.png] [mode] [yaw_deg] [material_filter] [noskin]
```

Modes: `toon`, `simple`, `unlit`, `uv`, `tex`, `atlas`

This tool is essential for diagnosing rendering issues without the GUI loop. It also prints per-primitive UV stats with CPU-side texture sampling averages, useful for verifying UV correctness independently of the GPU pipeline.

## Failure Modes To Avoid

- shader branches directly on imported VRM field names
- hardcoding too many unrelated features into one shader family
- hiding alpha decisions in arbitrary fragment logic
- making outline behavior dependent on the main lighting path
- overfitting early shader logic to one avatar instead of representative samples
