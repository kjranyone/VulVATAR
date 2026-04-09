# Shader Implementation Notes

## Purpose

This document captures implementation-facing shader notes for VulVATAR.

It is intentionally more technical and lower-level than:

- [rendering-materials.md](/C:/lib/github/kjranyone/VulVATAR/docs/rendering-materials.md)
- [mtoon-compatibility.md](/C:/lib/github/kjranyone/VulVATAR/docs/mtoon-compatibility.md)
- [vulkano-renderer-design.md](/C:/lib/github/kjranyone/VulVATAR/docs/vulkano-renderer-design.md)

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

The main avatar vertex shader should handle:

- vertex skinning
- optional cloth deformation application
- normal transformation
- tangent transformation if used
- clip-space projection
- passing normalized material inputs to the fragment stage

It should not contain:

- material-schema interpretation
- output sink logic
- tracking logic

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

Recommended first path:

- up to four weights per vertex
- skinning matrices uploaded per instance
- matrix blend in the vertex shader

Pseudo flow:

```text
skinned_position =
    weight0 * (joint_matrix0 * rest_position) +
    weight1 * (joint_matrix1 * rest_position) +
    weight2 * (joint_matrix2 * rest_position) +
    weight3 * (joint_matrix3 * rest_position)
```

The same principle applies to normals, but with the appropriate normal-space transform rather than blindly reusing the position path.

## Cloth Deformation Notes

The first renderer path should treat cloth as an additive deformation input or mapped replacement, depending on `ClothMeshMapping`.

Recommended first rule:

- skinning establishes base pose
- cloth deformation is applied after skinning in the vertex stage

This is consistent with cloth reacting to already-driven body motion.

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

## Failure Modes To Avoid

- shader branches directly on imported VRM field names
- hardcoding too many unrelated features into one shader family
- hiding alpha decisions in arbitrary fragment logic
- making outline behavior dependent on the main lighting path
- overfitting early shader logic to one avatar instead of representative samples
