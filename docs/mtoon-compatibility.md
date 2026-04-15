# MToon Compatibility

## Purpose

This document defines the compatibility strategy for VRM MToon-style materials in VulVATAR.

It is not a full shader implementation note.

It defines:

- what "MToon-like" means in this project
- what compatibility means by phase
- what should be implemented first
- what can be approximate or deferred

Related documents:

- [architecture.md](/C:/lib/github/kjranyone/VulVATAR/docs/architecture.md)
- [rendering-materials.md](/C:/lib/github/kjranyone/VulVATAR/docs/rendering-materials.md)
- [shader-implementation-notes.md](/C:/lib/github/kjranyone/VulVATAR/docs/shader-implementation-notes.md)
- [mtoon-test-matrix.md](/C:/lib/github/kjranyone/VulVATAR/docs/mtoon-test-matrix.md)

## Core Position

The project should target:

1. stable MToon-like presentation first
2. selective compatibility for important imported parameters next
3. closer parity later only where it materially improves actual avatars

This avoids turning the early renderer into a spec-chasing exercise.

## Terminology

### `MToon-like`

Means:

- toon-styled light response
- stable shadow threshold behavior
- outline support
- visually plausible avatar presentation

It does not mean exact parity with every MToon feature from the start.

### `Compatibility`

Means:

- imported VRM material intent maps into renderer behavior with predictable results

Compatibility can be:

- exact enough
- approximate
- unsupported

## Compatibility Priorities

The renderer should prioritize imported material features that most strongly affect avatar readability.

High priority:

- base color
- texture sampling
- alpha mode
- culling mode
- shade threshold behavior
- outline enable and width

Medium priority:

- subtle shading controls
- emissive behavior
- more exact shadow softness tuning

Lower priority for first passes:

- niche parameter interactions
- exact matching of secondary stylistic modifiers

## First Compatibility Matrix

### `Must Support Early`

- base color factor
- base color texture
- opaque, cutout, and blend distinction
- double-sided handling where needed
- toon-like diffuse thresholding
- outline on or off
- outline width

### `Approximate Early`

- exact ramp shape
- exact shadow softness behavior
- exact face-friendly lighting heuristics
- exact color blending nuances

### `Defer`

- long-tail parameter interactions that are rarely visible in common avatars
- niche effects with low visual impact compared to implementation cost

## Face Rendering Priority

Face materials deserve special scrutiny.

Even an approximate MToon implementation fails perceptually if faces look unstable, too dark, or over-shaded.

That means the compatibility effort should prioritize:

- stable front-facing facial lighting
- controlled shadow thresholding
- avoiding harsh artifacts across facial normals

## Hair Priority

Hair often depends on:

- alpha behavior
- sorting expectations
- culling correctness
- outline behavior

Hair rendering quality should be treated as a primary compatibility check, not an edge case.

## Outline Compatibility

Outline should be compatible at the intent level first.

That means:

- materials that want outlines should get them
- width should behave predictably
- color should be controllable

Exact internal implementation details can differ from other engines as long as the visual outcome stays within expectation.

## Transparency Compatibility

Transparency should be treated as a compatibility-critical area.

The renderer should explicitly decide:

- alpha cutoff behavior
- blended material ordering policy
- outline interaction with transparent materials

Do not leave these as accidental side effects of a generic pipeline.

## Compatibility Testing Strategy

The project should maintain a small representative avatar set and evaluate:

- face look
- hair look
- outline readability
- alpha behavior
- overall silhouette

Compatibility decisions should be judged against real avatar results, not only against parameter tables.

## Success Criteria

The early renderer succeeds if:

- common VRM avatars look recognizably correct
- faces remain stable and readable
- hair does not visibly break from alpha or culling issues
- outlines behave consistently

The renderer does not need full parity to be useful.

## Implementation Status and Known Issues

### Implemented MToon Features

- base color factor and texture (`pbrMetallicRoughness.baseColorFactor/Texture`)
- shade color factor and texture (`shadeColorFactor`, `shadeMultiplyTexture`)
- shade shift / shade toony parameters
- emissive color and texture
- rim color, fresnel power, lift, lighting mix
- alpha modes (opaque, mask, blend)
- double-sided rendering
- outline width and color (stencil-masked)

### Disabled Features

- **Matcap**: forced to blend=0.0. VRoid placeholder `MatcapWarp` textures (8x8 pixels) cause severe over-brightening. Requires proper MToon matcap sampling implementation.
- **UV animation**: scroll/rotation speeds are uploaded as raw values without time multiplication. Currently static only.

### Known Visual Issues

- **Toon shading lacks ambient term**: the ToonLike path multiplies by `light_color * light_intensity` without adding `ambient_term`. When normals face away from light, fragments go fully to `shade_color` with no ambient floor. This makes untextured or light-colored meshes appear dark/brown from certain angles. The `SimpleLit` path correctly includes ambient. The fix is to add an ambient floor to the toon path as described in shader-implementation-notes.md.
- **VRoid shade_color values**: VRoid Studio exports shade_color as warm/brown tones for skin materials. This is intentional MToon behavior, but combined with the missing ambient term it can look overly dark.

## Failure Modes To Avoid

- calling the renderer "MToon compatible" too early
- chasing low-impact edge parameters before fixing face or hair quality
- equating imported parameter presence with a requirement for exact implementation
- making compatibility decisions without checking real avatars
