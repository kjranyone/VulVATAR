# T06: MToon Extended Features

## Priority: P2

## Source

- `docs/mtoon-compatibility.md` — compatibility priorities
- `docs/rendering-materials.md` — staged roadmap
- `docs/shader-implementation-notes.md` — toon-like shading notes

## Status

**IMPLEMENTED**

- ✅ emissive_color, rim_lighting, uv_animation in fragment shader
- ✅ MaterialUniform extended with emissive/rim/uv_anim fields
- ✅ MtoonStagedParams values flow through to GPU
- ✅ MtoonCompatibilityStatus all flags set to true

## Current Code

- `src/renderer/pipeline.rs` — fragment shader with emissive add, Fresnel rim, UV scroll/rotation
- `src/renderer/material.rs` — MaterialUniform with emissive_color, rim_color, rim_fresnel_power, rim_lift, rim_lighting_mix, uv_anim_scroll_x/y, uv_anim_rotation
- `src/renderer/mtoon.rs` — all compatibility flags set to true

## Features

### Emissive

- [x] Add emissive color + intensity to `MaterialUploadRequest`
- [x] Add emissive texture slot to `MaterialTextureSlots`
- [x] Implement emissive contribution in fragment shader
- [x] Parse emissive parameters from VRM MToon extension

### Rim Lighting

- [x] Add rim color + Fresnel power + lift texture to material data
- [x] Implement rim lighting in fragment shader (Fresnel-based)
- [x] Parse rim parameters from VRM MToon extension

### UV Animation

- [x] Add UV scroll speed and rotation to material data
- [x] Apply UV animation in vertex or fragment shader
- [x] Parse UV animation from VRM MToon extension

### Shade Color / Shade Texture

- [x] Verify shade color texture sampling in toon path
- [x] Add shade-shifting step control if needed

### MatCap

- [x] Parse MatCap texture index from VRM MToon extension
- [x] Add MatCap sampling in fragment shader (view-space normal lookup + matcap_texture sampler + matcap_blend uniform)

## Acceptance Criteria

- `MtoonCompatibilityStatus` fields set to `true` as features land
- Test with representative VRM 1.0 avatars from `docs/mtoon-test-matrix.md`
- No visual regression on existing unlit/simple-lit/toon paths
