# Rendering Materials

## Purpose

This document defines the renderer-facing material strategy for VulVATAR.

It describes:

- the staged rendering roadmap
- material responsibilities in the renderer
- the relationship between generic material support and MToon compatibility

Related documents:

- [architecture.md](/C:/lib/github/kjranyone/VulVATAR/docs/architecture.md)
- [mtoon-compatibility.md](/C:/lib/github/kjranyone/VulVATAR/docs/mtoon-compatibility.md)
- [vulkano-renderer-design.md](/C:/lib/github/kjranyone/VulVATAR/docs/vulkano-renderer-design.md)
- [shader-implementation-notes.md](/C:/lib/github/kjranyone/VulVATAR/docs/shader-implementation-notes.md)

## Core Rule

The renderer should not block on full MToon compatibility before it can render avatars acceptably.

The material system should be staged:

1. `Unlit`
2. `SimpleLit`
3. `ToonLike`
4. `Outline`
5. broader MToon compatibility

## Renderer Material Goals

The first renderer should:

- draw skinned meshes correctly
- separate material interpretation from mesh submission
- support a stable minimal material parameter set
- allow later extension toward closer MToon behavior

## Material Layers

The material design should distinguish:

- imported material description
- normalized renderer material data
- GPU-uploaded material data

### Imported Material Description

This is what comes from VRM or glTF import.

It may contain:

- standard glTF material data
- VRM material metadata
- MToon-specific parameter sources

### Normalized Renderer Material Data

This is the renderer-facing abstraction.

It should translate imported material information into:

- shader model selection
- normalized parameter fields
- texture bindings
- transparency or alpha behavior
- outline behavior flags

### GPU-Uploaded Material Data

This is the final layout required by shaders and pipelines.

It should be shaped by renderer constraints, not by the serialized VRM schema directly.

## Staged Roadmap

### Phase 1: `Unlit`

Purpose:

- verify asset loading
- verify skinning
- verify texture binding

Expected support:

- base color
- base texture
- alpha handling at a minimal level

### Phase 2: `SimpleLit`

Purpose:

- establish a stable lit path before toon-specific behavior

Expected support:

- lambert or similarly simple light response
- normal usage if available
- basic shadow term if the renderer has it

### Phase 3: `ToonLike`

Purpose:

- reach the intended visual category before full compatibility work

Expected support:

- light ramping
- threshold-based shadow behavior
- stylized diffuse response
- stable facial rendering expectations

### Phase 4: `Outline`

Purpose:

- add silhouette support that materially changes avatar presentation

Expected support:

- enable or disable outline per material
- width control
- color control

### Phase 5: broader compatibility

Purpose:

- improve fidelity for imported VRM material settings

Expected support:

- more exact parameter mapping
- more exact transparency behavior
- more exact outline behavior

## Material Modes

Recommended normalized renderer modes:

- `Unlit`
- `SimpleLit`
- `ToonLike`

The importer should choose a mode, but the renderer may allow debugging overrides.

## Alpha Policy

Material handling must declare:

- opaque
- cutout or mask
- blend

This matters early because:

- face materials may depend on stable alpha behavior
- hair and outlines are affected by order and culling decisions
- OBS compositing depends on correct alpha output

## Culling Policy

Material normalization should surface culling intent explicitly.

Recommended flags:

- back-face cull
- double-sided
- front-face cull only if required by a specific pass

Do not bury this inside shader ad hoc logic.

## Outline Strategy

Outline should be treated as a renderer feature layered onto materials, not as a separate imported material family.

The material system should expose:

- outline enabled
- outline width
- outline color source

The exact implementation can change later, but the material contract should stay stable.

## Texture Policy

The normalized material contract should support:

- base color texture
- shade or ramp texture if needed later
- normal texture if used
- emissive texture if used later

Texture slots should be explicit even if some are optional in the first pass.

## Parameter Normalization

The importer should convert imported material sources into normalized renderer values.

That normalization step should absorb:

- unit differences
- naming differences
- unsupported field fallback behavior

This avoids spreading VRM-specific conditionals across shader code and draw submission code.

## Failure Modes To Avoid

- making the shader read raw VRM schema concepts directly
- blocking all rendering on full MToon parity
- mixing outline logic into generic mesh submission flow
- hiding alpha mode decisions inside arbitrary shader branches
- treating every imported material as a unique special case
