# Architecture

## Scope

This project targets `VRM 1.0 only`.

The near-term goal is not full feature completeness. The goal is to build a POC architecture that can grow into:

- VRM 1.0 asset loading
- skinned rendering on `Vulkano`
- `MToon-like` shading
- secondary motion with spring bones
- cloth simulation for selected garments or accessories
- webcam-based motion tracking
- video output routable into OBS Studio

`VRM 0.x` compatibility is intentionally out of scope.

## Core Design Decision

Keep these systems separate:

- `asset layer`: parse VRM/glTF and build engine-native assets
- `avatar layer`: own runtime skeleton, pose, animation, and simulation state
- `simulation layer`: update spring bones, colliders, and cloth
- `renderer layer`: own Vulkan resources and shading
- `tracking layer`: estimate performer motion from camera input
- `capture/output layer`: export rendered frames to external video consumers

The shared contract across those layers is the avatar pose and deformed render state, not Vulkan objects and not physics-engine objects.

Tracking should output a normalized rig-driving signal, not direct renderer commands. Capture/output should consume finished frames, not participate in scene logic.

## Recommended Runtime Model

### Asset Layer

Responsibilities:

- read `.vrm` files
- parse glTF buffers, meshes, skins, nodes, animations, materials
- parse VRM 1.0 extensions
- convert source data into engine-native immutable assets

Outputs:

- `AvatarAsset`
- `SkeletonAsset`
- `MeshAsset`
- `MaterialAsset`
- `SpringBoneAsset`
- `ColliderAsset`
- `ClothAsset`

Rules:

- do not leak Vulkan types out of this layer
- do not leak Rapier types out of this layer
- absorb VRM-specific schema decisions here

### Avatar Layer

Responsibilities:

- own one live avatar instance in the scene
- hold local and global transforms
- evaluate animation
- merge animation, spring, and cloth outputs
- build final skinning matrices

Key runtime types:

- `AvatarInstance`
- `AvatarPose`
- `AnimationState`
- `SecondaryMotionState`
- `ClothState`

This is the center of the architecture. Renderer and simulation read or write through explicit interfaces, but neither should become the source of truth for the skeleton.

### Simulation Layer

Responsibilities:

- fixed timestep updates
- spring bone updates
- collider evaluation
- cloth simulation updates
- optional world-physics integration

Outputs:

- bone-space or node-space deltas for spring chains
- simulated cloth particle or vertex state
- collision contacts needed by secondary motion

Important distinction:

- `spring bone` is secondary motion on bone chains
- `cloth simulation` is surface or particle/constraint simulation on garment regions

Do not model cloth as "just many spring bones". That shortcut collapses once you need surface preservation, bend constraints, self-collision decisions, or pinned garment regions.

### Renderer Layer

Responsibilities:

- swapchain and frame lifecycle
- GPU resource ownership
- skinned mesh pipelines
- material pipelines
- upload of final pose and cloth-deformed data
- lighting and post effects

Inputs:

- meshes
- materials
- final skinning matrices
- cloth-deformed vertex buffers or simulation output
- camera and lights

Renderer should be able to draw with:

- final skeleton pose
- optional per-frame cloth deformation buffers
- material parameters

without caring how simulation produced them.

### Tracking Layer

Responsibilities:

- acquire webcam frames
- run pose or face or hand estimation
- retarget estimated landmarks into the avatar rig space
- smooth, filter, and stabilize tracking output

Outputs:

- `TrackingFrame`
- `TrackingRigPose`
- optional blendshape or expression weights

Rules:

- webcam and CV code do not become the authoritative skeleton state
- tracking writes into retargeting inputs, then avatar runtime resolves final pose
- tracking must tolerate missing landmarks and low-confidence frames

### Capture / Output Layer

Responsibilities:

- consume completed rendered frames
- provide a route for OBS ingestion
- optionally expose alpha-capable avatar output
- manage frame pacing and color format conversion if needed

Outputs:

- virtual-camera-compatible video stream
- optional shared texture or shared-memory transport
- optional encoded stream for external tools

Rules:

- output should consume renderer results through a narrow handoff
- OBS integration should not force changes into avatar, tracking, or simulation logic

## Physics Strategy

Do not make `Rapier` the center of the avatar runtime.

That is usually the wrong abstraction for VRM-style secondary motion. Hair, chest, ribbons, tails, and similar parts are judged by visual stability first, not by rigid-body realism.

Recommended strategy:

1. implement spring bones in a dedicated solver
2. add collider queries against avatar and world
3. add cloth simulation as a separate solver path
4. integrate Rapier only where world interaction actually benefits from it

Good candidates for Rapier:

- scene colliders
- character controller body
- props that must interact with the world
- broad-phase collision support for cloth or attachments

Bad candidate for first ownership:

- the avatar skeleton itself

## Motion Tracking Strategy

Webcam tracking is a separate input problem, not an animation substitute.

Recommended pipeline:

1. webcam frame acquisition
2. human landmark estimation
3. head, face, upper body, and optional hand extraction
4. confidence filtering and temporal smoothing
5. retargeting to avatar rig coordinates
6. merge with authored animation or idle stabilization

Tracking output should be treated as:

- head orientation
- neck and spine hints
- shoulder, arm, and hand targets
- facial expression weights
- blink and mouth parameters

not as raw final bone transforms.

### Tracking abstraction

Suggested types:

- `TrackingSource`
- `TrackingFrame`
- `TrackingConfidenceMap`
- `RetargetingProfile`
- `TrackingRigPose`

### Recommended first tracking scope

For the POC, target:

- head pose
- torso orientation
- upper arms
- facial expressions if feasible

Do not make lower-body full-body webcam inference a hard requirement for the first pass unless your camera setup and CV stack already support it robustly.

## Cloth Simulation Positioning

Cloth belongs in simulation, but it needs a different data model from spring bones.

### Why cloth is separate

Cloth needs concepts that bone springs do not:

- particles or simulated vertices
- distance constraints
- bend or shear constraints
- pinned vertices
- optional volume preservation
- collision proxies against body and environment

### Recommended first cloth target

Start with a narrow target:

- skirt hem
- coat flap
- cape edge
- ribbon sheet

Avoid full-body garment simulation in the first pass.

### Recommended cloth architecture

Immutable asset side:

- `ClothMeshRegion`
- `ClothConstraintSet`
- `ClothPinSet`
- `ClothCollisionProxySet`

Runtime side:

- `ClothInstance`
- `ClothSolverState`
- `ClothDeformOutput`

Integration options:

- `skinned source mesh -> cloth sim on selected region -> final deformed render mesh`
- `bone-driven attachment points -> cloth local simulation -> render buffer update`

For the POC, the second option is safer.

## MToon-like Shader Strategy

Do not block the renderer on full MToon compatibility.

Recommended stages:

1. `unlit` skinned mesh
2. `simple lit` skinned mesh
3. toon ramp and shadow threshold controls
4. outline pass
5. MToon-like parameter set
6. tighter VRM 1.0 material compatibility

The phrase `MToon-like` matters here. The first milestone is to achieve the visual category and material control shape, not exact spec parity.

### Suggested renderer split

- `SkinningPipeline`
- `ToonForwardPipeline`
- `OutlinePipeline`
- `DepthOnlyPipeline`
- `MaterialUploader`

## OBS Routing Strategy

There are multiple ways to get avatar video into OBS. Architecturally, they should all sit behind the capture/output layer.

Preferred output targets:

1. virtual camera output
2. shared texture or GPU interop path
3. shared-memory frame output
4. offline or network encoder path

For a POC, the cleanest architectural target is a pluggable frame sink:

- `FrameSink::VirtualCamera`
- `FrameSink::SharedTexture`
- `FrameSink::SharedMemory`
- `FrameSink::ImageSequence`

This keeps renderer ownership simple:

- renderer produces a completed color target
- output layer copies or maps it into the selected sink

### Output format concerns

Decide these early:

- RGB or RGBA output
- premultiplied alpha or straight alpha
- target frame rate
- sRGB versus linear handoff

If you want compositing in OBS, alpha support becomes an architectural requirement, not a later convenience.

## Data Model

### `AvatarAsset`

Immutable loaded data:

- skeleton hierarchy
- inverse bind matrices
- mesh primitives
- material definitions
- humanoid mapping
- spring definitions
- collider definitions
- cloth region definitions

### `AvatarInstance`

Live scene object:

- world transform
- current local pose
- current global pose
- animation runtime state
- spring runtime state
- cloth runtime state

### `TrackingRigPose`

Normalized performer input:

- head transform target
- body joint targets
- hand targets
- facial parameter weights
- confidence per channel

### `AvatarPose`

Final resolved skeleton state:

- local TRS per node
- global matrices
- skinning matrices

### `SecondaryMotionState`

Runtime state for spring chains and colliders:

- verlet or explicit integration caches
- collision cache
- chain-local parameters

### `ClothState`

Runtime state for cloth:

- particle positions
- previous particle positions or velocities
- constraint buffers
- pinned region bindings
- collision working set

### `OutputFrame`

Completed render handoff:

- color image handle
- optional alpha image handle
- width and height
- timestamp
- color space metadata

## Frame Update Order

Recommended order:

1. input update
2. webcam tracking acquisition
3. tracking inference and retargeting
4. locomotion or gameplay update
5. animation sampling
6. base local pose build
7. merge tracking inputs
8. spring bone simulation
9. cloth simulation
10. recompute final global pose
11. build skinning matrices
12. upload pose buffers
13. upload cloth deformation buffers
14. render
15. publish output frame to selected sink

Important rule:

Cloth should read the resolved body motion for this frame, not stale matrices from the previous render path.

Tracking should be merged before secondary motion, otherwise spring and cloth will react to stale performer input.

## POC Milestones

### Phase 0: Asset + Skeleton

Exit criteria:

- load one VRM 1.0 avatar
- build skeleton and skinning matrices
- render a static or bind pose

### Phase 1: Basic Rendering

Exit criteria:

- skinned mesh rendering on Vulkano
- camera control
- simple lit or unlit material path

### Phase 2: Secondary Motion

Exit criteria:

- spring bone solver works on selected chains
- avatar colliders affect spring motion

### Phase 3: Toon Shading

Exit criteria:

- toon-like lighting ramp
- outline pass
- core MToon-like controls

### Phase 4: Cloth POC

Exit criteria:

- one selected cloth region simulates stably
- cloth pins to skeleton correctly
- body collision proxies affect cloth

### Phase 5: World Physics

Exit criteria:

- optional Rapier-backed world interaction
- cloth and spring can query world collision where needed

### Phase 6: Tracking + Output

Exit criteria:

- webcam tracking drives head and upper body
- retargeted motion feeds the avatar runtime cleanly
- rendered output is routable to OBS through a frame sink

## Failure Modes To Avoid

- loader creates GPU buffers directly
- renderer owns the authoritative skeleton hierarchy
- Rapier rigid bodies become the source of truth for avatar bones
- variable timestep drives spring or cloth solvers
- cloth is forced into the spring-bone abstraction
- exact MToon parity becomes the first rendering milestone

## Repository Mapping

- `src/app/`: orchestration and frame update order
- `src/asset/`: VRM 1.0 loading and immutable asset conversion
- `src/avatar/`: runtime avatar state and pose ownership
- `src/simulation/`: spring, cloth, collider, and future world-physics logic
- `src/renderer/`: Vulkano rendering and material pipelines
- `src/tracking/`: webcam tracking and retargeting inputs
- `src/output/`: OBS-facing and external frame sink integration

The next likely additions are:

- `src/avatar/animation.rs`
- `src/avatar/retargeting.rs`
- `src/renderer/material.rs`
- `src/renderer/mtoon.rs`
- `src/simulation/spring.rs`
- `src/simulation/cloth_solver.rs`

## Recommended Directory Structure

The crate should stop being flat once real implementation begins. A scalable starting point is:

```text
src/
  main.rs
  app/
    mod.rs
  asset/
    mod.rs
    vrm.rs
  avatar/
    mod.rs
  simulation/
    mod.rs
    cloth.rs
  renderer/
    mod.rs
  tracking/
    mod.rs
  output/
    mod.rs
```

As the codebase grows, split by responsibility inside each top-level domain rather than adding more top-level files.

Good examples:

- `src/avatar/pose.rs`
- `src/avatar/instance.rs`
- `src/avatar/retargeting.rs`
- `src/simulation/spring.rs`
- `src/simulation/cloth_solver.rs`
- `src/renderer/pipeline.rs`
- `src/renderer/material.rs`
- `src/output/frame_sink.rs`

Bad direction:

- `src/types.rs`
- `src/utils.rs`
- `src/common.rs`

Those files quickly become dumping grounds and erase the architectural boundaries.

## Dependency Direction Rules

Keep dependencies one-way:

- `app -> asset, avatar, simulation, renderer, tracking, output`
- `avatar -> asset, simulation::cloth, tracking`
- `simulation -> avatar, asset`
- `renderer -> avatar, output`
- `tracking -> no avatar ownership, outputs rig-driving data only`
- `output -> no scene ownership, consumes final frames only`

Avoid reverse edges such as:

- `asset -> renderer`
- `simulation -> renderer`
- `tracking -> renderer`
- `output -> avatar`

If a feature pressures you into one of those edges, the boundary is probably wrong.

## Recommended Next Design Task

The next useful step is not more implementation. It is to pin down concrete Rust structs for:

- `AvatarAsset`
- `AvatarInstance`
- `AvatarPose`
- `SpringBoneAsset`
- `ClothAsset`
- `ClothState`
- `TrackingRigPose`
- `OutputFrame`

That will decide whether the architecture is actually clean enough to implement.
