# Architecture

## Scope

This project targets `VRM 1.0 only`. The implemented scope:

- VRM 1.0 asset loading
- skinned rendering on `Vulkano`
- `MToon-like` shading
- secondary motion with spring bones
- cloth simulation for selected garments or accessories
- webcam-based motion tracking
- video output routable into OBS Studio

`VRM 0.x` compatibility is intentionally out of scope (a best-effort
loader shim exists, but no compatibility guarantees).

## Core Design Decision

Keep these systems separate:

- `asset layer`: parse VRM/glTF and build engine-native assets
- `editor / authoring layer`: create optional cloth overlays and save project-local authoring data
- `avatar layer`: own runtime skeleton, pose, animation, and simulation state
- `simulation layer`: update spring bones, colliders, and cloth
  - cloth simulation is CPU XPBD (`simulation/cloth_solver`,
    Macklin/Müller/Chentanez MIG 2016); a GPU compute path
    (`renderer/pipeline::cloth_verlet_cs`, `cloth_constraint_*_cs`,
    `cloth_normal_cs`) is wired and selected per-cloth via
    `ClothState::solver_backend`. Production attach paths default to
    `Cpu`; nothing in `RuntimeGpuBudget` currently flips the backend,
    so the GPU path is exercised only by parity tests and manual
    override today.
- `renderer layer`: own Vulkan resources and shading
- `tracking layer`: estimate performer motion from camera input
- `capture/output layer`: export rendered frames to external video consumers

The shared contract across those layers is the avatar pose and deformed render state, not Vulkan objects and not physics-engine objects.

Tracking should output a normalized rig-driving signal, not direct renderer commands. Capture/output should consume finished frames, not participate in scene logic.
Editor tools should author optional overlay assets, not rewrite the imported VRM into a runtime-only format.

## Recommended Runtime Model

### Asset Layer

Responsibilities:

- read `.vrm` files
- read optional project-local overlay data for authored features such as cloth
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
- optional `ClothAsset`

Rules:

- do not leak Vulkan types out of this layer
- do not leak Rapier types out of this layer
- absorb VRM-specific schema decisions here
- absorb project-local overlay schema decisions here

### Editor / Authoring Layer

Responsibilities:

- author optional cloth overlays for imported avatars
- save or load project-local cloth overlay data
- stay separate from playback-only runtime code

Outputs:

- optional `ClothAsset`
- editor metadata needed to re-open authored cloth setups

Rules:

- base VRM data remains generic imported content
- cloth is an on-demand authored overlay, not a required part of VRM import
- runtime must function when no cloth overlay exists
- editor state must not become a required dependency for playback

Detailed GUI flow and authoring interactions live in [editor-cloth-authoring.md](/C:/lib/github/kjranyone/VulVATAR/docs/editor-cloth-authoring.md).

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

- acquire webcam frames on a non-render-thread path
- run pose or face or hand estimation asynchronously
- retarget estimated landmarks into the avatar rig space
- smooth, filter, and stabilize tracking output
- publish the latest completed tracking sample to the main app loop

Outputs:

- `TrackingFrame`
- `TrackingRigPose`
- optional blendshape or expression weights

Rules:

- webcam and CV code do not become the authoritative skeleton state
- tracking writes into retargeting inputs, then avatar runtime resolves final pose
- tracking must tolerate missing landmarks and low-confidence frames
- the POC should run tracking asynchronously and hand off timestamped results through a narrow mailbox or queue
- render and simulation must not block on a slow inference step; they should consume the latest valid sample

### Capture / Output Layer

Responsibilities:

- consume completed rendered frames
- provide a route for OBS ingestion
- optionally expose alpha-capable avatar output
- manage frame pacing and color format conversion if needed
- hand off frames to sinks on a non-render-thread path
- apply explicit backpressure policy when sinks fall behind
- prefer GPU-resident shared-frame handoff over CPU readback when the target sink supports it

Outputs:

- virtual-camera-compatible video stream
- optional shared texture or shared-memory transport
- optional encoded stream for external tools
- synchronization objects needed to safely consume exported GPU frames

Rules:

- output should consume renderer results through a narrow handoff
- OBS integration should not force changes into avatar, tracking, or simulation logic
- the POC should publish frames to output sinks asynchronously
- output stalls should not block simulation or rendering; dropping or replacing stale frames is preferable to stalling the frame loop
- the preferred performance path is exportable GPU images plus explicit synchronization, not unconditional CPU copies

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

### Cloth authoring source

Base VRM assets stay generic.

Cloth is authored on demand inside this application and stored as project-local overlay data. The runtime then decides whether a given avatar instance attaches that overlay and enables cloth simulation for it.

That means:

- VRM import succeeds even when no cloth data exists
- cloth authoring and cloth playback are separate concerns
- `ClothAsset` comes from editor-authored overlay data, not from standard VRM 1.0 content
- the first runtime target should support zero or one optional cloth overlay per avatar instance

The recommended authored representation is not "a few attachment points with local cloth motion". It is a dedicated cloth simulation representation that can drive the rendered garment region. The editor-facing workflow for creating that data is specified separately in [editor-cloth-authoring.md](/C:/lib/github/kjranyone/VulVATAR/docs/editor-cloth-authoring.md).

### Recommended cloth architecture

Immutable asset side:

- `ClothSimulationMesh`
- `ClothRenderRegionBinding`
- `ClothMeshMapping`
- `ClothConstraintSet`
- `ClothPinSet`
- `ClothCollisionProxySet`
- `ClothLodSet`

Runtime side:

- `ClothInstance`
- `ClothSolverState`
- `ClothDeformOutput`

Integration options:

- `skinned source mesh -> cloth sim on selected region -> final deformed render mesh`
- `bone-driven attachment points -> cloth local simulation -> render buffer update`

For this project direction, the first option is the target architecture. A dedicated simulation mesh can be lower resolution than the final render mesh, but the runtime contract should preserve a clear mapping from simulation output to rendered geometry.

For the first implementation pass, simplify by limiting garment scope and collision complexity, not by collapsing the data model into attachment-point cloth.

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
- exported GPU memory handle type and synchronization primitive

### Output handoff model

`handoff` means the ownership and synchronization contract between renderer and output.

For a high-performance path, that contract should be:

- renderer produces an exportable GPU image
- renderer publishes a frame token that includes the image handle, metadata, and sync object
- output consumes that token asynchronously and releases it when the sink is done
- renderer must not recycle the underlying image until the handoff contract says it is safe

The fallback path can still allow CPU readback for debugging or unsupported sinks, but that should not be the architectural center.

If you want compositing in OBS, alpha support becomes an architectural requirement, not a later convenience.

## Data Model

The core type split — immutable assets (`AvatarAsset`, `ClothAsset`)
vs. mutable runtime state (`AvatarInstance`, `AvatarPose`,
`SecondaryMotionState`, `ClothState`) vs. handoff values
(`TrackingRigPose`, `OutputFrame`) — is specified field-by-field in
[data-model.md](/C:/lib/github/kjranyone/VulVATAR/docs/data-model.md);
the implemented structs in `src/asset/mod.rs`, `src/avatar/instance.rs`,
and `src/output/mod.rs` are the authoritative shapes.

## Frame Update Order

Recommended order:

1. input update
2. read the latest completed tracking sample from the async tracking worker
3. locomotion or gameplay update
4. animation sampling
5. base local pose build
6. merge tracking inputs
7. compute driven global pose for the body before secondary motion
8. fixed-timestep spring bone simulation against that pose
9. recompute global pose after spring outputs are applied
10. fixed-timestep cloth simulation against the resolved body motion for this frame
11. recompute final global pose
12. build skinning matrices
13. upload pose buffers
14. upload optional cloth deformation buffers
15. render
16. publish output frame to the async output sink

Important rule:

Cloth should read the resolved body motion for this frame, not stale matrices from the previous render path.

Tracking should be merged before secondary motion, otherwise spring and cloth will react to stale performer input.

Tracking acquisition and inference still happen, but they happen off the main render loop. The frame loop consumes the latest completed tracking result rather than waiting for a fresh inference every frame.

### Timestep policy

The simulation contract should also state:

- use an accumulator-based fixed timestep for spring and cloth
- define a maximum substep count per render frame to avoid spiral-of-death behavior
- define whether render uses latest-sim state directly or interpolates between sim states
- latch tracking input once per render frame before entering simulation steps
- if tracking or output falls behind, prefer stale-sample reuse or frame drop over blocking the main loop

## GUI → pipeline settings wiring

Every user-facing setting reaches the pipeline through exactly one of
four sanctioned paths. When adding a new setting, pick the matching
path — do not invent a fifth, and do not mix paths within one setting.

### Path 1: Reconciled (`GuiApp::sync_app_settings`)

Continuous scene / view / output parameters: viewport camera, lighting,
background, output extent / fps / alpha / colour space / MSAA, avatar
world transform.

- The GUI struct (`RenderingGuiState`, `OutputGuiState`, …) owns the
  value and is what persistence reads/writes.
- `sync_app_settings` pushes the value into its `Application` mirror
  field once per `update`, before `run_frame`. Writes must be
  idempotent; `Application` setters that do real work on change
  (e.g. `set_user_render_fps`) must be cheap to call every frame.
- Use this for values the render thread or output worker reads outside
  the frame step.

### Path 2: FrameConfig (`GuiApp::build_frame_config`)

Per-frame pipeline inputs consumed inside `Application::run_frame`:
runtime toggles, smoothing params, hand/face/lower-body/root-translation
flags, fade-on-loss, mouth source, material mode.

- The GUI struct owns the value; it rides a `FrameConfig` value struct
  into `run_frame` and never needs an `Application` field.
- Use this for values that only the frame step (pose solver,
  simulation, material selection) reads.

### Path 3: Requested (Phase D, `Application::set_requested_*`)

Stateful runtime resources that can fail to start or stop: output sink,
lipsync capture (and, structurally, the tracking worker).

- `Application` owns both the *requested* state (user intent, what
  persistence saves) and the *active* state (what is actually running);
  they diverge when a runtime transition fails.
- The GUI holds **no** copy — it reads the requested/active pair via
  getters and writes via `set_requested_*`. Failures surface as
  notifications; the requested value survives for retry/autosave.
- Use this whenever applying the setting can fail at runtime.

### Path 4: Instance-bound (`AvatarInstance` fields)

Per-avatar runtime state: expression weights, collider enable mask,
cloth attachments, `avatar.cloth_enabled`.

- Inspector widgets bind directly to the `AvatarInstance` field; there
  is no GUI-side copy and no project persistence. The state resets with
  the avatar (reload / switch) by design.
- Use this for state whose lifetime *is* the avatar instance's.

### Anti-patterns

- A GUI-side shadow copy of a Path 3 value (drifts on runtime failure —
  the exact bug Phase C/D removed).
- Gating a persisted Path 2 toggle on non-persisted session state
  (the pre-refactor `toggle_cloth && sim_playing` bug: cloth silently
  froze after every restart).
- Syncing a value into an `Application` field *and* passing it through
  `FrameConfig` — one consumer model per setting.

## Failure Modes To Avoid

- loader creates GPU buffers directly
- renderer owns the authoritative skeleton hierarchy
- Rapier rigid bodies become the source of truth for avatar bones
- variable timestep drives spring or cloth solvers
- cloth is forced into the spring-bone abstraction
- cloth data model collapses into attachment-point-only simulation and blocks higher-fidelity garments later
- exact MToon parity becomes the first rendering milestone
- render or simulation blocks on tracking inference
- render loop blocks on output sink latency
- cloth authoring data is mixed into generic imported VRM data without an explicit overlay boundary
- output handoff falls back to unconditional CPU readback as the primary architecture

## Repository Mapping

- `src/app/`: orchestration and frame update order
- `src/asset/`: VRM 1.0 loading and immutable asset conversion
- `src/editor/`: cloth authoring tools and overlay persistence
- `src/avatar/`: runtime avatar state and pose ownership
- `src/simulation/`: spring, cloth, collider, and future world-physics logic
- `src/renderer/`: Vulkano rendering and material pipelines
- `src/tracking/`: webcam tracking and retargeting inputs
- `src/output/`: OBS-facing and external frame sink integration

The implementations the original baseline anticipated are all in
place; large modules have since grown into directory modules with
per-concern sibling files:

- `src/avatar/animation.rs`, `src/avatar/instance.rs`, `src/avatar/pose.rs`
- `src/editor/cloth_authoring.rs`
- `src/renderer/material.rs`, `src/renderer/mtoon.rs`
- `src/simulation/spring.rs`
- `src/simulation/cloth_solver/{mod,integrator,constraints,collision,output,tests}.rs`
- `src/asset/vrm/{mod,extensions,gltf_decode,mtoon,v0,v1,metadata_tests}.rs`
- `src/tracking/rtmw3d/{mod,consts,decode,preprocess,skeleton,wrist,face,annotation,session,math}.rs`
- `src/gui/inspector/{mod,avatar,preview,tracking,rendering,output,cloth,library,settings}.rs`

Directory discipline: split by responsibility inside each top-level
domain rather than letting any single file grow past ~1.5 kLOC. Never
add `src/types.rs` / `src/utils.rs` / `src/common.rs` — dumping-ground
modules erase the architectural boundaries.

## Dependency Direction Rules

Keep dependencies one-way:

- `app -> asset, editor, avatar, simulation, renderer, tracking, output`
- `editor -> asset`
- `avatar -> asset, tracking`
- `simulation -> avatar, asset`
- `renderer -> avatar`
- `tracking -> no avatar ownership, outputs timestamped rig-driving data only`
- `output -> no scene ownership, consumes final frames only`

Avoid reverse edges such as:

- `asset -> renderer`
- `asset -> editor`
- `simulation -> renderer`
- `tracking -> renderer`
- `output -> avatar`
- `avatar -> simulation`
- `renderer -> output`

If a feature pressures you into one of those edges, the boundary is probably wrong.

## Document Map

For application-level GUI structure and user workflows, see [application-design.md](/C:/lib/github/kjranyone/VulVATAR/docs/application-design.md).

Detailed contracts per domain:

- [data-model.md](/C:/lib/github/kjranyone/VulVATAR/docs/data-model.md) — data shape contracts and identifier strategy
- [project-persistence.md](/C:/lib/github/kjranyone/VulVATAR/docs/project-persistence.md) — file formats, versioning, migration
- [threading-model.md](/C:/lib/github/kjranyone/VulVATAR/docs/threading-model.md) — thread roles, ownership, shutdown order
- [gpu-runtime-roadmap.md](/C:/lib/github/kjranyone/VulVATAR/docs/gpu-runtime-roadmap.md) — GPU pressure policy (`RuntimeGpuBudget`)
- [tracking-retargeting.md](/C:/lib/github/kjranyone/VulVATAR/docs/tracking-retargeting.md) — tracking → rig contract
- [onnx-tracking-pipeline.md](/C:/lib/github/kjranyone/VulVATAR/docs/onnx-tracking-pipeline.md) — inference pipeline, coordinate conventions
- [calibration-ux.md](/C:/lib/github/kjranyone/VulVATAR/docs/calibration-ux.md) — pose calibration spec
- [output-interop.md](/C:/lib/github/kjranyone/VulVATAR/docs/output-interop.md) — GPU frame handoff / VGTK sidecar protocol
- [mf-virtual-camera.md](/C:/lib/github/kjranyone/VulVATAR/docs/mf-virtual-camera.md) — MediaFoundation virtual camera contract
- [vulkano-renderer-design.md](/C:/lib/github/kjranyone/VulVATAR/docs/vulkano-renderer-design.md) — renderer design, API boundary, materials, MToon status
- [shader-implementation-notes.md](/C:/lib/github/kjranyone/VulVATAR/docs/shader-implementation-notes.md) — shader-level contracts and gotchas
- [editor-cloth-authoring.md](/C:/lib/github/kjranyone/VulVATAR/docs/editor-cloth-authoring.md) — cloth authoring UX spec + status
- [model-library.md](/C:/lib/github/kjranyone/VulVATAR/docs/model-library.md) — avatar library and folder watching
- [profiling.md](/C:/lib/github/kjranyone/VulVATAR/docs/profiling.md) — instrumentation recipes and known bottlenecks
