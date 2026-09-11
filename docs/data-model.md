# Data Model

## Purpose

This document defines the core runtime and persistence-facing data shapes for VulVATAR.

It sits between:

- `architecture.md`, which defines ownership and layer boundaries
- the implemented Rust types (see "Implemented Shapes" at the end)

Related documents:

- [architecture.md](architecture.md)
- [project-persistence.md](project-persistence.md)
- [tracking-v2-design.md](tracking-v2-design.md)
- [output-interop.md](output-interop.md)

## Modeling Rules

- asset types are immutable after load
- runtime state is mutable and instance-local
- editor-only state must not leak into playback types
- GPU resource handles must not appear in asset-layer data
- identifiers used for persistence must be stable across reloads when possible

## Identifier Strategy

The project needs stable identifiers for rebinding authored overlays after reload.

Recommended identifier classes:

- `AvatarAssetId`
- `NodeId`
- `MeshId`
- `PrimitiveId`
- `MaterialId`
- `ClothOverlayId`

For persisted references, prefer explicit authored IDs over import-order indices.

## Asset Types

### `AvatarAsset`

Purpose:

- immutable avatar data produced by import

Fields:

- `id: AvatarAssetId`
- `source_path: PathBuf`
- `source_hash: AssetSourceHash`
- `skeleton: SkeletonAsset`
- `meshes: Vec<MeshAsset>`
- `materials: Vec<MaterialAsset>`
- `humanoid: Option<HumanoidMap>`
- `spring_bones: Vec<SpringBoneAsset>`
- `colliders: Vec<ColliderAsset>`
- `default_expressions: ExpressionAssetSet`

### `SkeletonAsset`

Purpose:

- immutable bone hierarchy and skinning source data

Fields:

- `nodes: Vec<SkeletonNode>`
- `root_nodes: Vec<NodeId>`
- `inverse_bind_matrices: Vec<Mat4>`

### `SkeletonNode`

Fields:

- `id: NodeId`
- `name: String`
- `parent: Option<NodeId>`
- `children: Vec<NodeId>`
- `rest_local: Transform`
- `humanoid_bone: Option<HumanoidBone>`

### `MeshAsset`

Fields:

- `id: MeshId`
- `name: String`
- `primitives: Vec<MeshPrimitiveAsset>`

### `MeshPrimitiveAsset`

Fields:

- `id: PrimitiveId`
- `vertex_count: u32`
- `index_count: u32`
- `material_id: MaterialId`
- `skin: Option<SkinBinding>`
- `bounds: Aabb`

### `MaterialAsset`

Fields:

- `id: MaterialId`
- `name: String`
- `base_mode: MaterialMode`
- `base_color: Vec4`
- `texture_bindings: MaterialTextureSet`
- `toon_params: ToonMaterialParams`

### `SpringBoneAsset`

Fields:

- `chain_root: NodeId`
- `joints: Vec<NodeId>`
- `stiffness: f32`
- `drag_force: f32`
- `gravity_dir: Vec3`
- `gravity_power: f32`
- `radius: f32`
- `collider_refs: Vec<ColliderRef>`

### `ColliderAsset`

Fields:

- `id: ColliderId`
- `node: NodeId`
- `shape: ColliderShape`
- `offset: Vec3`

## Cloth Types

### `ClothAsset`

Purpose:

- immutable authored overlay data that can be attached to an avatar instance

Fields:

- `id: ClothOverlayId`
- `target_avatar: AvatarAssetId`
- `target_avatar_hash: AssetSourceHash`
- `stable_refs: ClothStableRefSet`
- `simulation_mesh: ClothSimulationMesh`
- `render_bindings: Vec<ClothRenderRegionBinding>`
- `mesh_mapping: ClothMeshMapping`
- `pins: Vec<ClothPin>`
- `constraints: ClothConstraintSet`
- `collision_bindings: Vec<ClothCollisionBinding>`
- `lods: Vec<ClothLod>`
- `solver_params: ClothSolverParams`
- `metadata: ClothOverlayMetadata`

### `ClothStableRefSet`

Purpose:

- persisted references back into imported avatar data

Fields:

- `node_refs: Vec<NodeRef>`
- `mesh_refs: Vec<MeshRef>`
- `primitive_refs: Vec<PrimitiveRef>`

Rule:

- references must be resolvable without relying on raw import order alone

### `ClothSimulationMesh`

Fields:

- `vertices: Vec<ClothSimVertex>`
- `indices: Vec<u32>`
- `rest_lengths: Vec<f32>`
- `attachment_classes: Vec<AttachmentClassId>`
- `region_tags: Vec<ClothRegionTag>`

### `ClothRenderRegionBinding`

Fields:

- `primitive: PrimitiveRef`
- `vertex_subset: VertexSubsetRef`
- `mapping_region: ClothRegionTag`

### `ClothMeshMapping`

Purpose:

- maps simulated cloth motion back onto rendered geometry

Fields:

- `mapping_mode: ClothMappingMode`
- `entries: Vec<ClothMappingEntry>`

### `ClothSolverParams`

Fields:

- `substeps: u32`
- `iterations: u32`
- `gravity_scale: f32`
- `damping: f32`
- `self_collision: bool`
- `collision_margin: f32`
- `wind_response: f32`

## Runtime Types

### `AvatarInstance`

Purpose:

- live scene instance of one imported avatar

Fields:

- `id: AvatarInstanceId`
- `asset_id: AvatarAssetId`
- `asset: Arc<AvatarAsset>`
- `world_transform: Transform`
- `pose: AvatarPose`
- `animation_state: AnimationState`
- `secondary_motion: SecondaryMotionState`
- `expression_weights: Vec<ResolvedExpressionWeight>`
- `expression_state: ExpressionState`
- `retarget_state: RetargetState`
- `cloth_overlays: Vec<ClothOverlaySlot>`
- `collider_enabled: Vec<bool>`

### `AvatarPose`

Fields:

- `local_transforms: Vec<Transform>`
- `global_transforms: Vec<Mat4>`
- `skinning_matrices: Vec<Mat4>`
- `pose_timestamp: FrameTimestamp`

### `AnimationState`

Fields:

- `active_clip: Option<AnimationClipId>`
- `playhead_seconds: f32`
- `speed: f32`
- `looping: bool`

### `SecondaryMotionState`

Fields:

- `spring_states: Vec<SpringChainState>`
- `collision_cache: SecondaryMotionCollisionCache`

### `ClothState`

Fields:

- `overlay_id: ClothOverlayId`
- `enabled: bool`
- `sim_positions: Vec<Vec3>`
- `prev_sim_positions: Vec<Vec3>`
- `sim_normals: Vec<Vec3>`
- `constraint_cache: ClothConstraintRuntimeCache`
- `collision_cache: ClothCollisionRuntimeCache`
- `deform_output: ClothDeformOutput`

### `ClothDeformOutput`

Fields:

- `deformed_positions: Vec<Vec3>`
- `deformed_normals: Option<Vec<Vec3>>`
- `version: u64`

## Tracking Types

### `SourceSkeleton`

Purpose:

- published frame sample holding 2D/3D joints, face pose candidates, ARKit expressions, and metric camera intrinsics (`src/tracking/source_skeleton.rs`)

Fields:

- `timestamp: f64`
- `joints: HashMap<HumanoidBone, SourceJoint>`
- `face: Option<FacePose>`
- `expressions: Vec<SourceExpression>`
- `intrinsics: Option<CameraIntrinsics>`
- `metric: Option<MetricFrameInfo>`
- `rig: Option<RigPose>`

### `RigPose`

Purpose:

- normalized performer-driving data produced by the tracking-v2 fusion estimator and consumed by `avatar::retarget::apply_rig_pose` (`src/tracking/fusion/output.rs`)

Fields:

- `t: f64` (device capture timestamp)
- `bones: HashMap<HumanoidBone, RigBone>` (world deltas `delta_world` and marginal uncertainty `sigma`)
- `root_cam_m: [f32; 3]` (pelvis camera-space position in metres)
- `root_sigma_m: f32`
- `quality: f32`
- `shape_confidence: f32`
- `hand_confidence: [f32; 2]`
- `shoulder_span_m: f32`

## Output Types

### `OutputFrame`

Purpose:

- metadata wrapper for a completed render handoff

Fields:

- `frame_id: OutputFrameId`
- `timestamp: FrameTimestamp`
- `extent: [u32; 2]`
- `color_space: OutputColorSpace`
- `alpha_mode: AlphaMode`
- `gpu_token: Option<GpuFrameToken>`
- `handoff_path: HandoffPath`
- `fallback_reason: Option<FallbackReason>`
- `pixel_data: Option<Arc<Vec<u8>>>`

### `GpuFrameToken`

Purpose:

- ownership and synchronization token for GPU-resident output handoff

Fields:

- `resource_id: ExportedResourceId`
- `handle_type: ExternalHandleType`
- `external_handle: Option<u64>`
- `sync: OutputSyncToken`
- `lease: FrameLease`

Rule:

- output code may not touch renderer-owned images without a valid token

## Mailbox and Queue Types

### `TrackingMailbox`

Purpose:

- lock-bounded latest-sample exchange between tracking worker and app loop (`src/tracking/mod.rs`)

Fields:

- `pose`: `Arc<Mutex<PoseMailboxInner>>` (carries latest `RigPose`)
- `preview`: `Arc<Mutex<PreviewMailboxInner>>` (carries latest `SourceSkeleton` and detection annotation)
- `diagnostics`: `Arc<Mutex<DiagnosticsMailboxInner>>`

### `FrameSinkQueuePolicy`

Purpose:

- declares behavior when output sinks cannot keep up

Variants:

- `DropOldest`
- `DropNewest`
- `ReplaceLatest`
- `BlockNotAllowed`

## Implemented Shapes

The types above are all implemented; the code is the authoritative
field-level reference:

- `AvatarAsset`, `ClothAsset`, identifier types — `src/asset/mod.rs` (VRM via `src/asset/vrm/`, FBX via `src/asset/fbx/`, VRChat PhysBones via `src/asset/vrc/`)
- `AvatarInstance`, `AvatarPose` — `src/avatar/instance.rs`,
  `src/avatar/pose.rs` (pose stores local TRS, global matrices, and
  skinning matrices)
- `ClothState`, `ClothDeformOutput` (dual-path: CPU solver output,
  GPU-consumed SSBO) — `src/avatar/instance.rs`,
  `src/simulation/cloth.rs`
- `RigPose`, `SourceSkeleton`, `TrackingMailbox` — `src/tracking/fusion/output.rs`,
  `src/tracking/source_skeleton.rs`, `src/tracking/mod.rs`
- `OutputFrame`, `GpuFrameToken` — `src/output/mod.rs`,
  `src/frame_handoff.rs`

