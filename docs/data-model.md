# Data Model

## Purpose

This document defines the core runtime and persistence-facing data shapes for VulVATAR.

It sits between:

- `architecture.md`, which defines ownership and layer boundaries
- implementation, which will turn these contracts into concrete Rust types

Related documents:

- [architecture.md](/C:/lib/github/kjranyone/VulVATAR/docs/architecture.md)
- [project-persistence.md](/C:/lib/github/kjranyone/VulVATAR/docs/project-persistence.md)
- [tracking-retargeting.md](/C:/lib/github/kjranyone/VulVATAR/docs/tracking-retargeting.md)
- [output-interop.md](/C:/lib/github/kjranyone/VulVATAR/docs/output-interop.md)

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
- `world_transform: Transform`
- `pose: AvatarPose`
- `animation_state: AnimationState`
- `secondary_motion: SecondaryMotionState`
- `tracking_input: Option<TrackingRigPose>`
- `attached_cloth: Option<ClothOverlayId>`
- `cloth_enabled: bool`
- `cloth_state: Option<ClothState>`

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

### `TrackingFrame`

Purpose:

- raw or near-raw input sample from a tracking source

Fields:

- `source_id: TrackingSourceId`
- `capture_timestamp: TimeStamp`
- `frame_size: UVec2`
- `image_format: TrackingImageFormat`

### `TrackingRigPose`

Purpose:

- normalized performer-driving data for avatar retargeting

Fields:

- `source_timestamp: TimeStamp`
- `head: Option<RigTarget>`
- `neck: Option<RigTarget>`
- `spine: Option<RigTarget>`
- `shoulders: RigShoulderTargets`
- `arms: RigArmTargets`
- `hands: Option<RigHandTargets>`
- `expressions: ExpressionWeightSet`
- `confidence: TrackingConfidenceMap`

## Output Types

### `OutputFrame`

Purpose:

- metadata wrapper for a completed render handoff

Fields:

- `frame_id: OutputFrameId`
- `timestamp: FrameTimestamp`
- `extent: UVec2`
- `color_space: OutputColorSpace`
- `alpha_mode: AlphaMode`
- `gpu_token: GpuFrameToken`

### `GpuFrameToken`

Purpose:

- ownership and synchronization token for GPU-resident output handoff

Fields:

- `resource_id: ExportedResourceId`
- `handle_type: ExternalHandleType`
- `sync: GpuSyncToken`
- `lifetime: FrameLifetimeContract`

Rule:

- output code may not touch renderer-owned images without a valid token

## Mailbox and Queue Types

### `TrackingMailbox`

Purpose:

- lock-bounded or lock-free latest-sample exchange between tracking worker and app loop

Fields:

- `latest_pose: Option<TrackingRigPose>`
- `sequence: u64`

### `FrameSinkQueuePolicy`

Purpose:

- declares behavior when output sinks cannot keep up

Variants:

- `DropOldest`
- `DropNewest`
- `ReplaceLatest`
- `BlockNotAllowed`

## Rust Skeleton

The following sketch is intentionally incomplete but sets the shape:

```rust
pub struct AvatarAsset {
    pub id: AvatarAssetId,
    pub source_path: PathBuf,
    pub source_hash: AssetSourceHash,
    pub skeleton: SkeletonAsset,
    pub meshes: Vec<MeshAsset>,
    pub materials: Vec<MaterialAsset>,
    pub humanoid: Option<HumanoidMap>,
    pub spring_bones: Vec<SpringBoneAsset>,
    pub colliders: Vec<ColliderAsset>,
    pub default_expressions: ExpressionAssetSet,
}

pub struct ClothAsset {
    pub id: ClothOverlayId,
    pub target_avatar: AvatarAssetId,
    pub target_avatar_hash: AssetSourceHash,
    pub stable_refs: ClothStableRefSet,
    pub simulation_mesh: ClothSimulationMesh,
    pub render_bindings: Vec<ClothRenderRegionBinding>,
    pub mesh_mapping: ClothMeshMapping,
    pub pins: Vec<ClothPin>,
    pub constraints: ClothConstraintSet,
    pub collision_bindings: Vec<ClothCollisionBinding>,
    pub lods: Vec<ClothLod>,
    pub solver_params: ClothSolverParams,
    pub metadata: ClothOverlayMetadata,
}

pub struct AvatarInstance {
    pub id: AvatarInstanceId,
    pub asset_id: AvatarAssetId,
    pub world_transform: Transform,
    pub pose: AvatarPose,
    pub animation_state: AnimationState,
    pub secondary_motion: SecondaryMotionState,
    pub tracking_input: Option<TrackingRigPose>,
    pub attached_cloth: Option<ClothOverlayId>,
    pub cloth_enabled: bool,
    pub cloth_state: Option<ClothState>,
}

pub struct OutputFrame {
    pub frame_id: OutputFrameId,
    pub timestamp: FrameTimestamp,
    pub extent: UVec2,
    pub color_space: OutputColorSpace,
    pub alpha_mode: AlphaMode,
    pub gpu_token: GpuFrameToken,
}
```

## Next Decisions

The next unresolved modeling choices are:

- exact stable-reference encoding for cloth persistence
- whether `AvatarPose` stores matrices, TRS, or both as canonical
- whether expression weights live inside `TrackingRigPose`, `AvatarInstance`, or both
- whether `ClothDeformOutput` is CPU-side, GPU-side, or dual-path
