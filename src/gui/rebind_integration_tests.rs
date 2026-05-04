//! End-to-end coverage for the partial-rebinding path on avatar
//! reimport. The cloth_rebind algorithm is already unit-tested
//! (`src/asset/cloth_rebind.rs`); these tests exercise the GUI-side
//! dispatch + persistence layer through a real `.vvtcloth` file
//! and a real `GuiApp`.
//!
//! Coverage matrix:
//!
//! * **Clean** (name match, primary tier) — verified via
//!   `restore_cloth_overlay_paths`: overlay is attached, no
//!   notification, on-disk file untouched.
//! * **Failed** (unresolved name) — overlay is *not* attached,
//!   "Could not auto-bind" notification fires, on-disk file
//!   untouched.
//! * **Partial** (secondary/tertiary tier resolution) — currently
//!   unreachable through the resolver: see the TODO at
//!   `cloth_rebind.rs:198-204`. The persistence leg
//!   (`save_rebound_overlay` writing `last_rebound_with`) is
//!   covered by a focused IO test below; once tier-2/3 resolution
//!   lands, replace that with a true E2E Partial test.
use super::project::save_rebound_overlay;
use super::*;
use crate::asset::{
    Aabb, AssetSourceHash, AvatarAsset, AvatarAssetId, ClothAsset, ClothOverlayId, ClothPin,
    ExpressionAssetSet, NodeId, NodeRef, SkeletonAsset, SkeletonNode, Transform, VrmMeta,
};
use crate::avatar::{AvatarInstance, AvatarInstanceId};
use std::sync::Arc;

fn make_skeleton_node(id: u64, name: &str) -> SkeletonNode {
    SkeletonNode {
        id: NodeId(id),
        name: name.to_string(),
        parent: None,
        children: vec![],
        rest_local: Transform::default(),
        humanoid_bone: None,
    }
}

fn make_avatar_with_node(node_id: u64, node_name: &str) -> Arc<AvatarAsset> {
    let node = make_skeleton_node(node_id, node_name);
    Arc::new(AvatarAsset {
        id: AvatarAssetId(1),
        source_path: std::path::PathBuf::from("test.vrm"),
        source_hash: AssetSourceHash([0u8; 32]),
        skeleton: SkeletonAsset {
            root_nodes: vec![NodeId(node_id)],
            nodes: vec![node],
            inverse_bind_matrices: vec![],
        },
        meshes: vec![],
        materials: vec![],
        humanoid: None,
        spring_bones: vec![],
        colliders: vec![],
        default_expressions: ExpressionAssetSet { expressions: vec![] },
        animation_clips: vec![],
        node_to_mesh: std::collections::HashMap::new(),
        vrm_meta: VrmMeta::default(),
        root_aabb: Aabb::empty(),
        loaded_from_cache: false,
    })
}

fn install_avatar(harness: &mut GuiApp, asset: Arc<AvatarAsset>) {
    harness
        .app
        .add_avatar(AvatarInstance::new(AvatarInstanceId(1), asset));
}

fn make_overlay_pinning_to(name: &str, old_id: u64) -> ClothAsset {
    let mut overlay =
        ClothAsset::new_empty(ClothOverlayId(0), AvatarAssetId(1), AssetSourceHash([0u8; 32]));
    overlay.pins.push(ClothPin {
        sim_vertex_indices: vec![0],
        binding_node: NodeRef {
            id: NodeId(old_id),
            name: name.to_string(),
        },
        offset: [0.0, 0.0, 0.0],
    });
    overlay
}

/// Write an overlay to disk in the canonical `ClothOverlayFile`
/// envelope so `load_cloth_overlay` can read it back.
fn write_overlay_file(path: &std::path::Path, asset: &ClothAsset) {
    let file = crate::persistence::ClothOverlayFile {
        format_version: crate::persistence::OVERLAY_FORMAT_VERSION,
        created_with: "test".to_string(),
        last_saved_with: "test".to_string(),
        overlay_name: "test_overlay".to_string(),
        target_avatar_path: None,
        cloth_asset: Some(asset.clone()),
        last_rebound_with: None,
    };
    let json = serde_json::to_string_pretty(&file).expect("serialise overlay file");
    crate::persistence::atomic_write(path, &json).expect("write overlay file");
}

/// Allocate a unique tempdir per test so concurrent runs don't
/// collide on Windows.
fn make_tempdir(suffix: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("vulvatar_rebind_test_{}_{}", suffix, nanos));
    std::fs::create_dir_all(&dir).expect("create tempdir");
    dir
}

#[test]
fn clean_rebind_attaches_silently_and_does_not_touch_disk() {
    // Avatar exposes a "Hips" node at id=10. The overlay was authored
    // when the same name lived at id=99 — id-only validation would
    // break, but name-based primary-tier resolution should rebind it
    // cleanly, attach the overlay, and leave the file alone.
    let mut harness = GuiApp::for_test();
    install_avatar(&mut harness, make_avatar_with_node(10, "Hips"));

    let dir = make_tempdir("clean");
    let overlay_path = dir.join("clean.vvtcloth");
    let overlay_asset = make_overlay_pinning_to("Hips", 99);
    write_overlay_file(&overlay_path, &overlay_asset);

    let paths = vec![overlay_path.to_string_lossy().to_string()];
    harness.restore_cloth_overlay_paths(&paths);

    // Attached on the active avatar.
    let avatar = harness.app.active_avatar().expect("active avatar");
    assert_eq!(
        avatar.cloth_overlay_count(),
        1,
        "Clean rebind should still attach the overlay"
    );

    // No notification — Clean is a silent path.
    assert!(
        harness.notifications.is_empty(),
        "Clean rebind should not push a notification, got: {:?}",
        harness.notifications
    );

    // On-disk file's last_rebound_with stays None — Clean does not
    // trigger save_rebound_overlay.
    let on_disk = crate::persistence::load_cloth_overlay(&overlay_path)
        .expect("reload overlay after Clean rebind");
    assert!(
        on_disk.last_rebound_with.is_none(),
        "Clean must not stamp last_rebound_with"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn failed_rebind_skips_attach_and_pushes_notification() {
    // Avatar lacks the named bone the overlay was authored against —
    // primary tier fails, no fallback is wired up yet, so the
    // dispatcher must skip attach, post the "Could not auto-bind"
    // toast, and leave the file alone.
    let mut harness = GuiApp::for_test();
    install_avatar(&mut harness, make_avatar_with_node(10, "DifferentBone"));

    let dir = make_tempdir("failed");
    let overlay_path = dir.join("failed.vvtcloth");
    let overlay_asset = make_overlay_pinning_to("MissingBone", 99);
    write_overlay_file(&overlay_path, &overlay_asset);

    let paths = vec![overlay_path.to_string_lossy().to_string()];
    harness.restore_cloth_overlay_paths(&paths);

    // Did NOT attach.
    let avatar = harness.app.active_avatar().expect("active avatar");
    assert_eq!(
        avatar.cloth_overlay_count(),
        0,
        "Failed rebind must not attach the overlay"
    );

    // Notification fired with the Failed-path message.
    assert_eq!(
        harness.notifications.len(),
        1,
        "Failed rebind should post exactly one notification"
    );
    let msg = &harness.notifications[0].message;
    assert!(
        msg.contains("Could not auto-bind"),
        "expected 'Could not auto-bind' in notification, got: {}",
        msg
    );

    // Disk file unchanged — Failed does not call save_rebound_overlay.
    let on_disk = crate::persistence::load_cloth_overlay(&overlay_path)
        .expect("reload overlay after Failed rebind");
    assert!(on_disk.last_rebound_with.is_none());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn save_rebound_overlay_stamps_last_rebound_with_on_disk() {
    // Persistence-leg coverage for the (currently unreachable) Partial
    // dispatch path: when the dispatcher *would* call this on Partial,
    // it must produce a file whose last_rebound_with is set and whose
    // cloth_asset payload is the rebound (not the original) one. We
    // exercise the IO directly because the algorithm cannot yet
    // produce a Partial report — see the module docstring.
    let dir = make_tempdir("save");
    let overlay_path = dir.join("partial.vvtcloth");
    let original_asset = make_overlay_pinning_to("Hips", 99);
    write_overlay_file(&overlay_path, &original_asset);

    // Simulate the dispatcher having already rewritten the IDs in
    // memory before save (matching what the Partial branch does).
    let mut rebound_asset = original_asset.clone();
    rebound_asset.pins[0].binding_node.id = NodeId(7);

    save_rebound_overlay(&overlay_path, &rebound_asset).expect("save_rebound_overlay");

    let on_disk = crate::persistence::load_cloth_overlay(&overlay_path)
        .expect("reload after save_rebound_overlay");
    let stamp = on_disk
        .last_rebound_with
        .as_ref()
        .expect("save_rebound_overlay must stamp last_rebound_with");
    assert!(
        stamp.starts_with("VulVATAR "),
        "stamp should be 'VulVATAR <version>', got: {}",
        stamp
    );
    // last_saved_with is updated too — both should match.
    assert_eq!(&on_disk.last_saved_with, stamp);
    // Payload reflects the rebound IDs, not the originals.
    let saved_asset = on_disk
        .cloth_asset
        .expect("cloth_asset preserved through save");
    assert_eq!(saved_asset.pins[0].binding_node.id, NodeId(7));

    let _ = std::fs::remove_dir_all(&dir);
}
