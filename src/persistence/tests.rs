use super::*;
use serde_json::json;

fn settings_tempdir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "vulvatar_app_settings_{tag}_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create tempdir");
    dir
}

#[test]
fn app_settings_roundtrip() {
    let dir = settings_tempdir("roundtrip");
    let path = dir.join("settings.json");
    // Every field differs from `AppSettings::default()` so a save
    // or load that silently fell back to defaults is caught.
    let original = AppSettings {
        format_version: 1,
        locale: "ja".to_string(),
        zoom_sensitivity: 0.007,
        orbit_sensitivity: 0.9,
        pan_sensitivity: 3.5,
        cloth_autosave_consent: Some(true),
        last_project_path: Some("C:/projects/stream.vvtproj".to_string()),
        camera_serial: Some("1234567890".to_string()),
        cloth_gpu_backend: Some(true),
        auto_cloth: Some(false),
    };
    save_app_settings_to(&original, &path).expect("save");
    let loaded = load_app_settings_from(&path).expect("load");
    assert_eq!(loaded.locale, "ja");
    assert_eq!(loaded.zoom_sensitivity, 0.007);
    assert_eq!(loaded.orbit_sensitivity, 0.9);
    assert_eq!(loaded.pan_sensitivity, 3.5);
    assert_eq!(loaded.cloth_autosave_consent, Some(true));
    assert_eq!(
        loaded.last_project_path.as_deref(),
        Some("C:/projects/stream.vvtproj")
    );
    assert_eq!(loaded.camera_serial.as_deref(), Some("1234567890"));
    assert_eq!(loaded.cloth_gpu_backend, Some(true));
    let _ = std::fs::remove_dir_all(&dir);
}

/// A `settings.json` written before `last_project_path` existed
/// must keep loading (serde default → `None`), not reset the user's
/// preferences to defaults via the parse-failure fallback.
#[test]
fn app_settings_without_last_project_path_still_load() {
    let dir = settings_tempdir("pre_last_project");
    let path = dir.join("settings.json");
    let pre_split = json!({
        "format_version": 1,
        "locale": "zh",
        "zoom_sensitivity": 0.004,
        "orbit_sensitivity": 0.6,
        "pan_sensitivity": 2.0,
        "cloth_autosave_consent": null
    });
    std::fs::write(&path, pre_split.to_string()).expect("write old settings");
    let loaded = load_app_settings_from(&path).expect("old settings must parse");
    assert_eq!(loaded.locale, "zh");
    assert_eq!(loaded.last_project_path, None);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn app_settings_missing_file_is_none() {
    let dir = settings_tempdir("missing");
    assert!(load_app_settings_from(&dir.join("nope.json")).is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Pre-split installs carry the app settings inside the project
/// file's `settings` block. The one-time migration must lift them
/// out so an upgrade doesn't reset the user's language.
#[test]
fn legacy_last_session_settings_migrate() {
    let dir = settings_tempdir("migrate");
    let legacy_session = dir.join("last_session.vvtproj");
    // Minimal legacy file: serde(default)s fill the config blocks,
    // but `settings` must be the genuine pre-split shape.
    let legacy = json!({
        "format_version": 1,
        "created_with": "0.1.0",
        "last_saved_with": "0.1.0",
        "avatar_source_path": null,
        "avatar_source_hash": null,
        "avatar_transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": 1.0},
        "active_overlay_path": null,
        "tracking": {},
        "rendering": {},
        "output": {},
        "settings": {
            "locale": "ko",
            "zoom_sensitivity": 0.004,
            "orbit_sensitivity": 0.6,
            "pan_sensitivity": 2.0,
            "cloth_autosave_consent": false
        }
    });
    std::fs::write(&legacy_session, legacy.to_string()).expect("write legacy");
    let migrated =
        migrate_legacy_app_settings_from(&legacy_session).expect("legacy settings present");
    assert_eq!(migrated.locale, "ko");
    assert_eq!(migrated.zoom_sensitivity, 0.004);
    assert_eq!(migrated.cloth_autosave_consent, Some(false));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Post-split project files omit the `settings` key entirely
/// (skip_serializing_if) — re-saving a legacy file must not write
/// the key back, and a settings-less file must not migrate.
#[test]
fn post_split_project_files_omit_settings() {
    let minimal = json!({
        "format_version": 1,
        "created_with": "0.1.0",
        "last_saved_with": "0.1.0",
        "avatar_source_path": null,
        "avatar_source_hash": null,
        "avatar_transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": 1.0},
        "active_overlay_path": null,
        "tracking": {},
        "rendering": {},
        "output": {}
    });
    let file: ProjectFile = serde_json::from_value(minimal).expect("settings-less project parses");
    assert!(file.settings.is_none());
    let rewritten = serde_json::to_value(&file).expect("serialise");
    assert!(
        rewritten.get("settings").is_none(),
        "project files must not carry app settings any more"
    );
}

#[test]
fn tracking_config_silently_drops_legacy_smoothing_fields() {
    // `smoothing_strength`, `confidence_threshold`, and
    // `smoothing_joint_confidence` were removed when the GUI
    // sliders went away (the last after the v1 pose solver). Older
    // `.vvtproj` files still carry them; serde must accept the
    // unknown fields and load the rest of the config rather than
    // failing the load.
    let legacy = json!({
        "smoothing_strength": 0.7,
        "confidence_threshold": 0.4,
        "smoothing_joint_confidence": 0.3,
        "mirror": true,
    });
    let cfg: TrackingConfig =
        serde_json::from_value(legacy).expect("legacy fields must be ignored on load");
    assert!(cfg.mirror, "non-removed fields still load");
}

#[test]
fn read_format_version_defaults_to_zero_when_missing() {
    let no_field = json!({"foo": "bar"});
    assert_eq!(read_format_version(&no_field), 0);
}

#[test]
fn read_format_version_picks_up_field() {
    let with_field = json!({"format_version": 7});
    assert_eq!(read_format_version(&with_field), 7);
}

#[test]
fn migrate_chain_applies_each_step_with_correct_version_in_order() {
    // Real migrators don't exist yet — without this fake-stepper
    // exercise, the loop's "apply step → bump version → repeat" logic
    // would only fire for the first time on the day a v2 lands. Use a
    // stepper that records the (version → output) sequence so we can
    // verify each step sees the previous step's result with the right
    // version label.
    let stepper = |mut v: serde_json::Value, version: u32| -> Result<_, String> {
        let key = format!("step_{}", version);
        v.as_object_mut().unwrap().insert(key, json!(version));
        Ok(v)
    };
    let result = migrate_chain(json!({"start": true}), 0, 3, stepper).unwrap();
    let obj = result.as_object().unwrap();
    assert_eq!(obj.get("start"), Some(&json!(true)));
    assert_eq!(obj.get("step_0"), Some(&json!(0)));
    assert_eq!(obj.get("step_1"), Some(&json!(1)));
    assert_eq!(obj.get("step_2"), Some(&json!(2)));
    // Target version is 3 → step at version 3 must NOT be called.
    assert!(!obj.contains_key("step_3"));
}

#[test]
fn migrate_chain_propagates_step_error_and_stops() {
    // If a stepper returns an error mid-chain, the loop must abort
    // immediately — no later steps fire and the original error
    // bubbles up unmodified.
    let calls = std::cell::RefCell::new(Vec::new());
    let stepper = |v: serde_json::Value, version: u32| -> Result<_, String> {
        calls.borrow_mut().push(version);
        if version == 1 {
            Err(format!("boom at v{}", version))
        } else {
            Ok(v)
        }
    };
    let err = migrate_chain(json!({}), 0, 4, stepper).unwrap_err();
    assert!(
        err.contains("v1"),
        "expected error to mention v1, got: {}",
        err
    );
    // Stepper called for v0 (success) and v1 (failure) only — v2/v3 must
    // not be reached after the error.
    assert_eq!(*calls.borrow(), vec![0, 1]);
}

#[test]
fn migrate_chain_at_target_does_not_invoke_step() {
    // from == to is the "already current" case. The stepper must not
    // be called at all — invoking it would corrupt up-to-date files.
    let stepper = |_: serde_json::Value, _: u32| -> Result<_, String> {
        panic!("stepper should not be called when from_version == target_version");
    };
    let result = migrate_chain(json!({"x": 1}), 5, 5, stepper).unwrap();
    assert_eq!(result, json!({"x": 1}));
}

#[test]
fn migrate_cloth_overlay_at_current_version_is_passthrough() {
    let original = json!({"format_version": OVERLAY_FORMAT_VERSION, "overlay_name": "x"});
    let migrated = migrate_cloth_overlay_json(original.clone(), OVERLAY_FORMAT_VERSION).unwrap();
    assert_eq!(migrated, original);
}

#[test]
fn migrate_cloth_overlay_from_v0_surfaces_missing_migrator() {
    // No v0 → v1 migrator is currently registered. This test documents
    // the framework's failure mode: we want a clear error rather than
    // a silent accept, and we want the chain to fire on a real older
    // version. When a v2 lands, this test will need to be updated to
    // use whatever the *new* "below current" version is.
    let original = json!({"overlay_name": "x"});
    let err =
        migrate_cloth_overlay_json(original, 0).expect_err("expected error from un-migratable v0");
    assert!(
        err.contains("no migrator"),
        "expected 'no migrator' message, got: {}",
        err
    );
}

#[test]
fn load_cloth_overlay_rejects_future_version() {
    let dir = std::env::temp_dir().join("vulvatar_overlay_future_test");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("future.vvtcloth");
    let body = json!({
        "format_version": OVERLAY_FORMAT_VERSION + 99,
        "created_with": "test",
        "last_saved_with": "test",
        "overlay_name": "future",
        "target_avatar_path": null,
        "cloth_asset": null,
    });
    std::fs::write(&path, serde_json::to_string(&body).unwrap()).unwrap();
    let err = load_cloth_overlay(&path).expect_err("future version should be rejected");
    assert!(
        err.contains("newer VulVATAR"),
        "expected 'newer VulVATAR' message, got: {}",
        err
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn load_cloth_overlay_at_current_version_round_trips() {
    let dir = std::env::temp_dir().join("vulvatar_overlay_current_test");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("current.vvtcloth");
    let body = json!({
        "format_version": OVERLAY_FORMAT_VERSION,
        "created_with": "test",
        "last_saved_with": "test",
        "overlay_name": "current",
        "target_avatar_path": null,
        "cloth_asset": null,
    });
    std::fs::write(&path, serde_json::to_string(&body).unwrap()).unwrap();
    let overlay = load_cloth_overlay(&path).expect("v1 should load directly");
    assert_eq!(overlay.overlay_name, "current");
    assert_eq!(overlay.format_version, OVERLAY_FORMAT_VERSION);
    assert!(overlay.cloth_asset.is_none());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn load_project_rejects_future_version() {
    let dir = std::env::temp_dir().join("vulvatar_project_future_test");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("future.vvtproj");
    // Minimal future-version body — we only need format_version high
    // enough to trigger the early reject before deserialisation.
    let body = json!({"format_version": PROJECT_FORMAT_VERSION + 99});
    std::fs::write(&path, serde_json::to_string(&body).unwrap()).unwrap();
    let err = load_project(&path).expect_err("future project should reject");
    assert!(
        err.contains("newer VulVATAR"),
        "expected 'newer VulVATAR' message, got: {}",
        err
    );
    let _ = std::fs::remove_file(&path);
}
