//! Off-screen render-thread driver for two `GuiApp` features:
//!
//! 1. **Library thumbnails** — one PNG per `.vrm` rendered at avatar
//!    import (and on-demand re-render). The render thread produces
//!    raw RGBA8; this module encodes it to PNG, writes it atomically,
//!    and invalidates egui's image cache for the path so the next
//!    frame's library inspector picks up the new pixels.
//! 2. **Calibration target-pose snapshot** — one render of the active
//!    avatar in the calibration mode's reference pose (T-pose for
//!    FullBody, hands-at-sides for UpperBody) uploaded to an egui
//!    texture, which the calibration modal draws alongside the
//!    webcam preview pane so the user can mimic it.
//!
//! Both paths share the render-thread `request_thumbnail` machinery
//! and the GuiApp-side per-frame poll dispatch — keeping them
//! together makes that shared structure obvious.
//!
//! **Failure-path contract** for the thumbnail save: when the render
//! thread reports an error, *no on-disk file is touched*. A
//! placeholder PNG written on avatar import stays in place so the
//! library inspector keeps showing *something* instead of going
//! blank on a transient GPU hiccup. The contract is exercised by
//! [`super::thumbnail_failure_tests`].

use eframe::egui;
use log::{info, warn};

use super::{autoframe_aabb, calibration, GuiApp};

/// Encode a [`crate::renderer::ThumbnailRenderResult`] (raw RGBA8 +
/// extent) as PNG at `path`. Creates the parent directory if needed.
pub(super) fn save_thumbnail_png(
    path: &std::path::Path,
    thumb: &crate::renderer::ThumbnailRenderResult,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create thumbnail dir '{}': {}", parent.display(), e))?;
    }
    let img = image::RgbaImage::from_raw(thumb.width, thumb.height, thumb.rgba_pixels.clone())
        .ok_or_else(|| "thumbnail RGBA buffer length didn't match width*height*4".to_string())?;
    img.save(path)
        .map_err(|e| format!("save thumbnail PNG '{}': {}", path.display(), e))?;
    Ok(())
}

/// Decide what to do with a single thumbnail render response,
/// separated from [`GuiApp::poll_thumbnail_jobs`] so the failure-path
/// contract is testable without an `egui::Context`.
///
/// Returns `Ok(())` iff a fresh PNG was successfully written — only
/// then should the caller invalidate egui's image cache for that
/// path.
///
/// **Failure-path contract**: if the render thread reported an error,
/// or the encode step itself fails, **no on-disk file is touched**.
/// A pre-existing placeholder PNG (written eagerly on avatar import)
/// stays in place so the library inspector keeps showing *something*
/// instead of going blank on a transient GPU hiccup. The caller must
/// not, in either branch, replace or unlink the file outside this
/// helper.
pub(super) fn handle_thumbnail_response(
    path: &std::path::Path,
    response: Result<crate::renderer::ThumbnailRenderResult, String>,
) -> Result<(), String> {
    match response {
        Ok(thumb) => match save_thumbnail_png(path, &thumb) {
            Ok(()) => {
                info!("thumbnail: wrote real-render PNG to {}", path.display());
                Ok(())
            }
            Err(e) => {
                warn!(
                    "thumbnail: failed to write '{}': {} (keeping any existing placeholder)",
                    path.display(),
                    e
                );
                Err(e)
            }
        },
        Err(e) => {
            warn!(
                "thumbnail: render failed for '{}': {} (keeping any existing placeholder)",
                path.display(),
                e
            );
            Err(e)
        }
    }
}

impl GuiApp {
    /// Build a `RenderFrameInput` for a one-shot thumbnail render of
    /// `avatar`. Camera is auto-framed to the avatar's bounding box,
    /// lighting is a fixed studio default (so all entries compare
    /// fairly), background is transparent (PNG with alpha), and the
    /// pose is whatever the avatar already has. Output extent matches
    /// `crate::renderer::thumbnail::THUMBNAIL_*`.
    fn build_thumbnail_frame_input(
        &self,
        avatar: &crate::avatar::AvatarInstance,
    ) -> crate::renderer::frame_input::RenderFrameInput {
        use crate::asset::AlphaMode;
        use crate::renderer::frame_input::*;
        use crate::renderer::material::MaterialUploadRequest;
        use std::sync::Arc;

        let extent = [
            crate::renderer::thumbnail::THUMBNAIL_WIDTH,
            crate::renderer::thumbnail::THUMBNAIL_HEIGHT,
        ];

        let mesh_instances: Vec<RenderMeshInstance> = avatar
            .asset
            .meshes
            .iter()
            .flat_map(|mesh| {
                mesh.primitives.iter().map(move |prim| {
                    let material_asset = avatar
                        .asset
                        .materials
                        .iter()
                        .find(|m| m.id == prim.material_id);
                    let material_binding = material_asset
                        .map(MaterialUploadRequest::from_asset_material)
                        .unwrap_or_else(MaterialUploadRequest::default_material);
                    let outline = OutlineSnapshot {
                        enabled: material_binding.outline_width > 0.0,
                        width: material_binding.outline_width,
                        color: material_binding.outline_color,
                    };
                    let alpha_mode = match material_binding.alpha_mode {
                        AlphaMode::Opaque => RenderAlphaMode::Opaque,
                        AlphaMode::Mask(_) => RenderAlphaMode::Cutout,
                        AlphaMode::Blend => RenderAlphaMode::Blend,
                    };
                    let cull_mode = if material_binding.double_sided {
                        RenderCullMode::DoubleSided
                    } else {
                        RenderCullMode::BackFace
                    };
                    RenderMeshInstance {
                        mesh_id: mesh.id,
                        primitive_id: prim.id,
                        material_binding,
                        bounds: prim.bounds,
                        alpha_mode,
                        cull_mode,
                        outline,
                        primitive_data: Some(Arc::new(prim.clone())),
                        morph_weights: Vec::new(),
                    }
                })
            })
            .collect();

        // Auto-frame the avatar from the front (yaw=0, pitch=0) so all
        // entries share a comparable view.
        let aspect = extent[0] as f32 / extent[1].max(1) as f32;
        let fov_deg: f32 = 30.0;
        let (pan_y, distance) = autoframe_aabb(&avatar.asset.root_aabb, fov_deg, aspect);
        let cam = crate::app::ViewportCamera {
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            pan: [0.0, pan_y],
            distance,
            fov_deg,
        };
        let (view, eye_pos) = crate::app::Application::build_view_matrix(&cam);
        let projection =
            crate::app::Application::build_projection_matrix(fov_deg, aspect, 0.1, 1000.0);

        RenderFrameInput {
            camera: CameraState {
                view,
                projection,
                position_ws: eye_pos,
                viewport_extent: extent,
            },
            // Studio lighting: hard-coded so thumbnails are visually
            // consistent across avatars regardless of the user's current
            // scene lighting (which may be tuned for streaming, not
            // library browsing).
            lighting: LightingState {
                main_light_dir_ws: [-0.3, -0.7, -0.5],
                main_light_color: [1.0, 1.0, 1.0],
                main_light_intensity: 1.0,
                ambient_term: [0.3, 0.3, 0.3],
            },
            instances: vec![RenderAvatarInstance {
                instance_id: avatar.id,
                world_transform: avatar.world_transform.clone(),
                mesh_instances,
                skinning_matrices: avatar.pose.skinning_matrices.clone(),
                cloth_deform: None,
                debug_flags: RenderDebugFlags::default(),
            }],
            output_request: OutputTargetRequest {
                preview_enabled: true,
                output_enabled: true, // force the readback path
                extent,
                color_space: RenderColorSpace::Srgb,
                alpha_mode: RenderOutputAlpha::Premultiplied,
                export_mode: RenderExportMode::CpuReadback,
            },
            background_image_path: None,
            show_ground_grid: false,
            background_color: [0.0, 0.0, 0.0],
            transparent_background: true,
        }
    }

    /// Build a `RenderFrameInput` that snapshots the active avatar
    /// in the calibration target pose (T-pose for FullBody,
    /// hands-at-sides for UpperBody). Output extent is 240x270 so
    /// the rendered image lines up vertically with the modal's
    /// webcam preview pane.
    fn build_calibration_target_pose_frame_input(
        &self,
        avatar: &crate::avatar::AvatarInstance,
        mode: crate::tracking::CalibrationMode,
    ) -> crate::renderer::frame_input::RenderFrameInput {
        use crate::asset::AlphaMode;
        use crate::renderer::frame_input::*;
        use crate::renderer::material::MaterialUploadRequest;
        use std::sync::Arc;

        // Match the modal's webcam preview height (270 px) and pick a
        // width that fits a head-to-hips framing for both T-pose and
        // hands-at-sides without empty space. 240 keeps the modal
        // total width reasonable when the target snapshot is shown
        // alongside the 480-wide webcam.
        let extent: [u32; 2] = [240, 270];

        let mesh_instances: Vec<RenderMeshInstance> = avatar
            .asset
            .meshes
            .iter()
            .flat_map(|mesh| {
                mesh.primitives.iter().map(move |prim| {
                    let material_asset = avatar
                        .asset
                        .materials
                        .iter()
                        .find(|m| m.id == prim.material_id);
                    let material_binding = material_asset
                        .map(MaterialUploadRequest::from_asset_material)
                        .unwrap_or_else(MaterialUploadRequest::default_material);
                    let outline = OutlineSnapshot {
                        enabled: material_binding.outline_width > 0.0,
                        width: material_binding.outline_width,
                        color: material_binding.outline_color,
                    };
                    let alpha_mode = match material_binding.alpha_mode {
                        AlphaMode::Opaque => RenderAlphaMode::Opaque,
                        AlphaMode::Mask(_) => RenderAlphaMode::Cutout,
                        AlphaMode::Blend => RenderAlphaMode::Blend,
                    };
                    let cull_mode = if material_binding.double_sided {
                        RenderCullMode::DoubleSided
                    } else {
                        RenderCullMode::BackFace
                    };
                    RenderMeshInstance {
                        mesh_id: mesh.id,
                        primitive_id: prim.id,
                        material_binding,
                        bounds: prim.bounds,
                        alpha_mode,
                        cull_mode,
                        outline,
                        primitive_data: Some(Arc::new(prim.clone())),
                        morph_weights: Vec::new(),
                    }
                })
            })
            .collect();

        // Compute target-pose skinning matrices from the asset's
        // skeleton + bind pose without mutating the live avatar
        // instance — the live render keeps using the tracked pose
        // while the snapshot captures the reference pose.
        let skinning_matrices =
            calibration::target_pose_skinning_matrices(&avatar.asset, mode);

        let aspect = extent[0] as f32 / extent[1].max(1) as f32;
        let fov_deg: f32 = 30.0;
        let (pan_y, distance) = autoframe_aabb(&avatar.asset.root_aabb, fov_deg, aspect);
        let cam = crate::app::ViewportCamera {
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            pan: [0.0, pan_y],
            distance,
            fov_deg,
        };
        let (view, eye_pos) = crate::app::Application::build_view_matrix(&cam);
        let projection =
            crate::app::Application::build_projection_matrix(fov_deg, aspect, 0.1, 1000.0);

        RenderFrameInput {
            camera: CameraState {
                view,
                projection,
                position_ws: eye_pos,
                viewport_extent: extent,
            },
            // Studio lighting: hard-coded so the reference snapshot is
            // visually consistent regardless of the user's current
            // scene lighting (which may be tuned for streaming).
            lighting: LightingState {
                main_light_dir_ws: [-0.3, -0.7, -0.5],
                main_light_color: [1.0, 1.0, 1.0],
                main_light_intensity: 1.0,
                ambient_term: [0.3, 0.3, 0.3],
            },
            instances: vec![RenderAvatarInstance {
                instance_id: avatar.id,
                world_transform: avatar.world_transform.clone(),
                mesh_instances,
                skinning_matrices,
                cloth_deform: None,
                debug_flags: RenderDebugFlags::default(),
            }],
            output_request: OutputTargetRequest {
                preview_enabled: true,
                output_enabled: true,
                extent,
                color_space: RenderColorSpace::Srgb,
                alpha_mode: RenderOutputAlpha::Premultiplied,
                export_mode: RenderExportMode::CpuReadback,
            },
            background_image_path: None,
            show_ground_grid: false,
            background_color: [0.0, 0.0, 0.0],
            transparent_background: true,
        }
    }

    /// Kick off a one-shot render of the avatar in the calibration
    /// target pose. Idempotent: if a request for the same mode is
    /// already in flight, this does nothing. The poll path
    /// ([`Self::poll_calibration_target_pose_snapshot`]) drains the
    /// receiver and uploads to `calibration_target_pose_texture`.
    pub fn kick_calibration_target_pose_snapshot(
        &mut self,
        mode: crate::tracking::CalibrationMode,
    ) {
        // Already have an in-flight render for this mode — don't
        // double up. Pending for a different mode is also fine to
        // overwrite (the user changed their mind).
        if self.calibration_target_pose_pending.is_some()
            && self.calibration_target_pose_mode == Some(mode)
        {
            return;
        }
        // Already have the snapshot for this mode and no pending
        // request — nothing to do.
        if self.calibration_target_pose_pending.is_none()
            && self.calibration_target_pose_mode == Some(mode)
            && self.calibration_target_pose_texture.is_some()
        {
            return;
        }
        let Some(avatar) = self.app.active_avatar() else {
            return;
        };
        let input = self.build_calibration_target_pose_frame_input(avatar, mode);
        let Some(rt) = self.app.render_thread.as_ref() else {
            return;
        };
        let rx = rt.request_thumbnail(input);
        self.calibration_target_pose_pending = Some(rx);
        self.calibration_target_pose_mode = Some(mode);
    }

    /// Drain the pending target-pose snapshot, if any, and upload
    /// the resulting RGBA pixels to
    /// `calibration_target_pose_texture`. Called every frame from
    /// the modal's `draw_modal` entry while the modal is open.
    pub fn poll_calibration_target_pose_snapshot(&mut self, ctx: &egui::Context) {
        let Some(rx) = self.calibration_target_pose_pending.take() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(thumb)) => {
                let w = thumb.width as usize;
                let h = thumb.height as usize;
                if w > 0 && h > 0 && thumb.rgba_pixels.len() == w * h * 4 {
                    let color_image = egui::ColorImage::from_rgba_unmultiplied(
                        [w, h],
                        &thumb.rgba_pixels,
                    );
                    let options = egui::TextureOptions {
                        magnification: egui::TextureFilter::Linear,
                        minification: egui::TextureFilter::Linear,
                        ..Default::default()
                    };
                    if let Some(ref mut handle) = self.calibration_target_pose_texture {
                        handle.set(color_image, options);
                    } else {
                        let handle = ctx.load_texture(
                            "calibration_target_pose",
                            color_image,
                            options,
                        );
                        self.calibration_target_pose_texture = Some(handle);
                    }
                }
            }
            Ok(Err(e)) => {
                log::warn!(
                    "calibration target-pose render failed: {}; falling back to no-snapshot view",
                    e
                );
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                // Still rendering — put the receiver back.
                self.calibration_target_pose_pending = Some(rx);
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                log::warn!(
                    "calibration target-pose render channel disconnected; \
                     dropping snapshot request"
                );
            }
        }
    }

    /// Submit a thumbnail render request to the render thread for the
    /// active avatar and queue the receiver for later polling. No-op
    /// when there's no render thread or no active avatar.
    pub fn kick_thumbnail_job(&mut self, output_path: std::path::PathBuf) {
        let Some(avatar) = self.app.active_avatar() else {
            return;
        };
        // Snapshot the frame input here while we still have an immutable
        // borrow of `self.app`.
        let input = self.build_thumbnail_frame_input(avatar);
        let Some(rt) = self.app.render_thread.as_ref() else {
            return;
        };
        let rx = rt.request_thumbnail(input);
        self.pending_thumbnail_jobs.push((output_path, rx));
    }

    /// Drain any thumbnail jobs whose render thread response has
    /// arrived. On success the destination PNG is overwritten and
    /// egui's `file://` cache for that path is invalidated so the new
    /// pixels appear in the library inspector on the next frame. On
    /// failure (render error, write error, or channel disconnect) the
    /// existing on-disk PNG — typically a placeholder written on
    /// avatar import — is left untouched. See
    /// [`handle_thumbnail_response`] for the exact contract.
    pub(super) fn poll_thumbnail_jobs(&mut self, ctx: &egui::Context) {
        let mut still_pending = Vec::with_capacity(self.pending_thumbnail_jobs.len());
        for (path, rx) in std::mem::take(&mut self.pending_thumbnail_jobs) {
            match rx.try_recv() {
                Ok(response) => {
                    if handle_thumbnail_response(&path, response).is_ok() {
                        // Cache invalidation is gated on a successful
                        // write only — on failure we keep the existing
                        // placeholder texture so the inspector doesn't
                        // flash to nothing.
                        ctx.forget_image(&format!("file://{}", path.display()));
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    still_pending.push((path, rx));
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    warn!(
                        "thumbnail: response channel disconnected for '{}' (keeping any existing placeholder)",
                        path.display()
                    );
                }
            }
        }
        self.pending_thumbnail_jobs = still_pending;
    }
}
