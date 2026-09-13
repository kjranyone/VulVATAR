//! Off-screen render-thread driver for the library-thumbnail feature:
//! one PNG per `.vrm` rendered at avatar import (and on-demand
//! re-render). The render thread produces raw RGBA8; this module
//! encodes it to PNG, writes it atomically, and invalidates egui's
//! image cache for the path so the next frame's library inspector
//! picks up the new pixels.
//!
//! **Failure-path contract** for the thumbnail save: when the render
//! thread reports an error, *no on-disk file is touched*. A
//! placeholder PNG written on avatar import stays in place so the
//! library inspector keeps showing *something* instead of going
//! blank on a transient GPU hiccup. The contract is exercised by
//! [`super::thumbnail_failure_tests`].

use eframe::egui;
use log::{info, warn};

use super::{autoframe_aabb, GuiApp};

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
        use crate::renderer::frame_input::*;

        let extent = [
            crate::renderer::thumbnail::THUMBNAIL_WIDTH,
            crate::renderer::thumbnail::THUMBNAIL_HEIGHT,
        ];

        let mesh_instances: Vec<RenderMeshInstance> = avatar
            .asset
            .meshes
            .iter()
            .flat_map(|mesh| {
                mesh.primitives
                    .iter()
                    .map(|prim| RenderMeshInstance::from_primitive(avatar, mesh.id, prim))
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
                cloth_deforms: Vec::new(),
                debug_flags: RenderDebugFlags::default(),
            }],
            output_request: OutputTargetRequest {
                preview_enabled: true,
                output_enabled: true, // force the readback path
                extent,
                color_space: RenderColorSpace::Srgb,
                alpha_mode: RenderOutputAlpha::Premultiplied,
                export_mode: RenderExportMode::CpuReadback,
                msaa: crate::renderer::frame_input::MsaaMode::Off,
            },
            background_image_path: None,
            show_ground_grid: false,
            background_color: [0.0, 0.0, 0.0],
            transparent_background: true,
            avatar_opacity: 1.0,
            // Thumbnails must stay deterministic regardless of the
            // project's bloom / generative-background settings.
            bloom: Default::default(),
            generative_background: Default::default(),
            background_tracking: Default::default(),
            time_seconds: 0.0,
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
        self.library.pending_thumbnail_jobs.push((output_path, rx));
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
        let mut still_pending = Vec::with_capacity(self.library.pending_thumbnail_jobs.len());
        for (path, rx) in std::mem::take(&mut self.library.pending_thumbnail_jobs) {
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
        self.library.pending_thumbnail_jobs = still_pending;
    }
}
