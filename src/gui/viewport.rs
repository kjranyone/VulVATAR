use eframe::egui;

use crate::app::Application;
use crate::asset::Mat4;
use crate::gui::theme::{color, viz};
use crate::gui::GuiApp;
use crate::renderer::debug::{self, DebugDrawList};
use crate::t;

/// Transform a world-space point into clip space through the frame's
/// row-major view/projection pair — the exact transform chain the Vulkan
/// vertex shader uses.
fn world_to_clip(point: [f32; 3], view: &Mat4, proj: &Mat4) -> [f32; 4] {
    let vx = view[0][0] * point[0] + view[0][1] * point[1] + view[0][2] * point[2] + view[0][3];
    let vy = view[1][0] * point[0] + view[1][1] * point[1] + view[1][2] * point[2] + view[1][3];
    let vz = view[2][0] * point[0] + view[2][1] * point[1] + view[2][2] * point[2] + view[2][3];
    [
        proj[0][0] * vx + proj[0][1] * vy + proj[0][2] * vz + proj[0][3],
        proj[1][0] * vx + proj[1][1] * vy + proj[1][2] * vz + proj[1][3],
        proj[2][0] * vx + proj[2][1] * vy + proj[2][2] * vz + proj[2][3],
        proj[3][0] * vx + proj[3][1] * vy + proj[3][2] * vz + proj[3][3],
    ]
}

/// Project a world-space point to 2D pixels through the frame's exact
/// view/projection, mapped into `rect` — the rect the rendered image
/// occupies on screen (letterboxed), not the whole pane. `mirror_x`
/// reproduces the horizontal selfie-flip the display path applies to the
/// texture so the overlay flips with the character.
fn project_point(
    point: [f32; 3],
    view: &Mat4,
    proj: &Mat4,
    rect: &egui::Rect,
    mirror_x: bool,
) -> Option<egui::Pos2> {
    let clip = world_to_clip(point, view, proj);
    if clip[3] <= 0.001 {
        return None;
    }
    let ndc_x = clip[0] / clip[3];
    let ndc_y = clip[1] / clip[3];

    // Vulkan NDC: +x right, +y down, with the y flip already baked into the
    // projection matrix — so ndc→pixels is the plain [0,1] remap into the
    // image rect, matching the GPU's viewport transform.
    let mut px = rect.left() + (ndc_x * 0.5 + 0.5) * rect.width();
    let py = rect.top() + (ndc_y * 0.5 + 0.5) * rect.height();
    if mirror_x {
        px = rect.left() + rect.right() - px;
    }
    Some(egui::pos2(px, py))
}

/// Project a world-space radius at clip depth `clip_w` into screen pixels
/// via the projection's vertical scale (`|proj[1][1]|` — NDC per view unit).
fn project_radius(radius: f32, clip_w: f32, proj_y_scale: f32, viewport_height: f32) -> f32 {
    if clip_w < 0.001 {
        return 0.0;
    }
    (proj_y_scale.abs() * radius / clip_w) * viewport_height * 0.5
}

/// View/projection pair for the CPU debug overlays: the same camera the
/// Vulkan renderer used for the current frame. When the 1:1 sensor mirror is
/// active the renderer swaps to the depth sensor's intrinsics + front view
/// (`Application::render_frame`), so the overlay must swap too — projecting
/// through the orbit camera there puts every debug line in the wrong place.
fn overlay_camera(state: &GuiApp) -> (Mat4, Mat4) {
    if state.mirror_view {
        if let Some(sensor) = state
            .app
            .last_tracking_pose
            .as_ref()
            .and_then(|p| p.metric_frame_info.as_ref())
        {
            let target = state
                .app
                .avatars
                .first()
                .and_then(Application::avatar_upper_body_center)
                .unwrap_or([0.0, 1.0, 0.0]);
            let (view, _) =
                Application::build_sensor_view_matrix(target, sensor.anchor_cam_m[2].abs());
            let proj = Application::build_projection_from_intrinsics(&sensor.intrinsics, 0.1, 10.0);
            return (view, proj);
        }
    }
    let cam = &state.app.viewport_camera;
    let extent = state.app.render_extent();
    let aspect = extent[0] as f32 / extent[1].max(1) as f32;
    let (view, _) = Application::build_view_matrix(cam);
    let proj = Application::build_projection_matrix(cam.fov_deg, aspect, 0.1, 1000.0);
    (view, proj)
}

/// Fit a texture of `tex_size` inside `rect` while preserving aspect ratio
/// (letterbox / pillarbox). Shared by the image display and the debug
/// overlays — the overlays must project into this rect, not the pane rect,
/// or they desync from the character whenever the output aspect differs.
fn letterbox_rect(tex_size: egui::Vec2, rect: egui::Rect) -> egui::Rect {
    let tex_aspect = tex_size.x / tex_size.y.max(1.0);
    let vp_aspect = rect.width() / rect.height().max(1.0);
    let (draw_w, draw_h) = if tex_aspect > vp_aspect {
        // Wider than viewport: fit width.
        (rect.width(), rect.width() / tex_aspect)
    } else {
        // Taller than viewport: fit height.
        (rect.height() * tex_aspect, rect.height())
    };
    egui::Rect::from_center_size(rect.center(), egui::vec2(draw_w, draw_h))
}

/// Draw all debug primitives from a `DebugDrawList` using egui's `Painter`.
fn draw_debug_list(
    painter: &egui::Painter,
    rect: &egui::Rect,
    list: &DebugDrawList,
    view: &Mat4,
    proj: &Mat4,
    mirror_x: bool,
) {
    // Lines.
    for line in &list.lines {
        let p0 = project_point(line.start, view, proj, rect, mirror_x);
        let p1 = project_point(line.end, view, proj, rect, mirror_x);
        if let (Some(a), Some(b)) = (p0, p1) {
            let color = color_f32_to_egui(&line.color);
            painter.line_segment([a, b], egui::Stroke::new(1.0, color));
        }
    }

    // Spheres (drawn as circles).
    for sphere in &list.spheres {
        let center_2d = project_point(sphere.center, view, proj, rect, mirror_x);
        if let Some(c) = center_2d {
            let clip = world_to_clip(sphere.center, view, proj);
            let r = project_radius(sphere.radius, clip[3], proj[1][1], rect.height());
            let r = r.max(2.0); // minimum visible size
            let color = color_f32_to_egui(&sphere.color);
            painter.circle_stroke(c, r, egui::Stroke::new(1.0, color));
        }
    }
}

fn color_f32_to_egui(c: &[f32; 4]) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(
        (c[0].clamp(0.0, 1.0) * 255.0) as u8,
        (c[1].clamp(0.0, 1.0) * 255.0) as u8,
        (c[2].clamp(0.0, 1.0) * 255.0) as u8,
        (c[3].clamp(0.0, 1.0) * 255.0) as u8,
    )
}

pub fn draw(ctx: &egui::Context, state: &mut GuiApp) {
    egui::CentralPanel::default()
        .frame(egui::Frame::none())
        .show(ctx, |ui| {
            let desired = ui.available_size();
            let (rect, response) = ui.allocate_exact_size(desired, egui::Sense::click_and_drag());
            let painter = ui.painter_at(rect);

            // Notify the renderer of the current viewport size so it can
            // match the offscreen render target resolution.
            let vp_w = (rect.width() as u32).max(1);
            let vp_h = (rect.height() as u32).max(1);
            state.app.set_viewport_size(vp_w, vp_h);

            // ── Rendered image display ─────────────────────────────────
            // Check if the Application has produced new rendered pixels
            // and upload / update the egui texture accordingly.
            let has_rendered_image = if let Some((pixels, extent)) = state.app.rendered_pixels() {
                let frame_counter = state.app.rendered_frame_counter();
                let [w, h] = extent;
                if w > 0 && h > 0 && pixels.len() == (w as usize) * (h as usize) * 4 {
                    let color_image =
                        egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], pixels);
                    let options = egui::TextureOptions {
                        magnification: egui::TextureFilter::Linear,
                        minification: egui::TextureFilter::Linear,
                        ..Default::default()
                    };

                    if frame_counter != state.viewport.last_frame {
                        // Update existing texture handle or create a new one.
                        if let Some(ref mut handle) = state.viewport.texture {
                            handle.set(color_image, options);
                        } else {
                            let handle =
                                ui.ctx()
                                    .load_texture("viewport_render", color_image, options);
                            state.viewport.texture = Some(handle);
                        }
                        state.viewport.last_frame = frame_counter;
                    }
                    true
                } else {
                    false
                }
            } else {
                false
            };

            // Rect the rendered frame occupies on screen (letterboxed to the
            // render target's aspect). Computed once here so the debug
            // overlays below can project into the same rect the character is
            // drawn in.
            let image_rect = state
                .viewport
                .texture
                .as_ref()
                .map(|tex| letterbox_rect(tex.size_vec2(), rect));

            if has_rendered_image {
                if let Some(ref tex) = state.viewport.texture {
                    let draw_rect = image_rect.expect("image_rect exists when texture does");

                    // Fill letterbox/pillarbox margins. The preview always
                    // renders at the OUTPUT resolution (WYSIWYG framing for
                    // the virtual camera), so whenever the pane's aspect
                    // differs there are margins — e.g. opening the inspector
                    // narrows the pane and a band appears beside it. Those
                    // margins are chrome around the output picture, not a
                    // rendering void: fill them with the panel surface and
                    // (below) stroke the frame edge, so the boundary reads
                    // as a deliberate mat instead of an unexplained black
                    // gap.
                    painter.rect_filled(rect, 0.0, color::SURFACE_DIM);

                    // When alpha preview is on, draw a checkerboard behind the
                    // rendered image so transparent areas are visible.
                    if state.rendering.alpha_preview {
                        let check = 16.0_f32;
                        let c1 = viz::CHECKER_LIGHT_A;
                        let c2 = viz::CHECKER_LIGHT_B;
                        let cols = ((draw_rect.width() / check).ceil() as usize).min(256);
                        let rows = ((draw_rect.height() / check).ceil() as usize).min(256);
                        for row in 0..rows {
                            for col in 0..cols {
                                let color = if (row + col) % 2 == 0 { c1 } else { c2 };
                                let tile = egui::Rect::from_min_size(
                                    egui::pos2(
                                        draw_rect.left() + col as f32 * check,
                                        draw_rect.top() + row as f32 * check,
                                    ),
                                    egui::vec2(check, check),
                                )
                                .intersect(draw_rect);
                                if tile.is_positive() {
                                    painter.rect_filled(tile, 0.0, color);
                                }
                            }
                        }
                    } else {
                        // Composite against a solid dark background.
                        painter.rect_filled(draw_rect, 0.0, color::VIEWPORT_BG);
                    }

                    // Draw the rendered image (flip horizontally when mirror preview is on).
                    let uv = if state.tracking.tracking_mirror {
                        egui::Rect::from_min_max(egui::pos2(1.0, 0.0), egui::pos2(0.0, 1.0))
                    } else {
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0))
                    };
                    painter.image(tex.id(), draw_rect, uv, egui::Color32::WHITE);

                    // Frame edge on top of the image so the output picture
                    // is explicitly bounded against the margin fill above.
                    painter.rect_stroke(
                        draw_rect,
                        0.0,
                        egui::Stroke::new(1.0, color::OUTLINE_VARIANT),
                    );
                }
            } else {
                // ── Placeholder (no rendered image yet) ────────────────
                if state.rendering.transparent_background {
                    painter.rect_filled(rect, 0.0, color::VIEWPORT_BG);
                    let check = 16.0_f32;
                    let c1 = viz::CHECKER_DARK_A;
                    let c2 = viz::CHECKER_DARK_B;
                    let cols = ((rect.width() / check).ceil() as usize).min(256);
                    let rows = ((rect.height() / check).ceil() as usize).min(256);
                    for row in 0..rows {
                        for col in 0..cols {
                            let color = if (row + col) % 2 == 0 { c1 } else { c2 };
                            let tile = egui::Rect::from_min_size(
                                egui::pos2(
                                    rect.left() + col as f32 * check,
                                    rect.top() + row as f32 * check,
                                ),
                                egui::vec2(check, check),
                            )
                            .intersect(rect);
                            if tile.is_positive() {
                                painter.rect_filled(tile, 0.0, color);
                            }
                        }
                    }
                } else {
                    let [r, g, b] = state.rendering.background_color;
                    painter.rect_filled(
                        rect,
                        0.0,
                        egui::Color32::from_rgb(
                            (r * 255.0) as u8,
                            (g * 255.0) as u8,
                            (b * 255.0) as u8,
                        ),
                    );
                }

                // Grid overlay.
                let grid_color = viz::GRID;
                let spacing = 40.0_f32;

                let mut y = rect.top();
                while y <= rect.bottom() {
                    painter.line_segment(
                        [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
                        egui::Stroke::new(0.5, grid_color),
                    );
                    y += spacing;
                }

                let mut x = rect.left();
                while x <= rect.right() {
                    painter.line_segment(
                        [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                        egui::Stroke::new(0.5, grid_color),
                    );
                    x += spacing;
                }

                // Center crosshair and labels.
                let center = rect.center();
                let cross_color = viz::CROSSHAIR;
                painter.line_segment(
                    [
                        egui::pos2(center.x - 12.0, center.y),
                        egui::pos2(center.x + 12.0, center.y),
                    ],
                    egui::Stroke::new(1.0, cross_color),
                );
                painter.line_segment(
                    [
                        egui::pos2(center.x, center.y - 12.0),
                        egui::pos2(center.x, center.y + 12.0),
                    ],
                    egui::Stroke::new(1.0, cross_color),
                );

                // With an avatar loaded this is a transient "renderer
                // warming up" moment — one muted caption, no chrome.
                // (The old "Viewport / Render target" developer labels
                // said nothing a user could act on.) The no-avatar
                // empty state is drawn below, outside this branch,
                // because an empty scene still produces a rendered
                // texture — gating on "no texture" would never show it.
                if state.app.active_avatar().is_some() {
                    painter.text(
                        egui::pos2(center.x, center.y + 24.0),
                        egui::Align2::CENTER_CENTER,
                        t!("viewport.waiting_render"),
                        crate::gui::theme::typography::body(),
                        viz::LABEL_PRIMARY,
                    );
                }
            }

            // ── Empty state: no avatar loaded ───────────────────────
            // The app's mandatory first step gets a real call to
            // action in the biggest empty space on screen: drop hint
            // plus the two button paths (file picker / library pane).
            if state.app.active_avatar().is_none() {
                egui::Area::new(egui::Id::new("viewport_empty_state"))
                    .order(egui::Order::Middle)
                    .pivot(egui::Align2::CENTER_CENTER)
                    .fixed_pos(rect.center())
                    .show(ui.ctx(), |ui| {
                        ui.vertical_centered(|ui| {
                            ui.label(
                                egui::RichText::new(t!("viewport.empty_headline"))
                                    .font(crate::gui::theme::typography::heading())
                                    .color(viz::OVERLAY_TEXT),
                            );
                            ui.add_space(crate::gui::theme::space::XS);
                            ui.label(
                                egui::RichText::new(t!("viewport.empty_hint"))
                                    .font(crate::gui::theme::typography::body())
                                    .color(viz::LABEL_PRIMARY),
                            );
                            ui.add_space(crate::gui::theme::space::MD);
                            ui.horizontal(|ui| {
                                use crate::gui::components::{
                                    filled_button, tonal_button, ButtonTone,
                                };
                                if filled_button(ui, None, &t!("viewport.empty_open_file"), true)
                                    .clicked()
                                {
                                    crate::gui::top_bar::request_load_avatar_dialog(state);
                                }
                                if tonal_button(
                                    ui,
                                    None,
                                    &t!("viewport.empty_library"),
                                    ButtonTone::Primary,
                                    true,
                                )
                                .clicked()
                                {
                                    state.mode = crate::gui::AppMode::Avatar;
                                    state.inspector_open = true;
                                }
                            });
                        });
                    });
            }

            // Camera interaction: orbit (left-drag), pan (middle-drag/right-drag), zoom (scroll).
            //
            // While any viewport drag is active the cursor is locked and hidden
            // so the user gets Blender-style infinite movement.  Raw mouse
            // deltas come from `pointer.motion()` (DeviceEvent::MouseMotion)
            // which is independent of cursor position.
            let drag_orbit = response.dragged_by(egui::PointerButton::Primary);
            let drag_pan = response.dragged_by(egui::PointerButton::Secondary)
                || response.dragged_by(egui::PointerButton::Middle);
            let any_drag = drag_orbit || drag_pan;

            if any_drag && !state.viewport.cursor_grabbed {
                // First frame of a drag — lock the cursor.
                state.viewport.drag_origin = ctx.input(|i| i.pointer.interact_pos());
                ctx.send_viewport_cmd(egui::ViewportCommand::CursorVisible(false));
                ctx.send_viewport_cmd(egui::ViewportCommand::CursorGrab(
                    egui::viewport::CursorGrab::Locked,
                ));
                state.viewport.cursor_grabbed = true;
            } else if !any_drag && state.viewport.cursor_grabbed {
                // Drag ended — unlock the cursor and warp back to the origin.
                ctx.send_viewport_cmd(egui::ViewportCommand::CursorGrab(
                    egui::viewport::CursorGrab::None,
                ));
                ctx.send_viewport_cmd(egui::ViewportCommand::CursorVisible(true));
                if let Some(origin) = state.viewport.drag_origin.take() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::CursorPosition(origin));
                }
                state.viewport.cursor_grabbed = false;
            }

            // Use raw device motion when the cursor is locked; fall back to
            // the normal drag delta otherwise (first frame of drag, or if the
            // platform doesn't support CursorGrab::Locked).
            let delta = if state.viewport.cursor_grabbed {
                ctx.input(|i| i.pointer.motion())
                    .unwrap_or_else(|| response.drag_delta())
            } else {
                response.drag_delta()
            };

            if drag_orbit {
                state.camera_orbit.yaw_deg += delta.x * state.settings.orbit_sensitivity;
                state.camera_orbit.pitch_deg += delta.y * state.settings.orbit_sensitivity;
            }
            if drag_pan {
                let scale =
                    0.002 * state.camera_orbit.distance * 0.2 * state.settings.pan_sensitivity;
                state.camera_orbit.pan[0] += delta.x * scale;
                state.camera_orbit.pan[1] -= delta.y * scale;
            }
            if response.hovered() {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll.abs() > 0.0 {
                    // Exponential (ratio) zoom: a constant percentage change per
                    // scroll unit, independent of the current distance. The old
                    // formula `target -= scroll * target * sens` was linear in
                    // `scroll`, so one wheel notch (smooth_scroll_delta ≈ 50)
                    // with the default sens=0.1 gave `scroll*sens = 5` → the
                    // multiplier `(1 - 5)` went negative and slammed straight to
                    // the near clamp. `exp(-scroll*sens)` is always positive and
                    // never overshoots, so it stays smooth at any sensitivity.
                    // Clamp `sens` defensively so projects saved under the old
                    // (much larger) scale don't reintroduce the abrupt jump.
                    let sens = state.settings.zoom_sensitivity.clamp(0.0005, 0.01);
                    let factor = (-scroll * sens).exp();
                    state.camera_orbit.target_distance =
                        (state.camera_orbit.target_distance * factor).clamp(0.1, 1000.0);
                }
            }

            // ── Debug drawing (CPU-side fallback) ────────────────────────
            // Project through the exact camera (and into the exact image
            // rect + mirror state) the renderer used this frame, so the
            // overlay lines up with the character instead of living in a
            // different screen mapping.
            let (overlay_view, overlay_proj) = overlay_camera(state);
            let overlay_rect = image_rect.unwrap_or(rect);
            let overlay_mirror = state.tracking.tracking_mirror;

            // Orbit eye position, for the camera-info badge below.
            let yaw = state.camera_orbit.yaw_deg.to_radians();
            let pitch = state.camera_orbit.pitch_deg.to_radians();
            let (sy, cy) = (yaw.sin(), yaw.cos());
            let (sp, cp) = (pitch.sin(), pitch.cos());

            let right = [cy, 0.0, -sy];
            let up = [-sy * sp, cp, -cy * sp];
            let wx = state.camera_orbit.pan[0] * right[0] + state.camera_orbit.pan[1] * up[0];
            let wy = state.camera_orbit.pan[0] * right[1] + state.camera_orbit.pan[1] * up[1];
            let wz = state.camera_orbit.pan[0] * right[2] + state.camera_orbit.pan[1] * up[2];

            let cam_pos = [
                state.camera_orbit.distance * cp * sy + wx,
                state.camera_orbit.distance * sp + wy,
                state.camera_orbit.distance * cp * cy + wz,
            ];

            if let Some(avatar) = state.app.active_avatar() {
                // Skeleton debug.
                if state.rendering.toggle_skeleton_debug {
                    let skel = &avatar.asset.skeleton;
                    // Use skinning matrices from pose if available; otherwise
                    // fall back to identity per node.
                    let pose_matrices: Vec<crate::asset::Mat4> =
                        if !avatar.pose.global_transforms.is_empty() {
                            avatar.pose.global_transforms.clone()
                        } else {
                            vec![crate::asset::identity_matrix(); skel.nodes.len()]
                        };
                    let skel_list = debug::build_skeleton_debug(skel, &pose_matrices);
                    draw_debug_list(
                        &painter,
                        &overlay_rect,
                        &skel_list,
                        &overlay_view,
                        &overlay_proj,
                        overlay_mirror,
                    );
                }

                // Collider debug.
                if state.rendering.toggle_collision_debug {
                    let skel = &avatar.asset.skeleton;
                    let pose_matrices: Vec<crate::asset::Mat4> =
                        if !avatar.pose.global_transforms.is_empty() {
                            avatar.pose.global_transforms.clone()
                        } else {
                            vec![crate::asset::identity_matrix(); skel.nodes.len()]
                        };
                    let col_list =
                        debug::build_collider_debug(&avatar.asset.colliders, skel, &pose_matrices);
                    draw_debug_list(
                        &painter,
                        &overlay_rect,
                        &col_list,
                        &overlay_view,
                        &overlay_proj,
                        overlay_mirror,
                    );
                }

                // Cloth mesh debug.
                if state.rendering.toggle_cloth {
                    let cloth_states: Vec<_> = avatar
                        .cloth_state
                        .iter()
                        .chain(avatar.cloth_overlays.iter().map(|s| &s.state))
                        .collect();
                    for cloth_state in cloth_states {
                        if let Some(ref overlay) = state.app.editor.overlay_asset {
                            let cloth_list = debug::build_cloth_mesh_debug(
                                &overlay.simulation_mesh,
                                &cloth_state.sim_positions,
                            );
                            draw_debug_list(
                                &painter,
                                &overlay_rect,
                                &cloth_list,
                                &overlay_view,
                                &overlay_proj,
                                overlay_mirror,
                            );

                            if !cloth_state.sim_normals.is_empty() {
                                let positions = if !cloth_state.sim_positions.is_empty() {
                                    &cloth_state.sim_positions
                                } else {
                                    &vec![]
                                };
                                if !positions.is_empty() {
                                    let normal_list = debug::build_normal_debug(
                                        positions,
                                        &cloth_state.sim_normals,
                                    );
                                    draw_debug_list(
                                        &painter,
                                        &overlay_rect,
                                        &normal_list,
                                        &overlay_view,
                                        &overlay_proj,
                                        overlay_mirror,
                                    );
                                }
                            }
                        }
                    }
                }
            }

            // ── Camera PIP wipe ───────────────────────────────────────
            if state.viewport.show_camera_wipe {
                // Cheap gate first: only pull a full snapshot (which
                // refcounts the frame + clones pose/annotation) when the
                // preview mailbox actually advanced. Dedup on
                // `preview_sequence` (the preview-mailbox counter) not
                // `sequence` (pose): using pose here would cause "torn
                // snapshot" races to advance `camera_wipe_seq` while
                // still uploading the old frame, leaving the newer
                // frame permanently un-uploaded until the next publish.
                if state.app.tracking.mailbox().preview_sequence() != state.viewport.camera_wipe_seq
                {
                    let snap = state.app.tracking.mailbox().snapshot();
                    if let Some(ref frame) = snap.frame {
                        let w = frame.width as usize;
                        let h = frame.height as usize;
                        if w > 0 && h > 0 && frame.rgb_data.len() == w * h * 3 {
                            let needed = w * h * 4;
                            if state.viewport.camera_wipe_rgba_buf.len() != needed {
                                state.viewport.camera_wipe_rgba_buf.resize(needed, 255);
                            }
                            let rgba = &mut state.viewport.camera_wipe_rgba_buf;
                            for i in 0..w * h {
                                rgba[i * 4] = frame.rgb_data[i * 3];
                                rgba[i * 4 + 1] = frame.rgb_data[i * 3 + 1];
                                rgba[i * 4 + 2] = frame.rgb_data[i * 3 + 2];
                                rgba[i * 4 + 3] = 255;
                            }
                            let color_image =
                                egui::ColorImage::from_rgba_unmultiplied([w, h], rgba);
                            let options = egui::TextureOptions {
                                magnification: egui::TextureFilter::Linear,
                                minification: egui::TextureFilter::Linear,
                                ..Default::default()
                            };
                            if let Some(ref mut handle) = state.viewport.camera_wipe_texture {
                                handle.set(color_image, options);
                            } else {
                                let handle =
                                    ui.ctx().load_texture("camera_wipe", color_image, options);
                                state.viewport.camera_wipe_texture = Some(handle);
                            }
                        }
                    }
                    // Annotation rides the same freshness gate; kept on
                    // viewport state so the frames *between* publishes
                    // redraw the last overlay without re-snapshotting.
                    state.viewport.camera_wipe_annotation = snap.annotation;
                    state.viewport.camera_wipe_seq = snap.preview_sequence;
                }

                {
                    if let Some(ref tex) = state.viewport.camera_wipe_texture {
                        let pip_max_w = 240.0;
                        let tex_size = tex.size_vec2();
                        let scale = pip_max_w / tex_size.x.max(1.0);
                        let pip_w = tex_size.x * scale;
                        let pip_h = tex_size.y * scale;
                        let pip_rect = egui::Rect::from_min_size(
                            egui::pos2(rect.right() - pip_w - 12.0, rect.bottom() - pip_h - 12.0),
                            egui::vec2(pip_w, pip_h),
                        );
                        painter.rect_filled(pip_rect, 4.0, color::VIEWPORT_BG);
                        painter.rect_stroke(
                            pip_rect,
                            4.0,
                            egui::Stroke::new(2.0, color::VIEWPORT_OVERLAY_OUTLINE),
                        );

                        let uv = if state.tracking.tracking_mirror {
                            egui::Rect::from_min_max(egui::pos2(1.0, 0.0), egui::pos2(0.0, 1.0))
                        } else {
                            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0))
                        };
                        painter.image(tex.id(), pip_rect, uv, egui::Color32::WHITE);

                        // ── Detection annotations ──────────────────────
                        if state.viewport.show_detection_annotations {
                            if let Some(ref ann) = state.viewport.camera_wipe_annotation {
                                let kpt_color = viz::keypoint();
                                let line_color = viz::bone();
                                let bb_color = viz::bbox();

                                let mirror = state.tracking.tracking_mirror;
                                let map_x = |nx: f32| {
                                    let x = if mirror { 1.0 - nx } else { nx };
                                    pip_rect.left() + x * pip_rect.width()
                                };
                                let map_y = |ny: f32| pip_rect.top() + ny * pip_rect.height();

                                for &(kx, ky, conf) in &ann.keypoints {
                                    if conf < 0.1 {
                                        continue;
                                    }
                                    painter.circle_filled(
                                        egui::pos2(map_x(kx), map_y(ky)),
                                        3.0,
                                        kpt_color,
                                    );
                                }

                                for &(a, b) in &ann.skeleton {
                                    let ka = ann.keypoints.get(a);
                                    let kb = ann.keypoints.get(b);
                                    if let (Some(&(ax, ay, ac)), Some(&(bx, by, bc))) = (ka, kb) {
                                        if ac >= 0.1 && bc >= 0.1 {
                                            painter.line_segment(
                                                [
                                                    egui::pos2(map_x(ax), map_y(ay)),
                                                    egui::pos2(map_x(bx), map_y(by)),
                                                ],
                                                egui::Stroke::new(1.5, line_color),
                                            );
                                        }
                                    }
                                }

                                if let Some((bx1, by1, bx2, by2)) = ann.bounding_box {
                                    let px1 = map_x(bx1);
                                    let py1 = map_y(by1);
                                    let px2 = map_x(bx2);
                                    let py2 = map_y(by2);
                                    let bb_rect = egui::Rect::from_min_max(
                                        egui::pos2(px1.min(px2), py1.min(py2)),
                                        egui::pos2(px1.max(px2), py1.max(py2)),
                                    );
                                    painter.rect_stroke(
                                        bb_rect,
                                        0.0,
                                        egui::Stroke::new(1.5, bb_color),
                                    );
                                }
                            }
                        }
                    }
                }
            }

            // Show camera info overlay. Drawn on a dark semi-transparent badge
            // with a bright foreground so it stays legible over both light and
            // dark rendered content — the old flat dim-grey text was nearly
            // invisible against the viewport background.
            let info = t!(
                "viewport.camera_info",
                x = format!("{:.2}", cam_pos[0]),
                y = format!("{:.2}", cam_pos[1]),
                z = format!("{:.2}", cam_pos[2]),
                yaw = format!("{:.1}", state.camera_orbit.yaw_deg),
                pitch = format!("{:.1}", state.camera_orbit.pitch_deg),
                dist = format!("{:.2}", state.camera_orbit.distance)
            );
            let info_color = viz::OVERLAY_TEXT;
            let galley = painter.layout_no_wrap(info, egui::FontId::monospace(11.0), info_color);
            let pad = egui::vec2(6.0, 4.0);
            let text_tl = egui::pos2(rect.left() + 10.0, rect.bottom() - 10.0 - galley.size().y);
            painter.rect_filled(
                egui::Rect::from_min_size(text_tl - pad, galley.size() + pad * 2.0),
                4.0,
                viz::overlay_badge_bg(),
            );
            painter.galley(text_tl, galley, info_color);

            // T04: Draw cloth region selection overlay in ClothAuthoring mode.
            if state.mode == crate::gui::AppMode::ClothAuthoring {
                if let Some(avatar) = state.app.active_avatar() {
                    let sel_ref = state.cloth_authoring.region_selection.as_ref();
                    match sel_ref {
                        Some(sel) => {
                            let prim_label = avatar
                                .asset
                                .meshes
                                .iter()
                                .flat_map(|m| m.primitives.iter())
                                .find(|p| p.id == sel.target_primitive)
                                .map(|p| t!("viewport.primitive", id = p.id.0))
                                .unwrap_or_else(|| {
                                    t!("viewport.primitive", id = sel.target_primitive.0)
                                });

                            let vert_count = sel.selected_vertices.len();
                            let lines = [
                                t!("viewport.selected", count = vert_count),
                                t!("viewport.target", label = prim_label),
                                t!(
                                    "viewport.range",
                                    start = sel.selected_vertex_range.0,
                                    end = sel.selected_vertex_range.1
                                ),
                            ];

                            let mut y = rect.top() + 6.0;
                            for line in &lines {
                                painter.text(
                                    egui::pos2(rect.left() + 8.0, y),
                                    egui::Align2::LEFT_TOP,
                                    line,
                                    egui::FontId::monospace(12.0),
                                    viz::selection_text(),
                                );
                                y += 16.0;
                            }
                        }
                        None => {
                            painter.text(
                                egui::pos2(rect.left() + 8.0, rect.top() + 6.0),
                                egui::Align2::LEFT_TOP,
                                t!("viewport.no_region"),
                                egui::FontId::monospace(12.0),
                                viz::muted_overlay_text(),
                            );
                        }
                    }
                }
            }
        });
}
