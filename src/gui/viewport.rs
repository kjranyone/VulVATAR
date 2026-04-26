use eframe::egui;

use crate::gui::GuiApp;
use crate::t;
use crate::renderer::debug::{self, DebugDrawList};

/// Simple projection helper that maps a 3D world-space point to 2D viewport
/// coordinates using the current camera state stored in `GuiApp`.
///
/// This is a CPU-side fallback for when the Vulkan renderer is not fully
/// integrated.  It builds a basic perspective projection from the GUI camera
/// parameters (position, rotation, FOV) and maps the result into the supplied
/// egui `Rect`.
fn project_point(
    point: [f32; 3],
    cam_pos: [f32; 3],
    cam_rot: [f32; 3], // euler degrees [pitch, yaw, roll]
    fov_deg: f32,
    rect: &egui::Rect,
) -> Option<egui::Pos2> {
    // Translate into camera-relative space.
    let dx = point[0] - cam_pos[0];
    let dy = point[1] - cam_pos[1];
    let dz = point[2] - cam_pos[2];

    // Apply inverse camera rotation (yaw then pitch, ignoring roll for
    // simplicity).
    let yaw = cam_rot[1].to_radians();
    let pitch = cam_rot[0].to_radians();

    let (sy, cy) = (yaw.sin(), yaw.cos());
    let x1 = dx * cy - dz * sy;
    let z1 = dx * sy + dz * cy;

    let (sp, cp) = (pitch.sin(), pitch.cos());
    let y2 = dy * cp - z1 * sp;
    let z2 = -(dy * sp + z1 * cp);

    if z2 < 0.001 {
        return None;
    }

    let aspect = rect.width() / rect.height().max(1.0);
    let half_fov_rad = (fov_deg * 0.5).to_radians();
    let f = 1.0 / half_fov_rad.tan();

    let ndc_x = (f * x1) / (z2 * aspect);
    let ndc_y = (f * y2) / z2;

    // NDC (-1..1) to viewport pixel coordinates.  Y is flipped (screen Y grows
    // downward).
    let px = rect.center().x + ndc_x * rect.width() * 0.5;
    let py = rect.center().y - ndc_y * rect.height() * 0.5;

    Some(egui::pos2(px, py))
}

/// Project a world-space radius at a given depth into screen pixels.
fn project_radius(radius: f32, depth: f32, fov_deg: f32, viewport_height: f32) -> f32 {
    if depth < 0.001 {
        return 0.0;
    }
    let half_fov_rad = (fov_deg * 0.5).to_radians();
    let f = 1.0 / half_fov_rad.tan();
    (f * radius / depth) * viewport_height * 0.5
}

/// Approximate depth from camera to a point (simplified, uses the same
/// transform as `project_point`).
fn point_depth(point: [f32; 3], cam_pos: [f32; 3], cam_rot: [f32; 3]) -> f32 {
    let dx = point[0] - cam_pos[0];
    let dy = point[1] - cam_pos[1];
    let dz = point[2] - cam_pos[2];

    let yaw = cam_rot[1].to_radians();
    let pitch = cam_rot[0].to_radians();

    let (sy, cy) = (yaw.sin(), yaw.cos());
    let z1 = dx * sy + dz * cy;

    let (sp, cp) = (pitch.sin(), pitch.cos());
    -(dy * sp + z1 * cp)
}

/// Draw all debug primitives from a `DebugDrawList` using egui's `Painter`.
fn draw_debug_list(
    painter: &egui::Painter,
    rect: &egui::Rect,
    list: &DebugDrawList,
    cam_pos: [f32; 3],
    cam_rot: [f32; 3],
    fov_deg: f32,
) {
    // Lines.
    for line in &list.lines {
        let p0 = project_point(line.start, cam_pos, cam_rot, fov_deg, rect);
        let p1 = project_point(line.end, cam_pos, cam_rot, fov_deg, rect);
        if let (Some(a), Some(b)) = (p0, p1) {
            let color = color_f32_to_egui(&line.color);
            painter.line_segment([a, b], egui::Stroke::new(1.0, color));
        }
    }

    // Spheres (drawn as circles).
    for sphere in &list.spheres {
        let center_2d = project_point(sphere.center, cam_pos, cam_rot, fov_deg, rect);
        if let Some(c) = center_2d {
            let depth = point_depth(sphere.center, cam_pos, cam_rot);
            let r = project_radius(sphere.radius, depth, fov_deg, rect.height());
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

                    if frame_counter != state.viewport_last_frame {
                        // Update existing texture handle or create a new one.
                        if let Some(ref mut handle) = state.viewport_texture {
                            handle.set(color_image, options);
                        } else {
                            let handle =
                                ui.ctx()
                                    .load_texture("viewport_render", color_image, options);
                            state.viewport_texture = Some(handle);
                        }
                        state.viewport_last_frame = frame_counter;
                    }
                    true
                } else {
                    false
                }
            } else {
                false
            };

            if has_rendered_image {
                if let Some(ref tex) = state.viewport_texture {
                    // Scale the rendered image to fit the viewport while
                    // preserving aspect ratio.
                    let tex_size = tex.size_vec2();
                    let tex_aspect = tex_size.x / tex_size.y.max(1.0);
                    let vp_aspect = rect.width() / rect.height().max(1.0);

                    let (draw_w, draw_h) = if tex_aspect > vp_aspect {
                        // Wider than viewport: fit width.
                        (rect.width(), rect.width() / tex_aspect)
                    } else {
                        // Taller than viewport: fit height.
                        (rect.height() * tex_aspect, rect.height())
                    };

                    let draw_rect =
                        egui::Rect::from_center_size(rect.center(), egui::vec2(draw_w, draw_h));

                    // Fill letterbox/pillarbox area with background.
                    painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(18, 18, 22));

                    // When alpha preview is on, draw a checkerboard behind the
                    // rendered image so transparent areas are visible.
                    if state.rendering.alpha_preview {
                        let check = 16.0_f32;
                        let c1 = egui::Color32::from_rgb(180, 180, 180);
                        let c2 = egui::Color32::from_rgb(220, 220, 220);
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
                        painter.rect_filled(draw_rect, 0.0, egui::Color32::from_rgb(18, 18, 22));
                    }

                    // Draw the rendered image (flip horizontally when mirror preview is on).
                    let uv = if state.tracking.tracking_mirror {
                        egui::Rect::from_min_max(egui::pos2(1.0, 0.0), egui::pos2(0.0, 1.0))
                    } else {
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0))
                    };
                    painter.image(tex.id(), draw_rect, uv, egui::Color32::WHITE);
                }
            } else {
                // ── Placeholder (no rendered image yet) ────────────────
                if state.rendering.transparent_background {
                    painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(25, 25, 30));
                    let check = 16.0_f32;
                    let c1 = egui::Color32::from_rgb(30, 30, 35);
                    let c2 = egui::Color32::from_rgb(40, 40, 45);
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
                let grid_color = egui::Color32::from_rgb(45, 45, 55);
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
                let cross_color = egui::Color32::from_rgb(70, 70, 85);
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

                painter.text(
                    egui::pos2(center.x, center.y - 20.0),
                    egui::Align2::CENTER_CENTER,
                    t!("viewport.viewport"),
                    egui::FontId::proportional(20.0),
                    egui::Color32::from_rgb(100, 100, 115),
                );
                painter.text(
                    egui::pos2(center.x, center.y + 20.0),
                    egui::Align2::CENTER_CENTER,
                    t!("viewport.render_target"),
                    egui::FontId::proportional(13.0),
                    egui::Color32::from_rgb(75, 75, 85),
                );
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

            if any_drag && !state.viewport_cursor_grabbed {
                // First frame of a drag — lock the cursor.
                state.viewport_drag_origin = ctx.input(|i| i.pointer.interact_pos());
                ctx.send_viewport_cmd(egui::ViewportCommand::CursorVisible(false));
                ctx.send_viewport_cmd(egui::ViewportCommand::CursorGrab(
                    egui::viewport::CursorGrab::Locked,
                ));
                state.viewport_cursor_grabbed = true;
            } else if !any_drag && state.viewport_cursor_grabbed {
                // Drag ended — unlock the cursor and warp back to the origin.
                ctx.send_viewport_cmd(egui::ViewportCommand::CursorGrab(
                    egui::viewport::CursorGrab::None,
                ));
                ctx.send_viewport_cmd(egui::ViewportCommand::CursorVisible(true));
                if let Some(origin) = state.viewport_drag_origin.take() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::CursorPosition(origin));
                }
                state.viewport_cursor_grabbed = false;
            }

            // Use raw device motion when the cursor is locked; fall back to
            // the normal drag delta otherwise (first frame of drag, or if the
            // platform doesn't support CursorGrab::Locked).
            let delta = if state.viewport_cursor_grabbed {
                ctx.input(|i| i.pointer.motion())
                    .unwrap_or_else(|| response.drag_delta())
            } else {
                response.drag_delta()
            };

            if drag_orbit {
                state.camera_orbit.yaw_deg += delta.x * state.settings.orbit_sensitivity;
                state.camera_orbit.pitch_deg += delta.y * state.settings.orbit_sensitivity;
                state.project_dirty = true;
            }
            if drag_pan {
                let scale = 0.002 * state.camera_orbit.distance * 0.2 * state.settings.pan_sensitivity;
                state.camera_orbit.pan[0] += delta.x * scale;
                state.camera_orbit.pan[1] -= delta.y * scale;
                state.project_dirty = true;
            }
            if response.hovered() {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll.abs() > 0.0 {
                    state.camera_orbit.target_distance = (state.camera_orbit.target_distance
                        - scroll * state.camera_orbit.target_distance * state.settings.zoom_sensitivity)
                        .max(0.1);
                    state.project_dirty = true;
                }
            }

            // ── Debug drawing (CPU-side fallback) ────────────────────────
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
            let cam_rot = [
                state.camera_orbit.pitch_deg,
                state.camera_orbit.yaw_deg,
                0.0,
            ];
            let fov = state.rendering.camera_fov;

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
                    draw_debug_list(&painter, &rect, &skel_list, cam_pos, cam_rot, fov);
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
                    draw_debug_list(&painter, &rect, &col_list, cam_pos, cam_rot, fov);
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
                            draw_debug_list(&painter, &rect, &cloth_list, cam_pos, cam_rot, fov);

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
                                        &rect,
                                        &normal_list,
                                        cam_pos,
                                        cam_rot,
                                        fov,
                                    );
                                }
                            }
                        }
                    }
                }
            }

            // ── Camera PIP wipe ───────────────────────────────────────
            if state.show_camera_wipe {
                let snap = state.app.tracking.mailbox().snapshot();
                if let Some(ref frame) = snap.frame {
                    if snap.sequence != state.camera_wipe_seq {
                        let w = frame.width as usize;
                        let h = frame.height as usize;
                        if w > 0 && h > 0 && frame.rgb_data.len() == w * h * 3 {
                            let needed = w * h * 4;
                            if state.camera_wipe_rgba_buf.len() != needed {
                                state.camera_wipe_rgba_buf.resize(needed, 255);
                            }
                            let rgba = &mut state.camera_wipe_rgba_buf;
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
                            if let Some(ref mut handle) = state.camera_wipe_texture {
                                handle.set(color_image, options);
                            } else {
                                let handle =
                                    ui.ctx().load_texture("camera_wipe", color_image, options);
                                state.camera_wipe_texture = Some(handle);
                            }
                        }
                        state.camera_wipe_seq = snap.sequence;
                    }

                    if let Some(ref tex) = state.camera_wipe_texture {
                        let pip_max_w = 240.0;
                        let tex_size = tex.size_vec2();
                        let scale = pip_max_w / tex_size.x.max(1.0);
                        let pip_w = tex_size.x * scale;
                        let pip_h = tex_size.y * scale;
                        let pip_rect = egui::Rect::from_min_size(
                            egui::pos2(rect.right() - pip_w - 12.0, rect.bottom() - pip_h - 12.0),
                            egui::vec2(pip_w, pip_h),
                        );
                        painter.rect_filled(pip_rect, 4.0, egui::Color32::from_rgb(18, 18, 22));
                        painter.rect_stroke(
                            pip_rect,
                            4.0,
                            egui::Stroke::new(2.0, egui::Color32::from_rgb(60, 130, 200)),
                        );

                        let uv = if state.tracking.tracking_mirror {
                            egui::Rect::from_min_max(egui::pos2(1.0, 0.0), egui::pos2(0.0, 1.0))
                        } else {
                            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0))
                        };
                        painter.image(tex.id(), pip_rect, uv, egui::Color32::WHITE);

                        // ── Detection annotations ──────────────────────
                        if state.show_detection_annotations {
                            if let Some(ref ann) = snap.annotation {
                                let kpt_color =
                                    egui::Color32::from_rgba_unmultiplied(0, 255, 128, 220);
                                let line_color =
                                    egui::Color32::from_rgba_unmultiplied(0, 200, 255, 180);
                                let bb_color =
                                    egui::Color32::from_rgba_unmultiplied(255, 220, 80, 200);

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

            // Show camera info overlay.
            let info = t!("viewport.camera_info", x = format!("{:.2}", cam_pos[0]), y = format!("{:.2}", cam_pos[1]), z = format!("{:.2}", cam_pos[2]), yaw = format!("{:.1}", state.camera_orbit.yaw_deg), pitch = format!("{:.1}", state.camera_orbit.pitch_deg), dist = format!("{:.2}", state.camera_orbit.distance));
            painter.text(
                egui::pos2(rect.left() + 8.0, rect.bottom() - 8.0),
                egui::Align2::LEFT_BOTTOM,
                info,
                egui::FontId::monospace(11.0),
                egui::Color32::from_rgb(90, 90, 100),
            );

            // T04: Draw cloth region selection overlay in ClothAuthoring mode.
            if state.mode == crate::gui::AppMode::ClothAuthoring {
                if let Some(avatar) = state.app.active_avatar() {
                    let sel_ref = state.region_selection.as_ref();
                    match sel_ref {
                        Some(sel) => {
                            let prim_label = avatar
                                .asset
                                .meshes
                                .iter()
                                .flat_map(|m| m.primitives.iter())
                                .find(|p| p.id == sel.target_primitive)
                                .map(|p| t!("viewport.primitive", id = p.id.0))
                                .unwrap_or_else(|| t!("viewport.primitive", id = sel.target_primitive.0));

                            let vert_count = sel.selected_vertices.len();
                            let lines = [
                                t!("viewport.selected", count = vert_count),
                                t!("viewport.target", label = prim_label),
                                t!("viewport.range", start = sel.selected_vertex_range.0, end = sel.selected_vertex_range.1),
                            ];

                            let mut y = rect.top() + 6.0;
                            for line in &lines {
                                painter.text(
                                    egui::pos2(rect.left() + 8.0, y),
                                    egui::Align2::LEFT_TOP,
                                    line,
                                    egui::FontId::monospace(12.0),
                                    egui::Color32::from_rgba_unmultiplied(255, 220, 80, 220),
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
                                egui::Color32::from_rgba_unmultiplied(180, 180, 180, 160),
                            );
                        }
                    }
                }
            }
        });
}
