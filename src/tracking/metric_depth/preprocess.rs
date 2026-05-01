//! 644×476 BCHW RGB resize for MoGe-2.
//!
//! Output is float32 in `[0, 1]` (NOT ImageNet-normalized): MoGe's
//! encoder applies the ImageNet mean/std *internally*, so feeding
//! pre-normalized pixels would double-normalize and produce garbage.
//! See `moge/model/modules.py:96-97` upstream.

use ndarray::Array4;

use super::consts::{INPUT_H, INPUT_W};

pub(super) fn preprocess(rgb: &[u8], src_w: u32, src_h: u32) -> Array4<f32> {
    let dst_w = INPUT_W as usize;
    let dst_h = INPUT_H as usize;
    let mut tensor = Array4::<f32>::zeros((1, 3, dst_h, dst_w));
    if src_w == 0 || src_h == 0 {
        return tensor;
    }
    let src_w_us = src_w as usize;
    let src_h_us = src_h as usize;
    let max_x = src_w_us.saturating_sub(1);
    let max_y = src_h_us.saturating_sub(1);
    let scale_x = src_w as f32 / dst_w as f32;
    let scale_y = src_h as f32 / dst_h as f32;
    let stride = src_w_us * 3;
    let expected_len = stride * src_h_us;
    if rgb.len() < expected_len {
        return tensor;
    }
    let inv_255 = 1.0 / 255.0;

    for dy in 0..dst_h {
        let fy = ((dy as f32) + 0.5) * scale_y - 0.5;
        let y0f = fy.floor();
        let wy1 = (fy - y0f).clamp(0.0, 1.0);
        let wy0 = 1.0 - wy1;
        let y0 = (y0f as i32).clamp(0, max_y as i32) as usize;
        let y1 = ((y0f as i32) + 1).clamp(0, max_y as i32) as usize;
        let row0 = y0 * stride;
        let row1 = y1 * stride;
        for dx in 0..dst_w {
            let fx = ((dx as f32) + 0.5) * scale_x - 0.5;
            let x0f = fx.floor();
            let wx1 = (fx - x0f).clamp(0.0, 1.0);
            let wx0 = 1.0 - wx1;
            let x0 = (x0f as i32).clamp(0, max_x as i32) as usize;
            let x1 = ((x0f as i32) + 1).clamp(0, max_x as i32) as usize;
            let i00 = row0 + x0 * 3;
            let i01 = row0 + x1 * 3;
            let i10 = row1 + x0 * 3;
            let i11 = row1 + x1 * 3;
            let w00 = wy0 * wx0;
            let w01 = wy0 * wx1;
            let w10 = wy1 * wx0;
            let w11 = wy1 * wx1;
            let r = rgb[i00] as f32 * w00
                + rgb[i01] as f32 * w01
                + rgb[i10] as f32 * w10
                + rgb[i11] as f32 * w11;
            let g = rgb[i00 + 1] as f32 * w00
                + rgb[i01 + 1] as f32 * w01
                + rgb[i10 + 1] as f32 * w10
                + rgb[i11 + 1] as f32 * w11;
            let b = rgb[i00 + 2] as f32 * w00
                + rgb[i01 + 2] as f32 * w01
                + rgb[i10 + 2] as f32 * w10
                + rgb[i11 + 2] as f32 * w11;
            tensor[(0, 0, dy, dx)] = r * inv_255;
            tensor[(0, 1, dy, dx)] = g * inv_255;
            tensor[(0, 2, dy, dx)] = b * inv_255;
        }
    }
    tensor
}
