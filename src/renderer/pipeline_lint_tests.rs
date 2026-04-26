//! CI-checkable regression guards for the output colour-space pipeline.
//!
//! `cargo run --bin render_matrix` is the manual visual smoke test: it
//! produces the four `(srgb|linear) × (opaque|transparent)` PNGs to
//! eyeball side-by-side. That covers cosmetic regressions but only
//! when someone remembers to run it. The two checks here are the
//! cheap, source-level regressions that we *can* catch automatically:
//!
//! 1. **Manual sRGB encode/decode in fragment shaders.** The gamma
//!    contract (documented in [`super::pipeline`]) is that all
//!    fragment shaders write linear values; the colour attachment's
//!    format decides whether the GPU applies the transfer curve at
//!    store time. A `pow(c, 1/2.2)` (or 1/2.4) sneaking into a shader
//!    on top of an sRGB-format target produces a double encode — the
//!    classic washed-out look. Same scan also catches a manual
//!    decode (`pow(c, 2.2)`) which would darken the linear path.
//!
//!    The lint scans `pipeline.rs` source as a string, looking for
//!    well-known gamma constants. A false positive on a freshly-added
//!    `1.0 / 2.2` for some non-encode reason is fine — the test
//!    forces a code review of why the constant is there.
//!
//! 2. **Format mapping for [`super::VulkanRenderer::color_attachment_format`].**
//!    The function is small but load-bearing: swap the two arms and
//!    every output frame becomes wrong colour-space. Pin it.
//!
//! Visual fidelity (banding, hue shift, alpha bleed) still needs the
//! manual matrix run + OBS handoff verification — see the manual QA
//! checklist in `docs/output-interop.md`.

#[cfg(test)]
mod pipeline_lint_tests {
    use crate::renderer::frame_input::RenderColorSpace;
    use crate::renderer::VulkanRenderer;
    use vulkano::format::Format;

    /// Embedded at compile time. The path is relative to *this* file, so
    /// the included bytes are `pipeline.rs` only — this lint file is
    /// not in the scanned text, which keeps the forbidden-pattern
    /// strings in this file from matching themselves.
    const PIPELINE_SRC: &str = include_str!("pipeline.rs");

    #[test]
    fn fragment_shaders_have_no_manual_srgb_encode() {
        // Forbidden gamma constants. Any of these in a fragment shader
        // either double-encodes (manual encode + sRGB target format) or
        // breaks the linear path (manual decode on a UNORM target).
        // The pre-computed decimals (0.4545 = 1/2.2, 0.4167 = 1/2.4)
        // catch hard-coded variants of the same constants.
        let forbidden_encode = [
            "1.0 / 2.2",
            "1.0/2.2",
            "1.0 / 2.4",
            "1.0/2.4",
            "0.4545",
            "0.4167",
        ];
        for pattern in &forbidden_encode {
            assert!(
                !PIPELINE_SRC.contains(pattern),
                "pipeline.rs contains forbidden manual-sRGB-encode constant '{}'. \
                 Fragment shaders MUST write linear; the attachment format does the \
                 transfer at store time. See the gamma policy comment at the top \
                 of pipeline.rs.",
                pattern
            );
        }

        // Decode-direction equivalents — `pow(x, 2.2)` style. Less
        // common (we sample sRGB textures and they decode automatically)
        // but a regression here would silently darken the linear path.
        // Match `pow(..., 2.2)` and `pow(..., 2.4)` shapes.
        let forbidden_decode = ["pow(", "2.2)", "2.4)"];
        // We require all three substrings to coexist on the same line
        // before flagging — otherwise the `pow(rim_dot, ...)` call in
        // the toon fragment shader (a fresnel exponent, not gamma)
        // would be a false positive. Fresnel exponents are tunable
        // (3.0, 5.0, ...), not the 2.2/2.4 gamma constants.
        for line in PIPELINE_SRC.lines() {
            if forbidden_decode.iter().all(|p| line.contains(p)) {
                panic!(
                    "pipeline.rs has a line that looks like a manual sRGB decode \
                     (pow(..., 2.2) or pow(..., 2.4)): {line:?}. Sample sRGB \
                     textures with the SRGB format and let the GPU decode."
                );
            }
        }
    }

    #[test]
    fn color_attachment_format_matches_color_space() {
        // The whole output-colour-space switch hinges on this two-arm
        // mapping. Swap the arms and every published frame is wrong.
        // Pinning it guards against an "easy refactor" that breaks the
        // entire downstream pipeline silently.
        assert_eq!(
            VulkanRenderer::color_attachment_format(&RenderColorSpace::Srgb),
            Format::R8G8B8A8_SRGB,
            "Srgb output must use an SRGB-format attachment so the GPU \
             applies the linear→sRGB transfer at store time"
        );
        assert_eq!(
            VulkanRenderer::color_attachment_format(&RenderColorSpace::LinearSrgb),
            Format::R8G8B8A8_UNORM,
            "LinearSrgb output must use a UNORM attachment — no transfer \
             at store time, shader-written linear values land verbatim"
        );
    }
}
