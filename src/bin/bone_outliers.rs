//! Extract the worst N images per bone from the validate_pipeline
//! JSONL output, plus a top-level "worst overall" list. Used to
//! prioritise pose-solver fixes after a regression run.
//!
//! Usage:
//!   cargo run --bin bone_outliers -- <results.jsonl> [--top N]

use std::collections::HashMap;
use std::env;
use std::fs;

fn main() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or("usage: bone_outliers <results.jsonl> [--top N]")?;
    let mut top = 10usize;
    if let Some(arg) = args.next() {
        if arg == "--top" {
            top = args
                .next()
                .ok_or("--top needs N")?
                .parse()
                .map_err(|e| format!("--top parse: {e}"))?;
        }
    }
    let text = fs::read_to_string(&path).map_err(|e| format!("read {path}: {e}"))?;

    let mut per_bone: HashMap<String, Vec<(f64, String)>> = HashMap::new();
    let mut overall: Vec<(f64, String)> = Vec::new();
    for line in text.lines() {
        let image = extract(line, "\"image\":\"", "\"").unwrap_or("?").to_string();
        if let Some(s) = extract(line, "\"score_deg\":", ",") {
            if let Ok(v) = s.parse::<f64>() {
                overall.push((v, image.clone()));
            }
        }
        // Find each `"name":{"angle_deg":VALUE` triple. The previous
        // implementation used `rfind` for the bone-name look-back,
        // which silently shifted attribution by one bone whenever the
        // current angle_deg was not the first in the JSON line. We
        // now scan forward and pair adjacent matches deterministically.
        let mut cursor = 0;
        while let Some(open) = line[cursor..].find("\":{\"angle_deg\":") {
            let abs_open = cursor + open;
            // Name ends at the `"` immediately before `":{"angle_deg`.
            let name_close = abs_open; // position of bone name's closing `"`
            let name_open = line[..name_close].rfind('"').unwrap_or(0);
            let bone = &line[name_open + 1..name_close];
            let val_start = abs_open + "\":{\"angle_deg\":".len();
            let after = &line[val_start..];
            if let Some(end) = after.find(|c: char| c == ',' || c == '}') {
                if let Ok(v) = after[..end].parse::<f64>() {
                    per_bone
                        .entry(bone.to_string())
                        .or_default()
                        .push((v, image.clone()));
                }
            }
            cursor = val_start;
        }
    }

    println!("# Worst {top} overall");
    overall.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    for (s, img) in overall.iter().take(top) {
        println!("  {s:>6.1}°  {img}");
    }

    let mut bones: Vec<String> = per_bone.keys().cloned().collect();
    bones.sort();
    for bone in bones {
        let list = per_bone.get_mut(&bone).unwrap();
        list.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        println!();
        println!("# Worst {top} for {bone}");
        for (s, img) in list.iter().take(top) {
            println!("  {s:>6.1}°  {img}");
        }
    }
    Ok(())
}

fn extract<'a>(s: &'a str, lhs: &str, rhs: &str) -> Option<&'a str> {
    let start = s.find(lhs)? + lhs.len();
    let rest = &s[start..];
    let end = rest.find(rhs)?;
    Some(&rest[..end])
}
