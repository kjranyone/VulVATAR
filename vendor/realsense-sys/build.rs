//! Build script for finding / linking librealsense.
//!
//! This script has a few main functions:
//!
//! 1. Find librealsense on the current system
//! 2. If the buildtime-bindgen feature is enabled, we run bindgen over the librealsense headers
//!    and generate bindings.rs
//! 3. Link this crate to the librealsense2 library.
//!
//! NOTE: If we build in "docs-only" mode (the feature), then this script does nothing, since we
//! don't need to link to librealsense2 or regenerate bindings to build the docs.

fn main() {
    if cfg!(feature = "docs-only") {
        return;
    }

    // Probe libary
    let library = pkg_config::probe_library("realsense2")
        .expect("pkg-config failed to find realsense2 package");
    let major_version = library
        .version
        .find('.')
        .map(|i| &library.version[..i])
        .expect("failed to determine librealsense major version");

    if major_version != "2" {
        panic!(
            "librealsense2 version {} is not supported, expected major version 2",
            library.version
        )
    }

    // Pin the rerun conditions explicitly. Upstream emits no
    // rerun-if-changed directive, leaving cargo's default ("rerun if any
    // file in the package changed") active while the script rewrites
    // bindings inside the package directory on every run — a
    // self-sustaining loop that recompiles everything downstream on each
    // build. Watching the SDK include paths keeps bindings regenerating
    // on SDK upgrades.
    println!("cargo:rerun-if-changed=build.rs");
    for dir in &library.include_paths {
        println!("cargo:rerun-if-changed={}", dir.display());
    }

    // generate bindings
    #[cfg(feature = "buildtime-bindgen")]
    {

        // The function below will leave us with the directory <SDKHome>/include/librealsense2/
        let include_dir = library
            .include_paths
            .iter()
            .filter_map(|path| {
                let dir = std::path::Path::new(path).join("librealsense2");
                if dir.is_dir() {
                    Some(dir)
                } else {
                    None
                }
            })
            .next()
            .expect("fail find librealsense2 include directory");

        // pop the last item off of the include_dir to get `/include`, which we'll need to build rsutil_delegate.h
        let mut top_include = include_dir.clone();
        top_include.pop();
        let bindings = bindgen::Builder::default()
            .clang_arg("-fno-inline-functions")
            // Include... `<SDKHome>/include`
            // Again, this is just so that we can compile rsutil_delegate.h
            .clang_arg(String::from("-I") + top_include.to_str().unwrap())
            .header(include_dir.join("rs.h").to_str().unwrap())
            .header(
                include_dir
                    .join("h")
                    .join("rs_pipeline.h")
                    .to_str()
                    .unwrap(),
            )
            .header(
                include_dir
                    .join("h")
                    .join("rs_advanced_mode_command.h")
                    .to_str()
                    .unwrap(),
            )
            .header(include_dir.join("h").join("rs_config.h").to_str().unwrap())
            .allowlist_var("RS2_.*")
            .allowlist_type("rs2_.*")
            .allowlist_function("rs2_.*")
            .allowlist_function("_rs2_.*")
            .generate()
            .expect("Unable to generate bindings");

        // Write the bindings to OUT_DIR (cargo guarantees it exists)
        // rather than into the package directory: writing inside the
        // package trips the package-file change detection and re-runs
        // this script on every subsequent build.
        let bindings_file =
            std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("bindings.rs");
        bindings
            .write_to_file(bindings_file)
            .expect("Couldn't write bindings!");
    }

    // link the libraries specified by pkg-config.
    for dir in &library.link_paths {
        println!("cargo:rustc-link-search=native={}", dir.to_str().unwrap());
    }
    for lib in &library.libs {
        println!("cargo:rustc-link-lib={}", lib);
    }

    #[cfg(target_os = "windows")]
    if let Some(dll_loc) = &library.defines["DLL_FOLDER"] {
        // Move DLL from DLL_FOLDER location to the deps folder for this executable.
        //
        // The current_exe() function returns the directory:
        //
        // `<topLevel>/target/<buildType>/build/realsense-sys<hash>/executable.exe`
        //
        // ...however, the proper place for the DLL is actually in
        //
        // `<topLevel>/target/<buildType>/deps`
        //
        // So, pop three times, add two strings, and we're good to go with the right location.
        // Is it pretty? No. But it'll work for now.
        let mut exe_path = std::env::current_exe().unwrap();
        exe_path.pop();
        exe_path.pop();
        exe_path.pop();
        exe_path.push("deps");
        exe_path.push("realsense2.dll");
        let dll_dest = exe_path.to_str().unwrap();
        let mut dll_src = std::path::PathBuf::from(dll_loc);
        dll_src.push("realsense2.dll");
        // Re-run (and re-copy) when the SDK DLL is updated, so the copy
        // in deps/ never goes stale relative to the installed SDK.
        println!("cargo:rerun-if-changed={}", dll_src.display());
        match std::fs::copy(dll_src.clone(), dll_dest) {
            Ok(_) => println!("DLL successfully copied to deps folder."),
            Err(e) if std::path::Path::new(dll_dest).exists() => {
                // The deps copy is loaded by a running process (the app
                // itself, or a test/bench binary started from deps/).
                // This copy only refreshes an already-installed DLL, so
                // a sharing violation while one runs must not fail the
                // build: keep the existing copy. `rerun-if-changed` on
                // the SDK DLL keeps retrying on later builds, so a
                // genuinely stale copy refreshes once the lock clears.
                println!(
                    "cargo:warning=deps/realsense2.dll is locked by a running process ({e}); keeping the existing copy"
                );
            }
            Err(e) => panic!("{}; attempting from source {:#?}", e, dll_src),
        }
    }
}
