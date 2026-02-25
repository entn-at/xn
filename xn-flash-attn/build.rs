// Build script to run nvcc and generate the C glue code for launching the flash-attention kernel.
// The cuda build time is very long so compiled artifacts are cached in a build directory
// (defaulting to $HOME/xn/build/, overridable via XN_FLASH_ATTN_BUILD_DIR) and only
// recompiled when source files are newer than the cached library.
use anyhow::{Context, Result};
use std::path::PathBuf;

const KERNEL_FILES: [&str; 9] = [
    "csrc/flash_api.cu",
    "csrc/flash_fwd_hdim64_fp16_sm80.cu",
    "csrc/flash_fwd_hdim128_fp16_sm80.cu",
    "csrc/flash_fwd_hdim64_bf16_sm80.cu",
    "csrc/flash_fwd_hdim128_bf16_sm80.cu",
    "csrc/flash_fwd_hdim64_fp16_causal_sm80.cu",
    "csrc/flash_fwd_hdim128_fp16_causal_sm80.cu",
    "csrc/flash_fwd_hdim64_bf16_causal_sm80.cu",
    "csrc/flash_fwd_hdim128_bf16_causal_sm80.cu",
];

const HEADER_FILES: [&str; 10] = [
    "csrc/flash_fwd_kernel.h",
    "csrc/flash_fwd_launch_template.h",
    "csrc/flash.h",
    "csrc/philox.cuh",
    "csrc/softmax.h",
    "csrc/utils.h",
    "csrc/kernel_traits.h",
    "csrc/block_info.h",
    "csrc/static_switch.h",
    "csrc/hardware_info.h",
];

/// Return the most recent modification time across all source and header files.
fn newest_source_mtime() -> Option<std::time::SystemTime> {
    KERNEL_FILES
        .iter()
        .chain(HEADER_FILES.iter())
        .chain(std::iter::once(&"build.rs"))
        .filter_map(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok())
        .max()
}

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    for kernel_file in KERNEL_FILES.iter() {
        println!("cargo:rerun-if-changed={kernel_file}");
    }
    for header_file in HEADER_FILES.iter() {
        println!("cargo:rerun-if-changed={header_file}");
    }

    let build_dir = match std::env::var("XN_FLASH_ATTN_BUILD_DIR") {
        Ok(build_dir) => {
            let path = PathBuf::from(build_dir);
            path.canonicalize().expect(&format!(
                "Directory doesn't exists: {} (the current directory is {})",
                &path.display(),
                std::env::current_dir()?.display()
            ))
        }
        Err(_) => {
            let path =
                PathBuf::from(std::env::var("HOME").context("HOME not set")?).join(".cache/xn");
            std::fs::create_dir_all(&path)?;
            path.canonicalize()?
        }
    };

    let out_file = build_dir.join("libflashattention.a");

    // Skip recompilation if the cached library is newer than all sources.
    let lib_is_fresh = out_file
        .metadata()
        .and_then(|m| m.modified())
        .ok()
        .zip(newest_source_mtime())
        .map_or(false, |(lib_mtime, src_mtime)| lib_mtime >= src_mtime);

    if !lib_is_fresh {
        let kernels = KERNEL_FILES.iter().collect();
        let mut builder = bindgen_cuda::Builder::default()
            .kernel_paths(kernels)
            .out_dir(build_dir.clone())
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-U__CUDA_NO_HALF_OPERATORS__")
            .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
            .arg("-U__CUDA_NO_HALF2_OPERATORS__")
            .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
            .arg("-Icutlass/include")
            .arg("--expt-relaxed-constexpr")
            .arg("--expt-extended-lambda")
            .arg("--use_fast_math")
            .arg("--verbose");

        let mut is_target_msvc = false;
        if let Ok(target) = std::env::var("TARGET") {
            if target.contains("msvc") {
                is_target_msvc = true;
                builder = builder.arg("-D_USE_MATH_DEFINES");
            }
        }

        if !is_target_msvc {
            builder = builder.arg("-Xcompiler").arg("-fPIC");
        }

        builder.build_lib(out_file);
    }

    let is_target_msvc = std::env::var("TARGET").map_or(false, |t| t.contains("msvc"));
    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=flashattention");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
    Ok(())
}
