//! Phase G gate: Rust→PTX toolchain + exact CUDA kernel parity on real
//! NVIDIA hardware (feature `cuda` required).
//!
//! Builds the device PTX from src/cuda/kernels.rs with the pinned nightly,
//! loads it through the minimal driver binding, launches the Rust kernels,
//! and compares device output byte-for-byte with the CPU reference.
//!
//! Run: `cargo run --release --features cuda --example phase_g_gate`

use vole_gfx::evidence::receipt::{Receipt, emit_receipt};

const NIGHTLY: &str = "nightly-2026-07-24";

fn main() {
    // 1) build the PTX from Rust device code in this crate
    let ptx_path = std::path::Path::new("target").join("ptx/vole_gfx_dev.ptx");
    let _ = std::fs::remove_file(&ptx_path);
    let out = std::process::Command::new("rustup")
        .args([
            "run",
            NIGHTLY,
            "rustc",
            "--target",
            "nvptx64-nvidia-cuda",
            "--crate-type",
            "cdylib",
            "-C",
            "panic=abort",
            "-C",
            "overflow-checks=no",
            "-Z",
            "ub-checks=no",
            "-C",
            "target-cpu=sm_89",
            "-C",
            "unsafe-allow-abi-mismatch=target-cpu",
            "--emit=asm",
            "scripts/ptx_shim.rs",
            "-o",
        ])
        .arg(&ptx_path)
        .output()
        .expect("run rustup");
    assert!(
        out.status.success(),
        "ptx build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let ptx = std::fs::read(&ptx_path).expect("read ptx");
    let text = String::from_utf8_lossy(&ptx);
    assert!(
        text.contains(".visible .entry vgf_fill_u8"),
        "fill kernel entry missing"
    );
    assert!(
        text.contains(".visible .entry vgf_copy_u8"),
        "copy kernel entry missing"
    );
    assert!(text.contains(".target sm_89"), "sm target missing");
    eprintln!(
        "PTX built: {} bytes, entries vgf_fill_u8 + vgf_copy_u8 (sm_89)",
        ptx.len()
    );

    let mut r = Receipt::new("phase-g-ptx-cuda-parity", "cuda");
    let gpu = vole_gfx::cuda::host::device_info();
    if let Some(n) = &gpu.name {
        r.outputs.insert("gpu".into(), n.clone());
        eprintln!("GPU: {n}");
    }

    // 2) load module + run parity
    match vole_gfx::cuda::host::Module::load(&ptx) {
        Err(e) => {
            r.pass = false;
            r.unsupported_reason = Some(format!("CUDA driver/module load failed: {e}"));
            let path = emit_receipt(r).expect("emit");
            println!("PHASE G UNSUPPORTED (driver: {e}); receipt {path}");
            return;
        }
        Ok(m) => {
            // fill parity: 1920x1080 gray surface filled with 0xA5
            let n = 1920u64 * 1080;
            let got = m.fill(n, 0xA5).expect("device fill");
            assert_eq!(got.len() as u64, n);
            assert!(got.iter().all(|&b| b == 0xA5), "fill mismatch");
            eprintln!("vgf_fill_u8 parity OK ({} bytes)", n);

            // copy parity: pseudo-random buffer copied device-side
            let mut src = vec![0u8; 1920 * 1080 * 4];
            let mut seed = 123u64;
            for b in src.iter_mut() {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *b = (seed >> 33) as u8;
            }
            let got = m.copy_kernel(&src).expect("device copy");
            assert_eq!(got, src, "copy mismatch");
            eprintln!("vgf_copy_u8 parity OK ({} bytes)", src.len());

            r.pass = true;
            r.outputs.insert("surface".into(), "1920x1080".into());
            r.metrics.insert("fill_bytes".into(), n);
            r.metrics.insert("copy_bytes".into(), src.len() as u64);
            r.notes.push(format!(
                "Rust device code -> PTX (nightly {NIGHTLY}, sm_89) -> cuModuleLoadDataEx -> cuLaunchKernel; device output byte-identical to CPU reference for fill and copy kernels"
            ));
            r.canonical_hash = Some(vole_gfx::hash::sha256(&src).to_hex());
        }
    }
    let path = emit_receipt(r).expect("emit");
    println!("PHASE G PASS; receipt {path}");
}
