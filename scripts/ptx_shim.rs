// PTX build shim (Phase G).
//
// Standalone no_std crate root that includes the device kernel sources from
// src/cuda/kernels.rs so they can be compiled to PTX with rustc directly:
//
//   rustup run <pinned-nightly> rustc \
//     --target nvptx64-nvidia-cuda --crate-type cdylib \
//     -C panic=abort -C target-cpu=sm_89 \
//     --emit=asm scripts/ptx_shim.rs -o target/ptx/vole_gfx_dev.ptx
//
// (One package: this shim is build tooling inside the same crate, not a
// second Cargo package.)
#![no_std]
#![feature(abi_ptx, core_intrinsics, stdarch_nvptx)]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

include!("../src/cuda/kernels.rs");
