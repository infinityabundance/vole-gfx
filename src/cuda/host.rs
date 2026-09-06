//! Minimal CUDA driver host interaction (feature `cuda`).
//!
//! Hand-rolled FFI to `libcuda` (the driver API) via `libloading`: narrow and
//! dependency-light, matching the crate's dependency policy.  Used by the
//! Phase G parity courts to load the Rust-generated PTX, launch the Rust
//! kernels, and compare device output byte-for-byte with the CPU backends.
//!
//! Context ownership: a `Module` owns one driver context, but a CUDA context
//! is only *current* on the thread that called `cuCtxSetCurrent` for it.
//! Every device-touching method therefore routes through `Module::with_ctx`,
//! which makes the module's context current on the calling thread for the
//! duration of the call and restores the thread's previous current context on
//! the way out (`ContextGuard`).  No thread is ever left with the module's
//! context current after a call returns, and the module never relies on the
//! creating thread's TLS state — so sharing `&Module` across threads is
//! sound, and `Send`/`Sync` are justified *only* by that per-call binding
//! (see the SAFETY notes on the impls).  The one hard contract: `Drop` must
//! not race with an in-flight guarded call on another thread (standard
//! use-after-drop ownership rule).
//!
//! Every unsafe block has a SAFETY note; pointers are validated before use.

#![allow(clippy::missing_safety_doc)]

use std::ffi::c_void;
use std::sync::OnceLock;

type CudaResult = i32; // CUresult

/// Lazily-opened driver library + resolved symbols.  The `Library` handle is
/// kept alive for the process lifetime: dropping it would unload the symbols
/// and turn the function pointers dangling.
struct Driver {
    _lib: libloading::Library,
    init: unsafe extern "C" fn(u32) -> CudaResult,
    device_get: unsafe extern "C" fn(*mut i32, i32) -> CudaResult,
    ctx_create: unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> CudaResult,
    ctx_destroy: unsafe extern "C" fn(*mut c_void) -> CudaResult,
    ctx_get_current: unsafe extern "C" fn(*mut *mut c_void) -> CudaResult,
    ctx_set_current: unsafe extern "C" fn(*mut c_void) -> CudaResult,
    ctx_pop_current: unsafe extern "C" fn(*mut *mut c_void) -> CudaResult,
    module_load: unsafe extern "C" fn(
        *mut *mut c_void,
        *const c_void,
        u32,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CudaResult,
    module_get_function:
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const std::ffi::c_char) -> CudaResult,
    launch: unsafe extern "C" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CudaResult,
    mem_alloc: unsafe extern "C" fn(*mut *mut c_void, u64) -> CudaResult,
    mem_free: unsafe extern "C" fn(*mut c_void) -> CudaResult,
    memcpy_htod: unsafe extern "C" fn(*mut c_void, *const c_void, u64) -> CudaResult,
    memcpy_dtoh: unsafe extern "C" fn(*mut c_void, *const c_void, u64) -> CudaResult,
    device_name: unsafe extern "C" fn(*mut std::ffi::c_char, i32, u32) -> CudaResult,
}

fn driver() -> Result<&'static Driver, String> {
    static D: OnceLock<Result<Driver, String>> = OnceLock::new();
    D.get_or_init(|| {
        // SAFETY: dlopen of the system CUDA driver library; failure is
        // reported, never panics.
        let lib = unsafe { libloading::Library::new("libcuda.so.1") }
            .or_else(|_| unsafe { libloading::Library::new("libcuda.so") })
            .map_err(|e| format!("libcuda load failed: {e}"))?;
        // SAFETY: symbol addresses are typed per the CUDA driver API ABI.
        unsafe {
            Ok(Driver {
                init: *lib.get(b"cuInit").map_err(|e| e.to_string())?,
                device_get: *lib.get(b"cuDeviceGet").map_err(|e| e.to_string())?,
                ctx_create: *lib.get(b"cuCtxCreate_v2").map_err(|e| e.to_string())?,
                ctx_destroy: *lib.get(b"cuCtxDestroy_v2").map_err(|e| e.to_string())?,
                ctx_get_current: *lib.get(b"cuCtxGetCurrent").map_err(|e| e.to_string())?,
                ctx_set_current: *lib.get(b"cuCtxSetCurrent").map_err(|e| e.to_string())?,
                ctx_pop_current: *lib.get(b"cuCtxPopCurrent_v2").map_err(|e| e.to_string())?,
                module_load: *lib.get(b"cuModuleLoadDataEx").map_err(|e| e.to_string())?,
                module_get_function: *lib.get(b"cuModuleGetFunction").map_err(|e| e.to_string())?,
                launch: *lib.get(b"cuLaunchKernel").map_err(|e| e.to_string())?,
                mem_alloc: *lib.get(b"cuMemAlloc_v2").map_err(|e| e.to_string())?,
                mem_free: *lib.get(b"cuMemFree_v2").map_err(|e| e.to_string())?,
                memcpy_htod: *lib.get(b"cuMemcpyHtoD_v2").map_err(|e| e.to_string())?,
                memcpy_dtoh: *lib.get(b"cuMemcpyDtoH_v2").map_err(|e| e.to_string())?,
                device_name: *lib.get(b"cuDeviceGetName").map_err(|e| e.to_string())?,
                _lib: lib,
            })
        }
    })
    .as_ref()
    .map_err(|e| e.clone())
}

fn status(code: CudaResult) -> Result<(), String> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("CUDA driver error code {code}"))
    }
}

/// Like `status` but names the failing call for receipts.
fn ok(code: CudaResult, at: &str) -> Result<(), String> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("{at}: CUDA error {code}"))
    }
}

/// Query driver/runtime versions and device name (receipt fields).
pub struct DeviceInfo {
    pub present: bool,
    pub name: Option<String>,
}

pub fn device_info() -> DeviceInfo {
    let Ok(d) = driver() else {
        return DeviceInfo {
            present: false,
            name: None,
        };
    };
    let run = || -> Result<DeviceInfo, String> {
        // SAFETY: cuInit(0) then cuDeviceGet(0) under the driver lock protocol.
        unsafe {
            ok((d.init)(0), "cuInit")?;
            let mut dev = 0i32;
            ok((d.device_get)(&mut dev, 0), "cuDeviceGet")?;
            let mut name = [0i8; 256];
            ok(
                (d.device_name)(name.as_mut_ptr(), 256, dev as u32),
                "cuDeviceGetName",
            )?;
            let cstr = std::ffi::CStr::from_ptr(name.as_ptr());
            Ok(DeviceInfo {
                present: true,
                name: Some(cstr.to_string_lossy().into_owned()),
            })
        }
    };
    run().unwrap_or(DeviceInfo {
        present: false,
        name: None,
    })
}

/// A loaded module and its context (RAII).
///
/// The context is bound to no thread between calls: every device-touching
/// method acquires it for the calling thread through `with_ctx` and restores
/// the caller's previous current context before returning.
///
/// # SAFETY of `Send`/`Sync`
///
/// The raw handles are context/module IDs valid for the process lifetime of
/// this object.  `Send` is sound because moving the handle between threads
/// transfers ownership and no method relies on the creating thread's TLS
/// state (the guard re-binds per call).  `Sync` is sound because shared
/// `&Module` calls each make the context current on their own thread via
/// `cuCtxSetCurrent` (per-thread in the driver) before touching the driver,
/// and the driver serializes same-context operations internally.  The
/// caller must not drop the `Module` while another thread is inside a
/// guarded method.
pub struct Module {
    ctx: *mut c_void,
    module: *mut c_void,
}

unsafe impl Send for Module {}
unsafe impl Sync for Module {}

/// RAII binding of a module context to the calling thread.
///
/// On construction the module's context becomes current on this thread; on
/// drop the thread's previous current context is restored.  Not `Send`/`Sync`
/// by design: a guard is inherently bound to one thread's TLS slot.
struct ContextGuard<'d> {
    d: &'d Driver,
    prev: *mut c_void,
}

impl<'d> ContextGuard<'d> {
    /// Make `ctx` current on the calling thread, remembering the previous
    /// current context (possibly null) for restoration.
    fn acquire(d: &'d Driver, ctx: *mut c_void) -> Result<ContextGuard<'d>, String> {
        // SAFETY: cuCtxGetCurrent/cuCtxSetCurrent are thread-safe driver
        // calls on valid (possibly null) context handles; ctx belongs to this
        // live Module and is therefore valid until the Module is dropped,
        // which the caller must not do during a guarded call.
        unsafe {
            let mut prev: *mut c_void = std::ptr::null_mut();
            ok((d.ctx_get_current)(&mut prev), "cuCtxGetCurrent")?;
            ok((d.ctx_set_current)(ctx), "cuCtxSetCurrent")?;
            Ok(ContextGuard { d, prev })
        }
    }
}

impl<'d> Drop for ContextGuard<'d> {
    fn drop(&mut self) {
        // SAFETY: restoring a previously-valid (possibly null) context handle
        // on this thread; errors are ignored because the restore must not
        // panic during unwinding and the previous value was current here.
        unsafe {
            let _ = (self.d.ctx_set_current)(self.prev);
        }
    }
}

impl Module {
    /// Run `f` with this module's context current on the calling thread.
    fn with_ctx<R>(&self, f: impl FnOnce(&Driver) -> Result<R, String>) -> Result<R, String> {
        let d = driver()?;
        let _g = ContextGuard::acquire(d, self.ctx)?;
        f(d)
    }

    /// Load PTX text into a fresh context on device 0.  The created context
    /// is popped off the loading thread before returning so the module owns a
    /// context bound to no thread (each later call binds it explicitly).
    pub fn load(ptx: &[u8]) -> Result<Module, String> {
        let d = driver()?;
        // SAFETY: standard driver init sequence; the context is created
        // current on this thread, then popped before returning (below).
        unsafe {
            ok((d.init)(0), "cuInit")?;
            let mut dev = 0i32;
            ok((d.device_get)(&mut dev, 0), "cuDeviceGet")?;
            let mut ctx: *mut c_void = std::ptr::null_mut();
            ok((d.ctx_create)(&mut ctx, 0, dev), "cuCtxCreate")?;
            let mut module: *mut c_void = std::ptr::null_mut();
            let rc = (d.module_load)(
                &mut module,
                ptx.as_ptr().cast(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            if rc != 0 {
                let _ = (d.ctx_destroy)(ctx);
                return Err(format!("cuModuleLoadDataEx error {rc}"));
            }
            // Pop the context so no thread (including this one) is left with
            // it current after load returns.
            let mut popped: *mut c_void = std::ptr::null_mut();
            let prc = (d.ctx_pop_current)(&mut popped);
            if prc != 0 || popped != ctx {
                let _ = (d.ctx_destroy)(ctx);
                return Err(format!("cuCtxPopCurrent error {prc} (popped {popped:?})"));
            }
            Ok(Module { ctx, module })
        }
    }

    fn function(&self, name: &str) -> Result<*mut c_void, String> {
        self.with_ctx(|d| {
            // SAFETY: name is a NUL-terminated C string; module handle valid
            // and its context current (guarded).
            unsafe {
                let cname = std::ffi::CString::new(name).map_err(|e| e.to_string())?;
                let mut f: *mut c_void = std::ptr::null_mut();
                status((d.module_get_function)(&mut f, self.module, cname.as_ptr()))?;
                Ok(f)
            }
        })
    }

    fn alloc(&self, bytes: u64) -> Result<*mut c_void, String> {
        self.with_ctx(|d| {
            // SAFETY: cuMemAlloc on the (guarded) current context with size > 0.
            unsafe {
                let mut p: *mut c_void = std::ptr::null_mut();
                status((d.mem_alloc)(&mut p, bytes))?;
                Ok(p)
            }
        })
    }

    // SAFETY: frees a device pointer allocated by this module's context.
    fn free(&self, p: *mut c_void) -> Result<(), String> {
        self.with_ctx(|d| unsafe { status((d.mem_free)(p)) })
    }

    fn htod(&self, dev: *mut c_void, host: *const c_void, bytes: u64) -> Result<(), String> {
        self.with_ctx(|d| unsafe { status((d.memcpy_htod)(dev, host, bytes)) })
    }

    fn dtoh(&self, host: *mut c_void, dev: *const c_void, bytes: u64) -> Result<(), String> {
        self.with_ctx(|d| unsafe { status((d.memcpy_dtoh)(host, dev, bytes)) })
    }

    /// Launch `f` with `grid`/`block` geometry and per-argument pointers into
    /// an 8-byte-aligned argument buffer (legacy `kernelParams` mode).
    /// `offsets` gives each argument's byte offset inside `base`.
    fn launch(
        &self,
        f: *mut c_void,
        grid: u32,
        block: u32,
        base: *mut u8,
        offsets: &[usize],
    ) -> Result<(), String> {
        self.with_ctx(|d| {
            // SAFETY: base points to 8-byte-aligned live storage for the whole
            // argument set; kernelParams entries point inside it; the module
            // context is current (guarded); the default (legacy) stream
            // serializes with the subsequent blocking memcpys.
            unsafe {
                let mut kernel_params: Vec<*mut c_void> = offsets
                    .iter()
                    .map(|&o| base.add(o) as *mut c_void)
                    .collect();
                let kp = kernel_params.as_mut_ptr();
                status((d.launch)(
                    f,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    kp,
                    std::ptr::null_mut(),
                ))
            }
        })
    }

    /// Fill `n` bytes of device memory with `pattern` using the Rust PTX
    /// `vgf_fill_u8` kernel; returns the device buffer (caller copies out).
    pub fn fill(&self, n: u64, pattern: u8) -> Result<Vec<u8>, String> {
        if n == 0 || n > (1 << 30) {
            return Err("fill size out of range".into());
        }
        let f = self.function("vgf_fill_u8")?;
        let dev = self.alloc(n)?;
        // argument buffer: ptr(8) w u32 h u32 pattern u8 total u32, 8-aligned
        let mut area = vec![0u64; 3]; // 24 bytes, 8-byte aligned
        let base = area.as_mut_ptr() as *mut u8;
        // SAFETY: base is 8-aligned with 24 writable bytes (3 x u64).
        unsafe {
            std::ptr::copy_nonoverlapping(&(dev as u64).to_le_bytes()[0] as *const u8, base, 8);
            let w = n as u32;
            std::ptr::copy_nonoverlapping(&w.to_le_bytes()[0] as *const u8, base.add(8), 4);
            std::ptr::copy_nonoverlapping(&1u32.to_le_bytes()[0] as *const u8, base.add(12), 4);
            *base.add(16) = pattern;
            std::ptr::copy_nonoverlapping(&256u32.to_le_bytes()[0] as *const u8, base.add(20), 4);
        }
        let grid = 1u32;
        let block = 256u32;
        self.launch(f, grid, block, base, &[0, 8, 12, 16, 20])?;
        // read back
        let mut host = vec![0u8; n as usize];
        self.dtoh(host.as_mut_ptr().cast(), dev, n)?;
        self.free(dev)?;
        Ok(host)
    }

    /// Copy `src` (host) to `dst_dev` via the Rust PTX `vgf_copy_u8` kernel
    /// after uploading `src`; returns the device content.
    pub fn copy_kernel(&self, src: &[u8]) -> Result<Vec<u8>, String> {
        let n = src.len() as u64;
        if n == 0 || n > (1 << 30) {
            return Err("copy size out of range".into());
        }
        let f = self.function("vgf_copy_u8")?;
        let src_dev = self.alloc(n)?;
        let dst_dev = self.alloc(n)?;
        self.htod(src_dev, src.as_ptr().cast(), n)?;
        let mut area = vec![0u64; 3]; // 24 bytes, 8-byte aligned
        let base = area.as_mut_ptr() as *mut u8;
        // SAFETY: base is 8-aligned with 24 writable bytes (3 x u64).
        unsafe {
            std::ptr::copy_nonoverlapping(&(dst_dev as u64).to_le_bytes()[0] as *const u8, base, 8);
            std::ptr::copy_nonoverlapping(
                &(src_dev as u64).to_le_bytes()[0] as *const u8,
                base.add(8),
                8,
            );
            let w = n as u32;
            std::ptr::copy_nonoverlapping(&w.to_le_bytes()[0] as *const u8, base.add(16), 4);
            std::ptr::copy_nonoverlapping(&256u32.to_le_bytes()[0] as *const u8, base.add(20), 4);
        }
        let grid = 1u32;
        let block = 256u32;
        self.launch(f, grid, block, base, &[0, 8, 16, 20])?;
        let mut host = vec![0u8; n as usize];
        self.dtoh(host.as_mut_ptr().cast(), dst_dev, n)?;
        self.free(src_dev)?;
        self.free(dst_dev)?;
        Ok(host)
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        let Ok(d) = driver() else { return };
        // SAFETY: handles were created by this process; destruction order
        // module-before-context is what the driver expects.
        unsafe {
            let _ = (d.ctx_destroy)(self.ctx);
        }
    }
}

pub fn status_line() -> String {
    match device_info() {
        DeviceInfo {
            present: true,
            name: Some(n),
        } => format!("cuda host: present ({n})"),
        DeviceInfo { present: false, .. } => "cuda host: no device/driver reachable".into(),
        _ => "cuda host: no device/driver reachable".into(),
    }
}
