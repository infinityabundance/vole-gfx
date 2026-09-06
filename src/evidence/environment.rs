//! Environment snapshots for receipts: CPU/GPU/OS/compiler/build facts.

use serde::Serialize;
use std::path::Path;

#[derive(Debug, Clone, Default, Serialize, serde::Deserialize)]
pub struct CpuInfo {
    pub vendor_model: Option<String>,
    pub cores: Option<usize>,
    pub avx2: bool,
    pub avx512_subfeatures: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, serde::Deserialize)]
pub struct GpuInfo {
    pub present: bool,
    pub name: Option<String>,
    pub driver: Option<String>,
    pub cuda_umd: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, serde::Deserialize)]
pub struct CompilerInfo {
    pub rustc: Option<String>,
    pub rustc_version: Option<String>,
    pub llvm: Option<String>,
    pub profile: Option<String>,
    pub target_cpu: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, serde::Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub kernel: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, serde::Deserialize)]
pub struct Environment {
    pub cpu: CpuInfo,
    pub gpu: GpuInfo,
    pub compiler: CompilerInfo,
    pub system: SystemInfo,
    pub git_commit: Option<String>,
    pub git_dirty: Option<bool>,
}

fn read_first_line(path: &str) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| s.lines().next().map(|l| l.trim().to_string()))
        .filter(|s| !s.is_empty())
}

fn run_capture(cmd: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

impl Environment {
    /// Snapshot the current machine.  All probes are best-effort and never
    /// fatal.
    pub fn snapshot() -> Environment {
        let mut env = Environment::default();
        env.system.os = std::env::consts::OS.to_string();
        env.system.kernel = read_first_line("/proc/sys/kernel/osrelease");

        if let Ok(procs) = std::fs::read_to_string("/proc/cpuinfo") {
            let mut vendor_model = None;
            let mut flags: Vec<String> = Vec::new();
            let mut cores = 0usize;
            for line in procs.lines() {
                if let Some(v) = line.strip_prefix("model name") {
                    vendor_model = Some(v.trim_start_matches(':').trim().to_string());
                } else if let Some(f) = line.strip_prefix("flags") {
                    flags = f
                        .trim_start_matches(':')
                        .split_whitespace()
                        .map(|s| s.to_string())
                        .collect();
                } else if let Some(c) = line.strip_prefix("processor") {
                    let _ = c;
                    cores += 1;
                }
            }
            env.cpu.vendor_model = vendor_model;
            env.cpu.cores = Some(cores.max(1));
            env.cpu.avx2 = flags.iter().any(|f| f == "avx2");
            env.cpu.avx512_subfeatures = flags
                .iter()
                .filter(|f| f.starts_with("avx512"))
                .cloned()
                .collect();
        }

        if let Some(smi) = run_capture(
            "nvidia-smi",
            &["--query-gpu=name,driver_version", "--format=csv,noheader"],
        ) {
            let mut parts = smi.splitn(2, ',');
            if let (Some(name), Some(drv)) = (parts.next(), parts.next()) {
                env.gpu.present = true;
                env.gpu.name = Some(name.trim().to_string());
                env.gpu.driver = Some(drv.trim().to_string());
            }
        }
        env.gpu.cuda_umd = read_first_line("/proc/driver/nvidia/version").or_else(|| {
            run_capture("nvidia-smi", &[]).and_then(|s| {
                s.lines()
                    .find(|l| l.contains("CUDA UMD"))
                    .map(|l| l.split_whitespace().last().unwrap_or_default().to_string())
            })
        });

        env.compiler.rustc = run_capture("rustc", &["-vV"]).or_else(|| {
            std::env::var("RUSTC")
                .ok()
                .and_then(|c| run_capture(&c, &["-vV"]))
        });
        if let Some(vv) = &env.compiler.rustc {
            for line in vv.lines() {
                if let Some(v) = line.strip_prefix("release:") {
                    env.compiler.rustc_version = Some(v.trim().to_string());
                } else if let Some(v) = line.strip_prefix("LLVM version:") {
                    env.compiler.llvm = Some(v.trim().to_string());
                }
            }
        }
        env.compiler.profile = std::env::var("PROFILE").ok();
        env.compiler.target_cpu = std::env::var("VOLE_GFX_TARGET_CPU").ok();

        env.git_commit = run_capture("git", &["rev-parse", "HEAD"]);
        env.git_dirty = run_capture("git", &["status", "--porcelain"]).map(|s| !s.is_empty());
        let _ = Path::new("."); // keep Path import used if cfg differs
        env
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_is_best_effort() {
        let e = Environment::snapshot();
        assert!(!e.system.os.is_empty());
    }
}
