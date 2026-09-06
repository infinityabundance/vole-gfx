//! VOLE-GFX principal binary.
//!
//! Scriptable CLI; every subcommand supports `--json`.  Subcommands land in
//! the phase that implements them; unsupported subcommands report so and do
//! not fake success.

use clap::{Parser, Subcommand};
use serde::Serialize;

#[derive(Parser)]
#[command(name = "vole-gfx", version, about = "Deterministic procedural visual state as a graphics IR", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Print versioned universe identity and limits.
    Universe,
    /// Decode + summarize a `.volegfx` document.
    Inspect {
        /// Path to a canonical IR file.
        file: String,
        #[arg(long)]
        json: bool,
    },
    /// Validate a document (canonical + semantic).
    Validate {
        file: String,
        #[arg(long)]
        json: bool,
    },
    /// Canonicalize a byte string (decode + re-encode); fails if not canonical.
    Canonicalize {
        file: String,
        #[arg(long)]
        json: bool,
    },
    /// Materialize an observation request from a document.
    Materialize {
        file: String,
        /// observation time in nanoseconds
        t: u64,
        /// output width
        w: u32,
        /// output height
        h: u32,
        /// output format: gray8 | rgba8
        #[arg(long, default_value = "rgba8")]
        format: String,
        /// optional PPM/PGM output path
        #[arg(long)]
        out: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Verify an evidence receipt file.
    EvidenceVerify {
        file: String,
        #[arg(long)]
        json: bool,
    },
    /// Encode a document from a simple JSON description.
    Encode {
        file: String,
        /// output IR path
        out: String,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Serialize)]
struct JsonOut<T: Serialize> {
    ok: bool,
    command: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<T>,
}

fn main() {
    let cli = Cli::parse();
    let code = run(cli);
    std::process::exit(code);
}

fn run(cli: Cli) -> i32 {
    match cli.command {
        Commands::Universe => {
            let u = vole_gfx::universe::Universe::u1();
            println!(
                "{}",
                serde_json::to_string_pretty(&u.limits).unwrap_or_default()
            );
            0
        }
        Commands::Inspect { file, json } => {
            let doc = load(&file);
            match doc {
                Ok(doc) => {
                    let obj = vole_gfx::evidence::receipt::Receipt::new("cli-inspect", "ir");
                    let data = serde_json::json!({
                        "file": file,
                        "objects": doc.objects.len(),
                        "trajectories": doc.trajectories.len(),
                        "instances": doc.instances.len(),
                        "events": doc.events.len(),
                        "serialized_bytes": vole_gfx::ir::encode::encode(&doc).len(),
                    });
                    if json {
                        println!(
                            "{}",
                            serde_json::to_string_pretty(&JsonOut {
                                ok: true,
                                command: "inspect",
                                error: None,
                                data: Some(data)
                            })
                            .unwrap()
                        );
                    } else {
                        println!(
                            "objects={} trajectories={} instances={} events={}",
                            data["objects"],
                            data["trajectories"],
                            data["instances"],
                            data["events"]
                        );
                    }
                    let _ = obj;
                    0
                }
                Err(e) => fail("inspect", e, json),
            }
        }
        Commands::Validate { file, json } => {
            match load(&file).and_then(|d| {
                vole_gfx::ir::validate::validate(&d).map_err(|e| e.to_string())?;
                Ok(d)
            }) {
                Ok(_) => {
                    println_json_or_plain(
                        json,
                        "validate",
                        true,
                        serde_json::json!({ "file": file, "valid": true }),
                    );
                    0
                }
                Err(e) => fail("validate", e, json),
            }
        }
        Commands::Canonicalize { file, json } => {
            let bytes = std::fs::read(&file);
            match bytes {
                Ok(b) => match vole_gfx::ir::canonical::check_canonical(&b) {
                    Ok(()) => {
                        println_json_or_plain(
                            json,
                            "canonicalize",
                            true,
                            serde_json::json!({ "file": file, "canonical": true, "bytes": b.len() }),
                        );
                        0
                    }
                    Err(e) => fail("canonicalize", e, json),
                },
                Err(e) => fail("canonicalize", e.to_string(), json),
            }
        }
        Commands::Materialize {
            file,
            t,
            w,
            h,
            format,
            out,
            json,
        } => {
            let fmt = match format.as_str() {
                "gray8" => vole_gfx::color::ColorFormat::Gray8,
                "rgba8" => vole_gfx::color::ColorFormat::Rgba8,
                _ => {
                    eprintln!("format must be gray8|rgba8");
                    return 1;
                }
            };
            match load(&file).and_then(|doc| {
                let req = vole_gfx::observation::ObservationRequest::full_surface(t, w, h, fmt);
                let m = vole_gfx::materialize::scalar::materialize_document(&doc, &req)
                    .map_err(|e| e.to_string())?;
                Ok((doc, m))
            }) {
                Ok((doc, m)) => {
                    let hash = m.output.canonical_hash().to_hex();
                    let doc_bytes = vole_gfx::ir::encode::encode(&doc).len() as u64;
                    let persistent_bytes = doc_bytes;
                    if let Some(p) = out
                        && let Err(e) = vole_gfx::io::write_pnm(
                            std::path::Path::new(&p),
                            w,
                            h,
                            fmt,
                            &m.output.data,
                        ) {
                            return fail("materialize", e, json);
                        }
                    let data = serde_json::json!({
                        "file": file,
                        "time_ns": t,
                        "w": w,
                        "h": h,
                        "format": fmt.tag(),
                        "canonical_hash": hash,
                        "persistent_bytes": persistent_bytes,
                        "counters": {
                            "samples": m.counters.samples,
                            "instance_tests": m.counters.instance_tests,
                            "instance_draws": m.counters.instance_draws,
                            "ops_applied": m.counters.ops_applied,
                            "residual_records": m.counters.residual_records,
                        },
                    });
                    println_json_or_plain(json, "materialize", true, data);
                    0
                }
                Err(e) => fail("materialize", e, json),
            }
        }
        Commands::EvidenceVerify { file, json } => {
            match std::fs::read(&file).and_then(|b| {
                vole_gfx::evidence::receipt::Receipt::verify(&b).map_err(std::io::Error::other)
            }) {
                Ok(h) => {
                    println_json_or_plain(
                        json,
                        "evidence-verify",
                        true,
                        serde_json::json!({ "file": file, "hash": h.to_hex(), "valid": true }),
                    );
                    0
                }
                Err(e) => fail("evidence-verify", e.to_string(), json),
            }
        }
        Commands::Encode { file, out, json } => {
            match std::fs::read_to_string(&file)
                .map_err(|e| e.to_string())
                .and_then(|s| encode_from_json(&s))
                .and_then(|doc| {
                    vole_gfx::ir::validate::validate(&doc).map_err(|e| e.to_string())?;
                    Ok(vole_gfx::ir::encode::encode(&doc))
                }) {
                Ok(bytes) => match std::fs::write(&out, &bytes) {
                    Ok(()) => {
                        println_json_or_plain(
                            json,
                            "encode",
                            true,
                            serde_json::json!({ "out": out, "bytes": bytes.len() }),
                        );
                        0
                    }
                    Err(e) => fail("encode", e.to_string(), json),
                },
                Err(e) => fail("encode", e, json),
            }
        }
    }
}

fn load(path: &str) -> Result<vole_gfx::ir::Document, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    vole_gfx::ir::decode::decode(&bytes).map_err(|e| format!("decode: {e}"))
}

fn fail(cmd: &'static str, e: impl std::fmt::Display, json: bool) -> i32 {
    if json {
        let out = JsonOut::<serde_json::Value> {
            ok: false,
            command: cmd,
            error: Some(e.to_string()),
            data: None,
        };
        println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
    } else {
        eprintln!("{cmd}: {e}");
    }
    1
}

fn println_json_or_plain(json: bool, cmd: &'static str, ok: bool, data: serde_json::Value) {
    if json {
        let out = JsonOut {
            ok,
            command: cmd,
            error: None,
            data: Some(data),
        };
        println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
    } else {
        match data {
            serde_json::Value::Object(m) => {
                for (k, v) in m {
                    if !v.is_object() && !v.is_array() {
                        println!("{k}={v}");
                    }
                }
            }
            other => println!("{other}"),
        }
    }
}

/// Minimal JSON document builder for hand-authored tests: accepts a flat
/// object with counts of raster objects etc.  Full structured authoring lands
/// with the corpus tooling.
fn encode_from_json(_s: &str) -> Result<vole_gfx::ir::Document, String> {
    Err(
        "encode-from-json: structured JSON authoring is not implemented yet (Phase H tooling)"
            .into(),
    )
}
