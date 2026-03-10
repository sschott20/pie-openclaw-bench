//! Modular KV cache inferlet for per-module prompt caching.
//!
//! Receives modular requests via message IPC, caches KV pages per-module using
//! content-hash invalidation, and streams generated tokens back.
//!
//! Each prompt module (core_instructions, tool_schemas, skills, etc.) gets its
//! own named KV page handle. Unchanged modules are imported via memcpy instead
//! of recomputed on GPU.

use inferlet::forward::Forward;
use inferlet::stop_condition::{self, StopCondition};
use inferlet::{Args, ChatFormatter, Context, Result, Sampler, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Metadata stored alongside exported KV pages for a cached module.
#[derive(Serialize, Deserialize)]
struct CachedModuleState {
    content_hash: String,
    token_ids: Vec<u32>,
    kv_page_last_len: usize,
}

/// Incoming request from the benchmark harness.
#[derive(Deserialize)]
struct Request {
    program_id: String,
    turn_index: u32,
    modules: Vec<ModulePayload>,
    max_tokens: usize,
}

/// A single prompt module within a request.
#[derive(Deserialize)]
struct ModulePayload {
    name: String,
    content: String,
    hash: String,
}

/// Metrics reported back to the harness after each request.
#[derive(Serialize)]
struct ResponseMetrics {
    program_id: String,
    turn_index: u32,
    cache_hits: u32,
    cache_misses: u32,
    tokens_saved: u32,
    tokens_computed: u32,
    prefill_ms: f64,
    generation_ms: f64,
}

/// Store key prefix for module metadata in PIE's persistent KV store.
const MODULE_META_PREFIX: &str = "mod_meta_";

/// Load cached module metadata from PIE's persistent KV store.
fn load_module_meta(name: &str) -> Option<CachedModuleState> {
    let key = format!("{}{}", MODULE_META_PREFIX, name);
    inferlet::store_get(&key).and_then(|json| serde_json::from_str(&json).ok())
}

/// Save module metadata to PIE's persistent KV store.
fn save_module_meta(name: &str, state: &CachedModuleState) {
    let key = format!("{}{}", MODULE_META_PREFIX, name);
    let json = serde_json::to_string(state).expect("serialize module state");
    inferlet::store_set(&key, &json);
}

/// Delete module metadata from PIE's persistent KV store.
fn delete_module_meta(name: &str) {
    let key = format!("{}{}", MODULE_META_PREFIX, name);
    inferlet::store_delete(&key);
}

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    let default_max_tokens: usize = args
        .value_from_str(["-n", "--max-tokens"])
        .unwrap_or(200);

    let model = inferlet::get_auto_model();
    let eos_tokens = model.eos_tokens();

    // Persistent loop: receive requests, process with modular caching, respond.
    loop {
        // Wait for next request from harness
        let msg = inferlet::receive().await;

        // Check for shutdown signal
        if msg == "__SHUTDOWN__" {
            break;
        }

        let request: Request = match serde_json::from_str(&msg) {
            Ok(r) => r,
            Err(e) => {
                inferlet::send(&format!("__ERROR__{{\"error\":\"{}\"}}", e));
                continue;
            }
        };

        let max_tokens = if request.max_tokens > 0 {
            request.max_tokens
        } else {
            default_max_tokens
        };

        let queue = model.create_queue();
        let mut all_kv_pages = Vec::new();
        let mut all_token_ids: Vec<u32> = Vec::new();
        let mut last_kv_page_len: usize = 0;
        let mut cache_hits: u32 = 0;
        let mut cache_misses: u32 = 0;
        let mut tokens_saved: u32 = 0;
        let mut tokens_computed: u32 = 0;

        let prefill_start = Instant::now();

        for module in &request.modules {
            // Check if this module is cached with matching content hash
            if let Some(meta) = load_module_meta(&module.name) {
                if meta.content_hash == module.hash {
                    // CACHE HIT: import pre-computed KV pages
                    let pages = queue.import_kv_pages(&module.name);
                    all_kv_pages.extend(pages);
                    all_token_ids.extend(&meta.token_ids);
                    last_kv_page_len = meta.kv_page_last_len;
                    cache_hits += 1;
                    tokens_saved += meta.token_ids.len() as u32;
                    continue;
                }
                // Content changed: release old cache
                queue.release_exported_kv_pages(&module.name);
                delete_module_meta(&module.name);
            }

            // CACHE MISS: prefill this module and cache it
            let mut ctx = model.create_context();
            ctx.fill(&module.content);
            ctx.flush().await;

            let token_ids = ctx.get_token_ids().to_vec();
            let kv_page_last_len = ctx.get_kv_page_last_len();

            // Export KV pages under module name for future reuse
            ctx.queue()
                .export_kv_pages(&ctx.kv_pages, &module.name);

            // Save metadata
            save_module_meta(
                &module.name,
                &CachedModuleState {
                    content_hash: module.hash.clone(),
                    token_ids: token_ids.clone(),
                    kv_page_last_len,
                },
            );

            tokens_computed += token_ids.len() as u32;
            all_kv_pages.extend(ctx.kv_pages);
            all_token_ids.extend(token_ids);
            last_kv_page_len = kv_page_last_len;
            cache_misses += 1;
        }

        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        // Reconstruct full context from assembled module KV pages
        let mut ctx = Context::from_imported_state(
            &model,
            all_kv_pages,
            all_token_ids,
            last_kv_page_len,
        );

        // Generate response, streaming tokens back
        let gen_start = Instant::now();
        let stop_cond = stop_condition::max_len(max_tokens)
            .or(stop_condition::ends_with_any(eos_tokens.clone()));
        let sampler = Sampler::greedy();

        let mut token_count = 0u32;
        loop {
            let token_id = ctx.decode_step(&sampler).await;
            let text = model.get_tokenizer().detokenize(&[token_id]);

            // Check stop condition
            if stop_cond.check(&ctx) {
                break;
            }

            // Stream token to harness
            inferlet::send(&text);
            token_count += 1;
        }

        let generation_ms = gen_start.elapsed().as_secs_f64() * 1000.0;

        // Send completion message with metrics
        let metrics = ResponseMetrics {
            program_id: request.program_id,
            turn_index: request.turn_index,
            cache_hits,
            cache_misses,
            tokens_saved,
            tokens_computed,
            prefill_ms,
            generation_ms,
        };
        let metrics_json = serde_json::to_string(&metrics).expect("serialize metrics");
        inferlet::send(&format!("__DONE__{}", metrics_json));
    }

    Ok(())
}
