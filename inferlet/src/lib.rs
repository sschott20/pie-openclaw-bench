//! Modular KV cache inferlet for PIE — persistent architecture.
//!
//! A single long-running instance handles all cache-build and generate requests
//! via a receive/send message loop. KV pages are cached in-memory using
//! reference-counted KvPage clones — no export/import API needed.
//!
//! Uses `decode_step()` instead of `flush()` to commit pending tokens to KV.
//! decode_step processes all pending tokens in one forward pass and returns
//! a sampled token (which we discard). Unlike flush(), it works reliably
//! across unlimited calls in a single instance.
//!
//! Modes:
//!   - **cache_build**: Load cached prefix → fill module → decode_step → cache result
//!   - **generate**: Load cached prefix → fill remaining modules + user → generate
//!   - **warmup**: Quick forward pass to warm pipeline
//!   - **clear_cache**: Drop all cached KV pages (for reset between reps)
//!   - **shutdown**: Clean exit

use std::collections::HashMap;

use inferlet::forward::KvPage;
use inferlet::stop_condition::{self, StopCondition};
use inferlet::{Args, ChatFormatter, Context, Result, Sampler};
use serde::{Deserialize, Serialize};

// ── Wire types ──────────────────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(tag = "mode")]
enum Message {
    #[serde(rename = "warmup")]
    Warmup,

    #[serde(rename = "cache_build")]
    CacheBuild {
        /// Cache key to load previous prefix (null if first module)
        import_key: Option<String>,
        /// The module content to add
        module_content: String,
        /// Cache key to store the new prefix
        export_key: String,
        /// Whether this is the first module (needs ChatFormatter system prompt)
        is_first_module: bool,
    },

    #[serde(rename = "generate")]
    Generate {
        /// Cache key to load cached prefix (null if no cache)
        import_key: Option<String>,
        /// Number of modules already cached (for metrics)
        cache_hits: u32,
        /// Remaining modules to fill (after cached prefix)
        modules: Vec<ModuleContent>,
        /// Whether the first remaining module is the very first module
        first_module_is_system: bool,
        /// User prompt
        user_prompt: String,
        /// Max tokens to generate
        max_tokens: usize,
        /// For metrics
        program_id: String,
        turn_index: u32,
        total_modules: u32,
    },

    /// Drop all cached KV pages (for reset between repetitions)
    #[serde(rename = "clear_cache")]
    ClearCache,

    #[serde(rename = "shutdown")]
    Shutdown,
}

#[derive(Deserialize)]
struct ModuleContent {
    content: String,
}

#[derive(Clone)]
struct CachedPrefix {
    kv_pages: Vec<KvPage>,
    token_ids: Vec<u32>,
    kv_page_last_len: usize,
}

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

// ── Main ────────────────────────────────────────────────────────────────────

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    let default_max_tokens: usize = args
        .value_from_str(["-n", "--max-tokens"])
        .unwrap_or(200);

    let model = inferlet::get_auto_model();
    let eos_tokens = model.eos_tokens();

    // In-memory KV cache: key → (cloned KvPages, token state)
    // KvPage is Rc-based — clones share the underlying GPU pages,
    // deallocation only happens when the last reference drops.
    let mut cache: HashMap<String, CachedPrefix> = HashMap::new();

    // Persistent message loop
    loop {
        let msg = inferlet::receive().await;

        let message: Message = match serde_json::from_str(&msg) {
            Ok(m) => m,
            Err(e) => {
                inferlet::send(&format!("__ERROR__{{\"error\":\"{}\"}}", e));
                continue;
            }
        };

        match message {
            Message::Warmup => {
                let mut ctx = model.create_context();
                let template = model.get_prompt_template();
                let mut fmt = ChatFormatter::new();
                fmt.system("You are helpful.");
                let rendered = fmt.render(&template, false, true);
                ctx.fill(&rendered);
                ctx.fill_user("Hi");
                let stop = stop_condition::max_len(3)
                    .or(stop_condition::ends_with_any(eos_tokens.clone()));
                let text = ctx.generate(Sampler::greedy(), stop).await;
                inferlet::send(&format!("warmup OK: {}", text));
                inferlet::send("__WARMUP_DONE__");
            }

            Message::CacheBuild {
                import_key,
                module_content,
                export_key,
                is_first_module,
            } => {
                // Load cached prefix or start fresh
                let mut ctx = if let Some(ref key) = import_key {
                    if let Some(cached) = cache.get(key) {
                        // Clone KvPages (Rc ref count bump, shares GPU pages)
                        Context::from_imported_state(
                            &model,
                            cached.kv_pages.clone(),
                            cached.token_ids.clone(),
                            cached.kv_page_last_len,
                        )
                    } else {
                        model.create_context()
                    }
                } else {
                    model.create_context()
                };

                // Fill the new module
                if is_first_module {
                    let mut formatter = ChatFormatter::new();
                    formatter.system(&module_content);
                    let rendered = formatter.render(&model.get_prompt_template(), false, true);
                    ctx.fill(&rendered);
                } else {
                    ctx.fill(&format!("\n\n{}", module_content));
                }

                // Commit pending tokens to KV via decode_step (replaces flush).
                let _ = ctx.decode_step(&Sampler::greedy()).await;

                // Cache the KV state (clone KvPages — Rc keeps GPU pages alive)
                let cached = CachedPrefix {
                    kv_pages: ctx.kv_pages.clone(),
                    token_ids: ctx.get_token_ids().to_vec(),
                    kv_page_last_len: ctx.get_kv_page_last_len(),
                };
                let num_tokens = cached.token_ids.len();
                cache.insert(export_key.clone(), cached);

                // ctx drops here → KvPage Rc count decreases but cache still
                // holds clones → GPU pages stay alive

                inferlet::send(&format!(
                    "__CACHE_BUILT__{{\"key\":\"{}\",\"tokens\":{}}}",
                    export_key, num_tokens
                ));
            }

            Message::Generate {
                import_key,
                cache_hits,
                modules,
                first_module_is_system,
                user_prompt,
                max_tokens,
                program_id,
                turn_index,
                total_modules,
            } => {
                let max_tokens = if max_tokens > 0 { max_tokens } else { default_max_tokens };

                // Load cached KV or start fresh
                let mut ctx = if let Some(ref key) = import_key {
                    if let Some(cached) = cache.get(key) {
                        Context::from_imported_state(
                            &model,
                            cached.kv_pages.clone(),
                            cached.token_ids.clone(),
                            cached.kv_page_last_len,
                        )
                    } else {
                        model.create_context()
                    }
                } else {
                    model.create_context()
                };

                let tokens_before = ctx.get_token_ids().len() as u32
                    + ctx.token_ids_pending.len() as u32;

                // Fill remaining modules
                for (i, module) in modules.iter().enumerate() {
                    if i == 0 && first_module_is_system {
                        let mut formatter = ChatFormatter::new();
                        formatter.system(&module.content);
                        let rendered = formatter.render(&model.get_prompt_template(), false, true);
                        ctx.fill(&rendered);
                    } else {
                        ctx.fill(&format!("\n\n{}", module.content));
                    }
                }

                // Fill user prompt and generate
                ctx.fill_user(&user_prompt);
                let stop_cond = stop_condition::max_len(max_tokens)
                    .or(stop_condition::ends_with_any(eos_tokens.clone()));
                let text = ctx.generate(Sampler::greedy(), stop_cond).await;
                inferlet::send(&text);

                let tokens_after = ctx.get_token_ids().len() as u32;
                let tokens_computed = tokens_after.saturating_sub(tokens_before);
                let cache_misses = total_modules - cache_hits;

                let metrics = ResponseMetrics {
                    program_id,
                    turn_index,
                    cache_hits,
                    cache_misses,
                    tokens_saved: tokens_before,
                    tokens_computed,
                    prefill_ms: 0.0,
                    generation_ms: 0.0,
                };
                inferlet::send(&format!(
                    "__DONE__{}",
                    serde_json::to_string(&metrics).unwrap()
                ));
            }

            Message::ClearCache => {
                let count = cache.len();
                cache.clear(); // Drops all KvPage clones → GPU pages freed
                inferlet::send(&format!("__CACHE_CLEARED__{{\"count\":{}}}", count));
            }

            Message::Shutdown => {
                inferlet::send("__SHUTDOWN__");
                break;
            }
        }
    }

    Ok(())
}
