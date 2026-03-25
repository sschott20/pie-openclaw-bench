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
//!   - **modular_cache_build**: Prefill single module independently, pad to full pages
//!   - **modular_generate**: Concatenate independently-cached modules, then generate
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

    /// Prefill a single module independently with page-aligned padding.
    /// Each module is prefilled from a fresh context (no cross-module attention),
    /// padded so all KV pages are full, and cached for later concatenation.
    #[serde(rename = "modular_cache_build")]
    ModularCacheBuild {
        /// The module content to prefill
        module_content: String,
        /// Cache key to store the result
        export_key: String,
        /// Whether this is the first module (needs ChatFormatter system prompt wrapping)
        is_first_module: bool,
    },

    /// Concatenate independently-cached modules and generate.
    /// Loads each module's KV pages (all page-aligned), concatenates them,
    /// restores context via from_imported_state, re-masks padding regions,
    /// fills user prompt, and generates token-by-token (streaming).
    #[serde(rename = "modular_generate")]
    ModularGenerate {
        /// Ordered list of cache keys (one per module)
        cache_keys: Vec<String>,
        /// User prompt
        user_prompt: String,
        /// Max tokens to generate
        max_tokens: usize,
        /// For metrics
        program_id: String,
        turn_index: u32,
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

/// A single module's KV state cached with page-aligned padding.
/// All pages are full (kv_page_last_len == kv_page_size), enabling
/// clean concatenation across independently-prefilled modules.
#[derive(Clone)]
struct ModularCachedModule {
    kv_pages: Vec<KvPage>,
    token_ids: Vec<u32>,
    /// Number of real (non-padding) tokens. Tokens at indices
    /// [real_token_count .. token_ids.len()] are padding and must be masked.
    real_token_count: usize,
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

    // Modular cache: independently-prefilled modules with page-aligned padding.
    // Each entry has all KV pages full, enabling concatenation for modular_generate.
    let mut modular_cache: HashMap<String, ModularCachedModule> = HashMap::new();

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

                // Fill user prompt and generate token-by-token (streaming)
                ctx.fill_user(&user_prompt);
                let mut generated_tokens: Vec<u32> = Vec::new();

                loop {
                    let token_id = ctx.decode_step(&Sampler::greedy()).await;
                    ctx.fill_token(token_id);
                    generated_tokens.push(token_id);

                    // Send each token as it's generated
                    let token_text = ctx.tokenizer.detokenize(&[token_id]);
                    inferlet::send(&token_text);

                    // Check stop conditions
                    if generated_tokens.len() >= max_tokens {
                        break;
                    }
                    if eos_tokens.iter().any(|eos| generated_tokens.ends_with(eos)) {
                        break;
                    }
                }

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

            Message::ModularCacheBuild {
                module_content,
                export_key,
                is_first_module,
            } => {
                // Always start from a fresh context — no cross-module attention.
                let mut ctx = model.create_context();
                let page_size = ctx.kv_page_size;

                // Fill the module content
                if is_first_module {
                    let mut formatter = ChatFormatter::new();
                    formatter.system(&module_content);
                    let rendered = formatter.render(&model.get_prompt_template(), false, true);
                    ctx.fill(&rendered);
                } else {
                    ctx.fill(&module_content);
                }

                // Commit module tokens to KV via decode_step
                let _ = ctx.decode_step(&Sampler::greedy()).await;

                // Record real token count (everything committed so far)
                let real_token_count = ctx.token_ids.len();

                // Pad to fill the last KV page so all pages are full.
                // After decode_step, kv_page_last_len tells us how full the last page is.
                let last_len = ctx.kv_page_last_len;
                let pad_needed = if last_len == page_size {
                    0
                } else {
                    page_size - last_len
                };

                if pad_needed > 0 {
                    // Fill padding tokens (token ID 0)
                    for _ in 0..pad_needed {
                        ctx.fill_token(0);
                    }
                    // Commit padding to KV
                    let _ = ctx.decode_step(&Sampler::greedy()).await;

                    // Mask all padding tokens so they don't affect attention
                    ctx.mask_token_range(real_token_count, real_token_count + pad_needed, true);
                }

                // All pages should now be full
                let cached = ModularCachedModule {
                    kv_pages: ctx.kv_pages.clone(),
                    token_ids: ctx.get_token_ids().to_vec(),
                    real_token_count,
                };
                let total_tokens = cached.token_ids.len();
                let num_pages = cached.kv_pages.len();
                modular_cache.insert(export_key.clone(), cached);

                inferlet::send(&format!(
                    "__MODULAR_CACHE_BUILT__{{\"key\":\"{}\",\"tokens\":{},\"real_tokens\":{},\"pages\":{},\"pad_tokens\":{}}}",
                    export_key, total_tokens, real_token_count, num_pages, pad_needed
                ));
            }

            Message::ModularGenerate {
                cache_keys,
                user_prompt,
                max_tokens,
                program_id,
                turn_index,
            } => {
                let max_tokens = if max_tokens > 0 { max_tokens } else { default_max_tokens };
                let total_modules = cache_keys.len() as u32;

                // Concatenate all modules' KV pages and token_ids.
                // Track padding regions for re-masking after from_imported_state.
                let mut all_kv_pages: Vec<KvPage> = Vec::new();
                let mut all_token_ids: Vec<u32> = Vec::new();
                // (start_index, end_index) of padding regions to mask
                let mut padding_regions: Vec<(usize, usize)> = Vec::new();
                let mut cache_hits: u32 = 0;
                let mut missing_keys: Vec<String> = Vec::new();

                for key in &cache_keys {
                    if let Some(cached_mod) = modular_cache.get(key) {
                        let offset = all_token_ids.len();
                        all_kv_pages.extend(cached_mod.kv_pages.clone());
                        all_token_ids.extend(cached_mod.token_ids.clone());

                        // Track padding region if this module has padding
                        if cached_mod.real_token_count < cached_mod.token_ids.len() {
                            let pad_start = offset + cached_mod.real_token_count;
                            let pad_end = offset + cached_mod.token_ids.len();
                            padding_regions.push((pad_start, pad_end));
                        }
                        cache_hits += 1;
                    } else {
                        missing_keys.push(key.clone());
                    }
                }

                if !missing_keys.is_empty() {
                    inferlet::send(&format!(
                        "__ERROR__{{\"error\":\"modular_generate: missing cache keys: {:?}\"}}",
                        missing_keys
                    ));
                    continue;
                }

                if all_kv_pages.is_empty() {
                    inferlet::send("__ERROR__{\"error\":\"modular_generate: no modules to concatenate\"}");
                    continue;
                }

                // All pages are full, so kv_page_last_len == kv_page_size
                let page_size = model.get_kv_page_size() as usize;
                let mut ctx = Context::from_imported_state(
                    &model,
                    all_kv_pages,
                    all_token_ids,
                    page_size, // last page is full (all modules are page-aligned)
                );

                // Re-apply padding masks for each module's padding region
                for (pad_start, pad_end) in &padding_regions {
                    ctx.mask_token_range(*pad_start, *pad_end, true);
                }

                let tokens_before = ctx.token_ids.len() as u32;

                // Fill user prompt and generate token-by-token (streaming)
                ctx.fill_user(&user_prompt);
                let mut generated_tokens: Vec<u32> = Vec::new();

                loop {
                    let token_id = ctx.decode_step(&Sampler::greedy()).await;
                    ctx.fill_token(token_id);
                    generated_tokens.push(token_id);

                    // Send each token as it's generated
                    let token_text = ctx.tokenizer.detokenize(&[token_id]);
                    inferlet::send(&token_text);

                    // Check stop conditions
                    if generated_tokens.len() >= max_tokens {
                        break;
                    }
                    if eos_tokens.iter().any(|eos| generated_tokens.ends_with(eos)) {
                        break;
                    }
                }

                let tokens_after = ctx.get_token_ids().len() as u32;
                let tokens_computed = tokens_after.saturating_sub(tokens_before);

                let metrics = ResponseMetrics {
                    program_id,
                    turn_index,
                    cache_hits,
                    cache_misses: total_modules - cache_hits,
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
                let modular_count = modular_cache.len();
                cache.clear();
                modular_cache.clear();
                inferlet::send(&format!(
                    "__CACHE_CLEARED__{{\"count\":{},\"modular_count\":{}}}",
                    count, modular_count
                ));
            }

            Message::Shutdown => {
                inferlet::send("__SHUTDOWN__");
                break;
            }
        }
    }

    Ok(())
}
