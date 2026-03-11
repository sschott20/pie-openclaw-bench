//! Modular KV cache inferlet for PIE — one-shot architecture.
//!
//! Each instance handles exactly ONE operation then exits. Two modes:
//!
//! 1. **cache_build**: Import prefix KV → fill one module → flush → export → exit
//!    Builds per-module KV cache incrementally. Each instance does exactly one
//!    flush (which works on fresh instances).
//!
//! 2. **generate**: Import prefix KV → fill remaining modules + user → generate → exit
//!    Serves a request using cached KV. No flush needed — decode_step handles
//!    all pending tokens.
//!
//! 3. **warmup**: Quick forward pass to warm PIE server pipeline.

use inferlet::forward::Forward;
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
        /// Store key to import previous prefix KV (null if first module)
        import_key: Option<String>,
        /// The module content to add
        module_content: String,
        /// Store key to export the new prefix KV
        export_key: String,
        /// Whether this is the first module (needs ChatFormatter system prompt)
        is_first_module: bool,
    },

    #[serde(rename = "generate")]
    Generate {
        /// Store key to import cached prefix KV (null if no cache)
        import_key: Option<String>,
        /// Number of modules already cached (for metrics)
        cache_hits: u32,
        /// Remaining modules to fill (after cached prefix)
        modules: Vec<ModuleContent>,
        /// Whether the first remaining module is the very first module (needs system prompt)
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
}

#[derive(Deserialize)]
struct ModuleContent {
    content: String,
}

#[derive(Serialize, Deserialize)]
struct CachedState {
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

    // One-shot: receive a single message and process it
    let msg = inferlet::receive().await;

    let message: Message = match serde_json::from_str(&msg) {
        Ok(m) => m,
        Err(e) => {
            inferlet::send(&format!("__ERROR__{{\"error\":\"{}\"}}", e));
            return Ok(());
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
            // Build KV cache for one module layer
            let mut ctx = if let Some(ref key) = import_key {
                let state_key = format!("{}_state", key);
                let state_json = inferlet::store_get(&state_key)
                    .expect("cache_build: missing state for import_key");
                let state: CachedState = serde_json::from_str(&state_json).unwrap();
                let queue = model.create_queue();
                let pages = queue.import_kv_pages(key);
                Context::from_imported_state(&model, pages, state.token_ids, state.kv_page_last_len)
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

            // Single flush — computes KV for the new tokens
            ctx.flush().await;

            // Export KV pages and state
            let state = CachedState {
                token_ids: ctx.get_token_ids().to_vec(),
                kv_page_last_len: ctx.get_kv_page_last_len(),
            };
            ctx.queue().export_kv_pages(&ctx.kv_pages, &export_key);
            inferlet::store_set(
                &format!("{}_state", export_key),
                &serde_json::to_string(&state).unwrap(),
            );

            inferlet::send(&format!(
                "__CACHE_BUILT__{{\"key\":\"{}\",\"tokens\":{}}}",
                export_key,
                state.token_ids.len()
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

            // Import cached KV or start fresh
            let mut ctx = if let Some(ref key) = import_key {
                let state_key = format!("{}_state", key);
                if let Some(state_json) = inferlet::store_get(&state_key) {
                    let state: CachedState = serde_json::from_str(&state_json).unwrap();
                    let queue = model.create_queue();
                    let pages = queue.import_kv_pages(key);
                    if !pages.is_empty() {
                        Context::from_imported_state(
                            &model, pages, state.token_ids, state.kv_page_last_len,
                        )
                    } else {
                        model.create_context()
                    }
                } else {
                    model.create_context()
                }
            } else {
                model.create_context()
            };

            let tokens_before = ctx.get_token_ids().len() as u32
                + ctx.token_ids_pending.len() as u32;

            // Fill remaining modules (no flush — decode_step handles all pending)
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
    }

    Ok(())
}
