//! Baseline inferlet: same protocol as modular-kv-cache but no caching.
//!
//! Concatenates all module content and prefills from scratch each request.
//! Uses the same JSON request/response protocol for fair comparison.

use inferlet::stop_condition::{self, StopCondition};
use inferlet::{Args, ChatFormatter, Result, Sampler};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Request {
    program_id: String,
    turn_index: u32,
    modules: Vec<ModulePayload>,
    max_tokens: usize,
}

#[derive(Deserialize)]
struct ModulePayload {
    content: String,
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

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    let default_max_tokens: usize = args
        .value_from_str(["-n", "--max-tokens"])
        .unwrap_or(200);

    let model = inferlet::get_auto_model();
    let eos_tokens = model.eos_tokens();

    loop {
        let msg = inferlet::receive().await;

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

        // No caching: create fresh context and fill all modules as system prompt
        let mut ctx = model.create_context();
        let num_modules = request.modules.len() as u32;

        // Concatenate all module content
        let full_content: String = request.modules.iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        // Use ChatFormatter for proper template tokens (consistent with cache inferlet)
        let template = model.get_prompt_template();
        let mut fmt = ChatFormatter::new();
        fmt.system(&full_content);
        let rendered = fmt.render(&template, false, true);
        ctx.fill(&rendered);
        ctx.fill_user("Continue.");

        let tokens_computed = ctx.get_token_ids().len() as u32;

        let stop_cond = stop_condition::max_len(max_tokens)
            .or(stop_condition::ends_with_any(eos_tokens.clone()));

        // Use generate() which handles the decode loop internally
        let text = ctx.generate(Sampler::greedy(), stop_cond).await;
        inferlet::send(&text);

        let metrics = ResponseMetrics {
            program_id: request.program_id,
            turn_index: request.turn_index,
            cache_hits: 0,
            cache_misses: num_modules,
            tokens_saved: 0,
            tokens_computed,
            prefill_ms: 0.0,  // Timing measured Python-side (Instant not reliable in WASM)
            generation_ms: 0.0,
        };
        let metrics_json = serde_json::to_string(&metrics).expect("serialize metrics");
        inferlet::send(&format!("__DONE__{}", metrics_json));
    }

    Ok(())
}
