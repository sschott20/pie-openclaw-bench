// Test: Can generate() be called multiple times in a single inferlet instance?
//
// The flush() bug corrupts host state after first use in a looping inferlet.
// But agent-react and agent-codeact examples call generate() in a loop with
// zero flush() calls. This test confirms whether that pattern works reliably.
//
// Test plan:
//   1. Multi-generate: fill system + user, generate, fill new user, generate again (3x)
//   2. Intra-instance export/import: export KV after first turn, import in same instance
//   3. Print all outputs so we can verify coherence
//
// Run: pie run -p <wasm_path> -m <Pie.toml> -- '{"mode":"multi_generate","rounds":3}'
//   or: pie run -p <wasm_path> -m <Pie.toml> -- '{"mode":"intra_export_import"}'

use inferlet::forward::Forward;
use inferlet::stop_condition::{self, StopCondition};
use inferlet::{Args, Context, Result, Sampler};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
#[serde(tag = "mode")]
enum Message {
    /// Call generate() N times in a loop (like agent-react)
    #[serde(rename = "multi_generate")]
    MultiGenerate {
        #[serde(default = "default_rounds")]
        rounds: u32,
        #[serde(default = "default_max_tokens")]
        max_tokens: usize,
    },

    /// Export KV pages mid-instance, then import and continue generating
    #[serde(rename = "intra_export_import")]
    IntraExportImport {
        #[serde(default = "default_max_tokens")]
        max_tokens: usize,
    },
}

fn default_rounds() -> u32 { 3 }
fn default_max_tokens() -> usize { 30 }

#[derive(Serialize, Deserialize)]
struct CachedState {
    token_ids: Vec<u32>,
    kv_page_last_len: usize,
}

const SYSTEM: &str = "You are a helpful math tutor. Answer concisely.";

const QUESTIONS: &[&str] = &[
    "What is 2 + 2?",
    "What is 10 * 5?",
    "What is 100 / 4?",
    "What is 7 - 3?",
    "What is 9 * 9?",
];

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    let model = inferlet::get_auto_model();
    let eos_tokens = model.eos_tokens();

    // Get JSON from the free (positional) arguments that pico_args didn't consume
    let free_args = args.finish();
    let msg = free_args.first().expect("usage: pass JSON as first argument after --");
    let msg_str = msg.to_str().expect("argument is not valid UTF-8");
    println!("Received: {}", msg_str);
    let message: Message = serde_json::from_str(msg_str).expect("invalid JSON input");

    match message {
        Message::MultiGenerate { rounds, max_tokens } => {
            // ── Test 1: generate() called multiple times, no flush ──
            println!("=== Test: multi_generate (rounds={}, max_tokens={}) ===", rounds, max_tokens);

            let mut ctx = model.create_context();
            ctx.fill_system(SYSTEM);

            for i in 0..rounds {
                let question = QUESTIONS[i as usize % QUESTIONS.len()];
                println!("\n--- Round {} ---", i + 1);
                println!("  Question: {}", question);

                ctx.fill_user(question);

                let stop = stop_condition::max_len(max_tokens)
                    .or(stop_condition::ends_with_any(eos_tokens.clone()));
                let output = ctx.generate(Sampler::greedy(), stop).await;

                let total_tokens = ctx.get_token_ids().len();
                println!("  Answer: {}", output);
                println!("  Total KV tokens: {}", total_tokens);
                println!("  KV pages: {}", ctx.kv_pages.len());

                if output.is_empty() {
                    println!("  ERROR: generate() returned empty string on round {}", i + 1);
                }
            }

            println!("\n=== RESULT: multi_generate completed {} rounds ===", rounds);
            println!("  Final KV tokens: {}", ctx.get_token_ids().len());
            println!("  Final KV pages: {}", ctx.kv_pages.len());
        }

        Message::IntraExportImport { max_tokens } => {
            // ── Test 2: export KV, then import in same instance, then generate ──
            println!("=== Test: intra_export_import ===");

            // Phase A: build context, generate once
            let mut ctx = model.create_context();
            ctx.fill_system(SYSTEM);
            ctx.fill_user(QUESTIONS[0]);

            let stop = stop_condition::max_len(max_tokens)
                .or(stop_condition::ends_with_any(eos_tokens.clone()));
            let output1 = ctx.generate(Sampler::greedy(), stop).await;
            println!("Phase A - Question: {}", QUESTIONS[0]);
            println!("Phase A - Answer: {}", output1);
            println!("Phase A - KV tokens: {}, pages: {}", ctx.get_token_ids().len(), ctx.kv_pages.len());

            // Phase B: export current KV state (intra-instance)
            println!("\nExporting KV pages...");
            let state = CachedState {
                token_ids: ctx.get_token_ids().to_vec(),
                kv_page_last_len: ctx.get_kv_page_last_len(),
            };
            ctx.queue().export_kv_pages(&ctx.kv_pages, "test_export");
            inferlet::store_set("test_export_state", &serde_json::to_string(&state).unwrap());
            println!("  Exported {} tokens, {} pages", state.token_ids.len(), ctx.kv_pages.len());

            // Phase C: import into a NEW context in same instance
            println!("\nImporting KV pages into new context...");
            let queue = model.create_queue();
            let pages = queue.import_kv_pages("test_export");
            let state_json = inferlet::store_get("test_export_state").unwrap();
            let imported_state: CachedState = serde_json::from_str(&state_json).unwrap();
            println!("  Imported {} pages, {} tokens", pages.len(), imported_state.token_ids.len());

            let mut ctx2 = Context::from_imported_state(
                &model,
                pages,
                imported_state.token_ids,
                imported_state.kv_page_last_len,
            );

            // Phase D: ask a new question on the imported context, generate
            ctx2.fill_user(QUESTIONS[1]);
            let stop2 = stop_condition::max_len(max_tokens)
                .or(stop_condition::ends_with_any(eos_tokens.clone()));
            let output2 = ctx2.generate(Sampler::greedy(), stop2).await;
            println!("\nPhase D - Question: {}", QUESTIONS[1]);
            println!("Phase D - Answer: {}", output2);
            println!("Phase D - KV tokens: {}, pages: {}", ctx2.get_token_ids().len(), ctx2.kv_pages.len());

            if output2.is_empty() {
                println!("ERROR: generate() returned empty after import");
            }

            // Phase E: generate AGAIN on the imported context (tests generate loop after import)
            ctx2.fill_user(QUESTIONS[2]);
            let stop3 = stop_condition::max_len(max_tokens)
                .or(stop_condition::ends_with_any(eos_tokens.clone()));
            let output3 = ctx2.generate(Sampler::greedy(), stop3).await;
            println!("\nPhase E - Question: {}", QUESTIONS[2]);
            println!("Phase E - Answer: {}", output3);
            println!("Phase E - KV tokens: {}, pages: {}", ctx2.get_token_ids().len(), ctx2.kv_pages.len());

            // Cleanup
            queue.release_exported_kv_pages("test_export");

            println!("\n=== RESULT: intra_export_import completed ===");
        }
    }

    Ok(())
}
