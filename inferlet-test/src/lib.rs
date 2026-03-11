//! Diagnostic / warm-up inferlet.
//!
//! Supports several commands via inferlet::receive():
//!   "warmup"    — simple fill + flush + fill_user + generate, sends __WARMUP_DONE__
//!   "__SHUTDOWN__" — exit

use inferlet::stop_condition::{self, StopCondition};
use inferlet::{Args, ChatFormatter, Result, Sampler};

#[inferlet::main]
async fn main(mut _args: Args) -> Result<()> {
    let model = inferlet::get_auto_model();
    let eos = model.eos_tokens();

    loop {
        let msg = inferlet::receive().await;

        if msg == "__SHUTDOWN__" {
            break;
        }

        if msg == "warmup" {
            let template = model.get_prompt_template();
            let mut fmt = ChatFormatter::new();
            fmt.system("You are helpful.");
            let rendered = fmt.render(&template, false, true);
            let mut ctx = model.create_context();
            ctx.fill(&rendered);
            ctx.flush().await;
            ctx.fill_user("Hi");
            let stop = stop_condition::max_len(3).or(stop_condition::ends_with_any(eos.clone()));
            let text = ctx.generate(Sampler::greedy(), stop).await;
            inferlet::send(&format!("warmup OK: {}", text));
            inferlet::send("__WARMUP_DONE__");
        }
    }

    Ok(())
}
