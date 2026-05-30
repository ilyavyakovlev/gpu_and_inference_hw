import torch
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)


def optimized_loop(model, input_ids, n_steps):
    with torch.inference_mode():
        # Prefill: process the entire prompt in one pass to populate the KV cache.
        # This replaces the first slow_loop iteration, which also processed all
        # prompt tokens but discarded the cache.
        outputs = model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)  # (1, 1)
        token_tensors = [next_token]

        # Decode: each step processes only 1 new token, with past KV reused.
        # This avoids the O(seq_len^2) attention scaling of the slow baseline.
        for _ in range(n_steps - 1):
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            token_tensors.append(next_token)

        # Collect token values at the end to avoid 128 per-step CPU-GPU syncs.
        return [t.item() for t in token_tensors]


def profile(loop_fn, model, input_ids, trace_name: str):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        loop_fn(model, input_ids, PROFILE_STEPS)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    trace_path = RESULTS_DIR / trace_name
    prof.export_chrome_trace(str(trace_path))
    print(f"Chrome trace saved to {trace_path}")


def generate_optimized(optimized_trace_name: str) -> float:
    # Load in bfloat16: halves memory bandwidth per weight/activation vs float32,
    # which directly reduces time in memory-bandwidth-bound decode steps.
    model = build_model(torch.bfloat16)
    input_ids = get_input_ids()

    profile(optimized_loop, model, input_ids, optimized_trace_name)
    elapsed = time_generation(optimized_loop, model, input_ids, "Optimized")
    return elapsed


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix:
#
# 1. KV cache (use_cache=True) — by far the biggest win.
#    The slow baseline passes the full growing sequence (prompt + all generated
#    tokens so far) to the model at every step. Attention cost scales as
#    O(seq_len^2), so step N does O((1024+N)^2) work. Over 128 steps this is
#    roughly 128 × avg_seq_len^2 ≈ 128 × 1088^2 token-pair operations.
#    With the KV cache, the prefill pass (1024 tokens) happens once, and each
#    of the 127 remaining decode steps attends to exactly 1 new query token
#    against the cached keys/values. Total attention work drops by ~100×.
#    Estimated contribution: ~4–6× of the overall speedup.
#
# 2. bfloat16 model weights and activations.
#    Loading the model in bf16 instead of fp32 halves the bytes moved per
#    weight matrix and activation tensor. For memory-bandwidth-limited decode
#    steps (each forward pass reads the full model weights for a single token),
#    this cuts per-step time by close to 2×.
#    Estimated contribution: ~1.8–2× additional speedup on top of KV cache.
#
# 3. torch.inference_mode() around the generation loop.
#    Disables autograd bookkeeping (version counters, gradient tape). Minor
#    overhead reduction, but meaningful when many small ops are chained.
#    Estimated contribution: <5% on its own.
#
# 4. Deferred .item() calls.
#    The slow baseline calls next_token_id.item() inside every loop iteration,
#    forcing a CPU-GPU synchronization 128 times. We collect token tensors
#    and call .item() only after the loop, eliminating 127 unnecessary syncs
#    and allowing the GPU pipeline to stay full between steps.
#    Estimated contribution: ~10–15% additional speedup.
#
# Biggest impact and why:
#    The KV cache is overwhelmingly the most impactful change. Without it, every
#    decode step re-runs full self-attention over the entire sequence (1024+ tokens),
#    making per-step cost grow with sequence length. With it, each decode step
#    attends over exactly 1 query position and reuses cached keys/values, keeping
#    per-step cost constant at O(1). This changes the total work from O(n^2) to
#    O(n), which is fundamentally a different complexity class — not just a
#    constant-factor improvement. The bfloat16 change compounds on top by reducing
#    the memory traffic per step, but the KV cache change is what makes the
#    difference between a 1× and a 4×+ speedup.
