//! Benchmark: Flash Attention vs matmul-based attention.
//!
//! Run with: cargo run --release --example bench

use xn::cuda_backend::Device;
use xn::{Backend, Result, Tensor};

fn run_bench(
    batch_size: usize,
    num_heads: usize,
    seqlen_q: usize,
    seqlen_k: usize,
    head_dim: usize,
    num_iters: usize,
    device: &Device,
) -> Result<()> {
    println!(
        "\n--- q=[{batch_size}, {num_heads}, {seqlen_q}, {head_dim}], \
         k/v=[{batch_size}, {num_heads}, {seqlen_k}, {head_dim}] ---"
    );

    let q_numel = batch_size * num_heads * seqlen_q * head_dim;
    let kv_numel = batch_size * num_heads * seqlen_k * head_dim;

    let q_data: Vec<half::bf16> = (0..q_numel)
        .map(|i| half::bf16::from_f32((i % 127) as f32 * 0.01))
        .collect();
    let k_data: Vec<half::bf16> = (0..kv_numel)
        .map(|i| half::bf16::from_f32((i % 113) as f32 * 0.01))
        .collect();
    let v_data: Vec<half::bf16> = (0..kv_numel)
        .map(|i| half::bf16::from_f32((i % 97) as f32 * 0.01))
        .collect();

    let q: Tensor<half::bf16, Device> =
        Tensor::from_vec(q_data, (batch_size, num_heads, seqlen_q, head_dim), device)?;
    let k: Tensor<half::bf16, Device> =
        Tensor::from_vec(k_data, (batch_size, num_heads, seqlen_k, head_dim), device)?;
    let v: Tensor<half::bf16, Device> =
        Tensor::from_vec(v_data, (batch_size, num_heads, seqlen_k, head_dim), device)?;

    let softmax_scale = 1.0 / (head_dim as f32).sqrt();
    let flops = 4.0
        * batch_size as f64
        * num_heads as f64
        * seqlen_q as f64
        * seqlen_k as f64
        * head_dim as f64;

    // --- Flash Attention ---
    let flash_out = xn_flash_attn::flash_attn(&q, &k, &v, softmax_scale, false)?;
    device.synchronize()?;

    let start = std::time::Instant::now();
    for _ in 0..num_iters {
        let _out = xn_flash_attn::flash_attn(&q, &k, &v, softmax_scale, false)?;
    }
    device.synchronize()?;
    let elapsed = start.elapsed();

    let avg_us = elapsed.as_micros() as f64 / num_iters as f64;
    let tflops = flops * num_iters as f64 / elapsed.as_secs_f64() / 1e12;
    println!(
        "  flash-attn:  {num_iters} iters in {elapsed:.2?} ({avg_us:.1} us/iter, {tflops:.2} TFLOP/s)"
    );

    // --- Matmul Attention ---
    let scale = half::bf16::from_f32(softmax_scale);
    let matmul_out = {
        let scores = q.matmul_t(&k)?.scale(scale)?;
        let weights = scores.softmax()?;
        weights.matmul(&v)?
    };
    device.synchronize()?;

    let start = std::time::Instant::now();
    for _ in 0..num_iters {
        let scores = q.matmul_t(&k)?.scale(scale)?;
        let weights = scores.softmax()?;
        let _out = weights.matmul(&v)?;
    }
    device.synchronize()?;
    let elapsed = start.elapsed();

    let avg_us = elapsed.as_micros() as f64 / num_iters as f64;
    let tflops = flops * num_iters as f64 / elapsed.as_secs_f64() / 1e12;
    println!(
        "  matmul-attn: {num_iters} iters in {elapsed:.2?} ({avg_us:.1} us/iter, {tflops:.2} TFLOP/s)"
    );

    // --- Compare outputs ---
    let flash_vec = flash_out.to_vec()?;
    let matmul_vec = matmul_out.to_vec()?;
    let max_diff = flash_vec
        .iter()
        .zip(matmul_vec.iter())
        .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
        .fold(0.0f32, f32::max);
    println!("  max abs diff: {max_diff}");

    Ok(())
}

fn main() -> Result<()> {
    let device = Device::new(0)?;

    // Decode-like: short query, long KV cache.
    run_bench(32, 32, 1, 375, 64, 1000, &device)?;

    // Prefill-like: longer sequences where flash-attn should shine.
    run_bench(4, 32, 512, 512, 64, 1000, &device)?;
    run_bench(4, 32, 1024, 1024, 64, 500, &device)?;
    run_bench(2, 32, 2048, 2048, 64, 200, &device)?;
    run_bench(1, 32, 4096, 4096, 128, 100, &device)?;

    Ok(())
}
