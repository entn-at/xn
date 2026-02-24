//! Basic CUDA example demonstrating tensor operations on GPU.
//!
//! Run with: cargo run --release --example basic_cuda

use xn::{Backend, Result, Tensor, cuda_backend::Device};

fn main() -> Result<()> {
    println!("Initializing CUDA device...");
    let device = Device::new(0)?;
    println!("CUDA device initialized successfully!");

    // Create two matrices for multiplication
    // A = [[1, 2, 3],
    //      [4, 5, 6]]  (2x3)
    let a: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &device)?;

    // B = [[1, 2],
    //      [3, 4],
    //      [5, 6]]  (3x2)
    let b: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], &device)?;

    println!("A shape: {:?}", a.dims());
    println!("B shape: {:?}", b.dims());

    // C = A @ B = [[22, 28],
    //              [49, 64]]  (2x2)
    let c = a.matmul(&b)?;

    println!("C = A @ B");
    println!("C shape: {:?}", c.dims());
    println!("C data: {:?}", c.to_vec()?);
    // Benchmark matmul with bs=32, m=1, n=11264, k=2048
    run_mm_benchmark(32, 1, 11264, 2048, &device)?;
    run_mm_benchmark(1, 32, 11264, 2048, &device)?;

    // Benchmark flash attention vs matmul attention
    run_attn_benchmarks(1, 8, 4096, 8192, 64, &device)?;
    Ok(())
}

fn run_mm_benchmark(bs: usize, m: usize, n: usize, k: usize, device: &Device) -> Result<()> {
    println!("\nBenchmarking matmul ({bs}x{m}x{k}) @ (1x{k}x{n})...");
    let a_data: Vec<half::bf16> =
        (0..bs * m * k).map(|i| half::bf16::from_f32((i % 127) as f32 * 0.01)).collect();
    let b_data: Vec<half::bf16> =
        (0..k * n).map(|i| half::bf16::from_f32((i % 113) as f32 * 0.01)).collect();
    let a: Tensor<half::bf16, Device> = Tensor::from_vec(a_data, (bs, m, k), device)?;
    let b: Tensor<half::bf16, Device> = Tensor::from_vec(b_data, (k, n), device)?;

    // Warmup
    let _warmup = a.matmul(&b)?;
    device.synchronize()?;

    let num_iters = 1000;
    let start = std::time::Instant::now();
    for _ in 0..num_iters {
        let _c = a.matmul(&b)?;
    }
    device.synchronize()?;
    let elapsed = start.elapsed();

    let avg_us = elapsed.as_micros() as f64 / num_iters as f64;
    let flops = 2.0 * bs as f64 * m as f64 * n as f64 * k as f64;
    let tflops = flops * num_iters as f64 / elapsed.as_secs_f64() / 1e12;
    println!("{num_iters} iters in {elapsed:.2?} ({avg_us:.1} us/iter, {tflops:.2} TFLOP/s)");

    Ok(())
}

fn run_attn_benchmarks(
    batch_size: usize,
    num_heads: usize,
    len_q: usize,
    len_kv: usize,
    head_dim: usize,
    device: &Device,
) -> Result<()> {
    use xn::cuda_backend;

    let bs = batch_size * num_heads;
    let q_numel = bs * len_q * head_dim;
    let kv_numel = bs * len_kv * head_dim;

    let q_data: Vec<half::bf16> =
        (0..q_numel).map(|i| half::bf16::from_f32((i % 127) as f32 * 0.01)).collect();
    let k_data: Vec<half::bf16> =
        (0..kv_numel).map(|i| half::bf16::from_f32((i % 113) as f32 * 0.01)).collect();
    let v_data: Vec<half::bf16> =
        (0..kv_numel).map(|i| half::bf16::from_f32((i % 97) as f32 * 0.01)).collect();

    let q: Tensor<half::bf16, Device> = Tensor::from_vec(q_data, (bs, len_q, head_dim), device)?;
    let k: Tensor<half::bf16, Device> = Tensor::from_vec(k_data, (bs, len_kv, head_dim), device)?;
    let v: Tensor<half::bf16, Device> = Tensor::from_vec(v_data, (bs, len_kv, head_dim), device)?;

    // --- Flash attention ---
    println!(
        "\nBenchmarking flash-attn (bs={batch_size}, heads={num_heads}, \
         q_len={len_q}, kv_len={len_kv}, head_dim={head_dim})..."
    );
    let flash_dst: Tensor<half::bf16, Device> =
        Tensor::from_vec(vec![half::bf16::ZERO; q_numel], (bs, len_q, head_dim), device)?;
    {
        let q_s = q.storage()?;
        let k_s = k.storage()?;
        let v_s = v.storage()?;
        let mut dst_s = flash_dst.storage_mut()?;
        cuda_backend::flash_attn(
            &mut dst_s, &q_s, &k_s, &v_s, batch_size, num_heads, len_q, len_kv, head_dim,
        )?;
    }
    device.synchronize()?;

    let num_iters = 100;
    let start = std::time::Instant::now();
    for _ in 0..num_iters {
        let q_s = q.storage()?;
        let k_s = k.storage()?;
        let v_s = v.storage()?;
        let mut dst_s = flash_dst.storage_mut()?;
        cuda_backend::flash_attn(
            &mut dst_s, &q_s, &k_s, &v_s, batch_size, num_heads, len_q, len_kv, head_dim,
        )?;
    }
    device.synchronize()?;
    let elapsed = start.elapsed();

    let avg_us = elapsed.as_micros() as f64 / num_iters as f64;
    let flops =
        4.0 * batch_size as f64 * num_heads as f64 * len_q as f64 * len_kv as f64 * head_dim as f64;
    let tflops = flops * num_iters as f64 / elapsed.as_secs_f64() / 1e12;
    println!("{num_iters} iters in {elapsed:.2?} ({avg_us:.1} us/iter, {tflops:.2} TFLOP/s)");

    // --- Matmul attention ---
    println!(
        "\nBenchmarking matmul-attn (bs={batch_size}, heads={num_heads}, \
         q_len={len_q}, kv_len={len_kv}, head_dim={head_dim})..."
    );
    let scale = half::bf16::from_f32(1.0 / (head_dim as f32).sqrt());

    // Warmup
    let matmul_dst = {
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
    println!("{num_iters} iters in {elapsed:.2?} ({avg_us:.1} us/iter, {tflops:.2} TFLOP/s)");

    // --- Compare outputs ---
    let flash_vec = flash_dst.to_vec()?;
    let matmul_vec = matmul_dst.to_vec()?;
    let max_diff = flash_vec
        .iter()
        .zip(matmul_vec.iter())
        .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
        .fold(0.0f32, f32::max);
    println!("\nMax abs difference (flash vs matmul): {max_diff}");

    Ok(())
}
