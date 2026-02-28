pub mod ffi;

use std::ffi::c_void;

use xn::cuda_backend::Device;
use xn::{Result, Tensor, WithDType, WithDTypeF};

fn round_multiple(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
}

/// Run multi-head attention using Flash Attention 2.
///
/// Expected tensor layouts (contiguous, row-major):
///   - q: `[batch_size, num_heads_q, seqlen_q, head_dim]`
///   - k: `[batch_size, num_heads_k, seqlen_k, head_dim]`
///   - v: `[batch_size, num_heads_k, seqlen_k, head_dim]`
///
/// Returns the output tensor `[batch_size, num_heads_q, seqlen_q, head_dim]`.
///
/// `num_heads_q` must be a multiple of `num_heads_k` (for grouped-query / multi-query attention).
/// `head_dim` must be 64 or 128.
pub fn flash_attn<T: WithDType + WithDTypeF>(
    q: &Tensor<T, Device>,
    k: &Tensor<T, Device>,
    v: &Tensor<T, Device>,
    softmax_scale: f32,
    is_causal: bool,
) -> Result<Tensor<T, Device>> {
    if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
        xn::bail!("flash_attn: q, k, v must be 4D [batch, heads, seqlen, head_dim]");
    }

    let (batch_size, num_heads_q, seqlen_q, head_dim) = q.dims4()?;
    let (batch_size_k, num_heads_k, seqlen_k, head_dim_k) = k.dims4()?;
    let (batch_size_v, num_heads_v, seqlen_v, head_dim_v) = v.dims4()?;

    if batch_size != batch_size_k || batch_size != batch_size_v {
        xn::bail!("flash_attn: batch size mismatch");
    }
    if head_dim != head_dim_k || head_dim != head_dim_v {
        xn::bail!("flash_attn: head_dim mismatch");
    }
    if num_heads_v != num_heads_k || seqlen_v != seqlen_k {
        xn::bail!("flash_attn: k and v shape mismatch");
    }
    if !num_heads_q.is_multiple_of(num_heads_k) {
        xn::bail!("flash_attn: num_heads_q must be a multiple of num_heads_k");
    }

    let is_bf16 = match T::DTYPE {
        xn::DType::BF16 => true,
        xn::DType::F16 => false,
        _ => xn::bail!("flash_attn: only bf16 and f16 are supported"),
    };

    let device = q.device();

    // Allocate output tensor.
    let o: Tensor<T, Device> =
        Tensor::zeros((batch_size, num_heads_q, seqlen_q, head_dim), device)?;

    // Allocate softmax_lse: [batch_size, num_heads_q, seqlen_q] in f32.
    let softmax_lse: Tensor<f32, Device> =
        Tensor::zeros((batch_size, num_heads_q, seqlen_q), device)?;

    // Rounded dimensions (flash attention internal requirements).
    let d_rounded = round_multiple(head_dim, 32);
    let seqlen_q_rounded = round_multiple(seqlen_q, 128);
    let seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // Strides for contiguous [batch, heads, seqlen, dim] layout.
    let q_batch_stride = (num_heads_q * seqlen_q * head_dim) as i64;
    let k_batch_stride = (num_heads_k * seqlen_k * head_dim) as i64;
    let v_batch_stride = (num_heads_k * seqlen_k * head_dim) as i64;
    let o_batch_stride = (num_heads_q * seqlen_q * head_dim) as i64;

    let q_row_stride = head_dim as i64;
    let k_row_stride = head_dim as i64;
    let v_row_stride = head_dim as i64;
    let o_row_stride = head_dim as i64;

    let q_head_stride = (seqlen_q * head_dim) as i64;
    let k_head_stride = (seqlen_k * head_dim) as i64;
    let v_head_stride = (seqlen_k * head_dim) as i64;
    let o_head_stride = (seqlen_q * head_dim) as i64;

    let total_q = (batch_size * seqlen_q) as i32;

    {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let stream = device.stream();

        let q_s = q.storage()?;
        let k_s = k.storage()?;
        let v_s = v.storage()?;
        let mut o_s = o.storage_mut()?;
        let mut lse_s = softmax_lse.storage_mut()?;

        let (q_ptr, _q_guard) = q_s.data.device_ptr(stream);
        let (k_ptr, _k_guard) = k_s.data.device_ptr(stream);
        let (v_ptr, _v_guard) = v_s.data.device_ptr(stream);
        let (o_ptr, _o_guard) = o_s.data.device_ptr_mut(stream);
        let (lse_ptr, _lse_guard) = lse_s.data.device_ptr_mut(stream);

        let window_size_left = -1i32;
        let window_size_right = if is_causal { 0i32 } else { -1i32 };

        unsafe {
            ffi::run_mha(
                q_ptr as *const c_void,
                k_ptr as *const c_void,
                v_ptr as *const c_void,
                o_ptr as *const c_void,
                lse_ptr as *const c_void,
                std::ptr::null(), // alibi_slopes_ptr
                std::ptr::null(), // cu_seqlens_q
                std::ptr::null(), // cu_seqlens_k
                std::ptr::null(), // seqused_k
                std::ptr::null(), // leftpad_k
                q_batch_stride,
                k_batch_stride,
                v_batch_stride,
                o_batch_stride,
                0, // alibi_slopes_batch_stride
                q_row_stride,
                k_row_stride,
                v_row_stride,
                o_row_stride,
                q_head_stride,
                k_head_stride,
                v_head_stride,
                o_head_stride,
                batch_size as i32,
                num_heads_q as i32,
                num_heads_k as i32,
                head_dim as i32,
                d_rounded as i32,
                softmax_scale,
                seqlen_q as i32,
                seqlen_k as i32,
                seqlen_q_rounded as i32,
                seqlen_k_rounded as i32,
                total_q,
                is_bf16 as i32,
                is_causal as i32,
                1, // is_seqlens_k_cumulative
                0, // unpadded_lse
                0, // seqlenq_ngroups_swapped
                window_size_left,
                window_size_right,
                0.0, // softcap
            );
        }
    }

    Ok(o)
}
