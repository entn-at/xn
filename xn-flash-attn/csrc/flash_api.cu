#include <cstring>
#include <cmath>
#include <cstdint>
#include "flash_fwd_launch_template.h"

using namespace flash;

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
  FP16_SWITCH(!params.is_bf16, [&] {
      // Only instantiate head dims 64 and 128 (the ones we compile kernels for).
      if (params.d <= 64) {
          constexpr static int kHeadDim = 64;
          BOOL_SWITCH(params.is_causal, Is_causal, [&] {
              run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
          });
      } else if (params.d <= 128) {
          constexpr static int kHeadDim = 128;
          BOOL_SWITCH(params.is_causal, Is_causal, [&] {
              run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
          });
      }
  });
}

extern "C" void run_mha(
    void *q_ptr,
    void *k_ptr,
    void *v_ptr,
    void *o_ptr,
    void *softmax_lse_ptr,
    void *alibi_slopes_ptr,

    int32_t *cu_seqlens_q_ptr,
    int32_t *cu_seqlens_k_ptr,
    int32_t *seqused_k_ptr,
    int32_t *leftpad_k_ptr,

    int64_t q_batch_stride,
    int64_t k_batch_stride,
    int64_t v_batch_stride,
    int64_t o_batch_stride,
    int64_t alibi_slopes_batch_stride,

    int64_t q_row_stride,
    int64_t k_row_stride,
    int64_t v_row_stride,
    int64_t o_row_stride,

    int64_t q_head_stride,
    int64_t k_head_stride,
    int64_t v_head_stride,
    int64_t o_head_stride,

    int b,
    int h,
    int h_k,
    int d,
    int d_rounded,
    float softmax_scale,

    int seqlen_q,
    int seqlen_k,
    int seqlen_q_rounded,
    int seqlen_k_rounded,
    int total_q,

    int is_bf16,
    int is_causal,
    int is_seqlens_k_cumulative,
    int unpadded_lse,
    int seqlenq_ngroups_swapped,

    int window_size_left,
    int window_size_right,

    float softcap
) {
    Flash_fwd_params params;
    memset(&params, 0, sizeof(params));

    // Set the pointers.
    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.o_ptr = o_ptr;
    params.oaccum_ptr = nullptr;

    params.p_ptr = nullptr; // used for `return_softmax`
    params.softmax_lse_ptr = softmax_lse_ptr;
    params.softmax_lseaccum_ptr = nullptr;

    params.alibi_slopes_ptr = alibi_slopes_ptr;
    params.alibi_slopes_batch_stride = alibi_slopes_batch_stride;

    // All strides are in elements, not bytes.
    params.q_batch_stride = q_batch_stride;
    params.k_batch_stride = k_batch_stride;
    params.v_batch_stride = v_batch_stride;
    params.o_batch_stride = o_batch_stride;

    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.o_row_stride = o_row_stride;

    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;
    params.o_head_stride = o_head_stride;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_knew = 0;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;
    params.rotary_dim = 0;
    params.total_q = total_q;

    // Set the different scale values.
    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else {
        params.softcap = 0.0;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    // Cumulative sequence length pointers.
    params.cu_seqlens_q = cu_seqlens_q_ptr;
    params.cu_seqlens_k = cu_seqlens_k_ptr;
    params.leftpad_k = leftpad_k_ptr;
    params.seqused_k = seqused_k_ptr;

    // Block-sparse mask.
    params.blockmask = nullptr;

    // KV cache append (knew/vnew).
    params.knew_ptr = nullptr;
    params.vnew_ptr = nullptr;
    params.knew_batch_stride = 0;
    params.vnew_batch_stride = 0;
    params.knew_row_stride = 0;
    params.vnew_row_stride = 0;
    params.knew_head_stride = 0;
    params.vnew_head_stride = 0;

    // Rotary embedding.
    params.rotary_cos_ptr = nullptr;
    params.rotary_sin_ptr = nullptr;
    params.is_rotary_interleaved = false;

    // KV cache indexing.
    params.cache_batch_idx = nullptr;

    // Paged KV cache.
    params.block_table = nullptr;
    params.block_table_batch_stride = 0;
    params.page_block_size = 0;

    // Dropout (disabled: probability to keep = 1).
    params.p_dropout = 1.f;
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;

    // Local window size.
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    // Random state (unused without dropout).
    params.philox_seed = 0;
    params.philox_offset = 0;
    params.rng_state = nullptr;

    // Flags.
    params.is_bf16 = is_bf16;
    params.is_causal = is_causal;
    params.is_seqlens_k_cumulative = is_seqlens_k_cumulative;
    params.num_splits = 1;
    params.unpadded_lse = unpadded_lse;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;

    cudaStream_t stream = 0; // Use the default stream.
    run_mha_fwd(params, stream);
}
