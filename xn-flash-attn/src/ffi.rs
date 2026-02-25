use core::ffi::{c_int, c_void};

unsafe extern "C" {
    pub(crate) fn run_mha(
        q_ptr: *const c_void,
        k_ptr: *const c_void,
        v_ptr: *const c_void,
        o_ptr: *const c_void,
        softmax_lse_ptr: *const c_void,
        alibi_slopes_ptr: *const c_void,

        cu_seqlens_q_ptr: *const i32,
        cu_seqlens_k_ptr: *const i32,
        seqused_k_ptr: *const i32,
        leftpad_k_ptr: *const i32,

        q_batch_stride: i64,
        k_batch_stride: i64,
        v_batch_stride: i64,
        o_batch_stride: i64,
        alibi_slopes_batch_stride: i64,

        q_row_stride: i64,
        k_row_stride: i64,
        v_row_stride: i64,
        o_row_stride: i64,

        q_head_stride: i64,
        k_head_stride: i64,
        v_head_stride: i64,
        o_head_stride: i64,

        b: c_int,
        h: c_int,
        h_k: c_int,
        d: c_int,
        d_rounded: c_int,
        softmax_scale: f32,

        seqlen_q: c_int,
        seqlen_k: c_int,
        seqlen_q_rounded: c_int,
        seqlen_k_rounded: c_int,
        total_q: c_int,

        is_bf16: c_int,
        is_causal: c_int,
        is_seqlens_k_cumulative: c_int,
        unpadded_lse: c_int,
        seqlenq_ngroups_swapped: c_int,

        window_size_left: c_int,
        window_size_right: c_int,

        softcap: f32,
    );

}
