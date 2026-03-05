use crate::{Backend, Dim, Result, Tensor, WithDTypeF};

/// Sample according to the Gumbel-Softmax distribution.
pub fn gumbel_softmax<T: WithDTypeF, B: Backend, D: Dim>(
    logits: &Tensor<T, B>,
    temperature: f32,
    dim: D,
) -> Result<Tensor<i64, B>> {
    if temperature <= 0.0 {
        logits.argmax(dim)
    } else {
        // Cast to f32, doing the Gumbel softmax in bf16 is a bit unstable.
        let logits = logits.to::<f32>()?;
        let rand_uniform = if temperature == 1.0 {
            logits.rand_uniform_like(1e-7, 0.999)?
        } else {
            logits.rand_uniform_like(1e-7, 0.999 * temperature)?
        };
        let minus_g = rand_uniform.log()?.neg()?.log()?;
        logits.sub(&minus_g)?.argmax(dim)
    }
}
