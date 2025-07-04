use crate::error::ColbertError;
use candle_core::Tensor;

/// Normalizes a tensor using L2 normalization along the last dimension.
pub fn normalize_l2(v: &Tensor) -> Result<Tensor, ColbertError> {
    let norm_l2 = v.sqr()?.sum_keepdim(v.rank() - 1)?.sqrt()?;
    v.broadcast_div(&norm_l2).map_err(ColbertError::from)
}
