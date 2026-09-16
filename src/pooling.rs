use candle_core::{DType, Result, Tensor};

/// Mean-pool embeddings over the token axis, masking padding tokens.
/// An all-padding row yields a zero vector (never NaN/Inf), matching
/// sentence-transformers' `include_padding_embeddings=false` semantics
/// for real inputs while being NaN-safe for degenerate all-padding rows.
pub fn mask_mean_pool(embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let mask = attention_mask.to_dtype(DType::F32)?.unsqueeze(2)?;
    let sum_mask = mask.sum(1)?;
    let summed = (embeddings.broadcast_mul(&mask)?).sum(1)?;
    // Guard: rows whose token count is 0 must produce zeros.
    let safe = {
        let dims = sum_mask.dims();
        let zero = Tensor::zeros(dims, DType::F32, sum_mask.device())?;
        let one = Tensor::ones(dims, DType::F32, sum_mask.device())?;
        let nonempty = sum_mask.ne(&zero)?;
        let denom = sum_mask.broadcast_maximum(&one)?;
        let keep = nonempty.to_dtype(DType::F32)?;
        summed.broadcast_mul(&keep)?.broadcast_div(&denom)?
    };
    Ok(safe)
}

/// L2-normalize the last axis of a 2D tensor.
pub fn l2_normalize(v: &Tensor) -> Result<Tensor> {
    v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    fn t(data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape, &Device::Cpu).unwrap()
    }

    #[test]
    fn mask_mean_pool_hand_computed() {
        // batch=2, tokens=3, hidden=2
        let emb = t(
            &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            &[2, 3, 2],
        );
        let mask = t(&[1., 1., 1., 1., 0., 0.], &[2, 3]);
        let pooled = super::mask_mean_pool(&emb, &mask).unwrap();
        // row0: mean([1,2],[3,4],[5,6]) = [3,4]
        // row1: mean([7,8], masked rest) = [7,8]
        let got: Vec<f32> = pooled.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(got, vec![3., 4., 7., 8.]);
    }

    #[test]
    fn mask_mean_pool_zero_rows_are_zero_not_nan() {
        let emb = t(
            &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            &[2, 3, 2],
        );
        // row1 keeps only its first token so pooled = [7,8]
        let mask = t(&[0., 0., 0., 1., 0., 0.], &[2, 3]);
        let pooled = super::mask_mean_pool(&emb, &mask).unwrap();
        let got: Vec<f32> = pooled.flatten_all().unwrap().to_vec1().unwrap();
        // all-padding row must be zeros, not NaN/Inf
        assert!(got[0].is_finite() && got[1] == 0.0);
        assert_eq!(&got[2..], &[7., 8.]);
    }

    #[test]
    fn l2_normalize_unit_length() {
        let v = t(&[3., 4., 0.], &[1, 3]);
        let n = super::l2_normalize(&v).unwrap();
        let got: Vec<f32> = n.flatten_all().unwrap().to_vec1().unwrap();
        let len = (got[0] * got[0] + got[1] * got[1] + got[2] * got[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-6);
        assert!((got[0] - 0.6).abs() < 1e-6 && (got[1] - 0.8).abs() < 1e-6);
    }
}
