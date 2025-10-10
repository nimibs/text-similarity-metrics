use std::fmt;

#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;

#[derive(Debug, Clone, PartialEq)]
pub enum CosineSimilarityError {
    DifferentLengths { len1: usize, len2: usize },
    ZeroVector,
}

impl fmt::Display for CosineSimilarityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CosineSimilarityError::DifferentLengths { len1, len2 } => {
                write!(f, "Vectors have different lengths: {} vs {}", len1, len2)
            }
            CosineSimilarityError::ZeroVector => {
                write!(f, "Cannot compute cosine similarity with zero vector")
            }
        }
    }
}

impl std::error::Error for CosineSimilarityError {}

// Computes cosine similarity between two vectors, normalized to [0, 1] range.
pub fn cosine_similarity(vec1: &[f64], vec2: &[f64]) -> Result<f64, CosineSimilarityError> {
    if vec1.len() != vec2.len() {
        return Err(CosineSimilarityError::DifferentLengths {
            len1: vec1.len(),
            len2: vec2.len(),
        });
    }

    let (dot_product, norm1, norm2) = {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                unsafe { compute_dot_and_norms_avx512(vec1, vec2) }
            } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe { compute_dot_and_norms_avx2(vec1, vec2) }
            } else {
                compute_dot_and_norms_scalar(vec1, vec2)
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if is_aarch64_feature_detected!("neon") {
                unsafe { compute_dot_and_norms_neon(vec1, vec2) }
            } else {
                compute_dot_and_norms_scalar(vec1, vec2)
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            compute_dot_and_norms_scalar(vec1, vec2)
        }
    };

    // Check for zero vectors
    if norm1 == 0.0 || norm2 == 0.0 {
        return Err(CosineSimilarityError::ZeroVector);
    }

    // Compute cosine similarity: dot(A, B) / (||A|| * ||B||)
    let norm_mul = norm1 * norm2;
    let norms_sq = if norm_mul.is_finite() {
        norm_mul.sqrt()
    } else {
        norm1.sqrt() * norm2.sqrt()
    };
    let cosine = dot_product / (norms_sq);

    let similarity = (cosine + 1.0) / 2.0;

    Ok(similarity)
}

#[inline]
fn compute_dot_and_norms_scalar(vec1: &[f64], vec2: &[f64]) -> (f64, f64, f64) {
    let mut dot_product = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;

    for i in 0..vec1.len() {
        dot_product += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }

    (dot_product, norm1, norm2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn compute_dot_and_norms_avx2(vec1: &[f64], vec2: &[f64]) -> (f64, f64, f64) {
    use std::arch::x86_64::*;

    let len = vec1.len();
    let simd_len = len / 4 * 4;

    let mut dot_sum = _mm256_setzero_pd();
    let mut norm1_sum = _mm256_setzero_pd();
    let mut norm2_sum = _mm256_setzero_pd();

    // Process 4 f64 values at a time
    let mut i = 0;
    while i < simd_len {
        let v1 = _mm256_loadu_pd(vec1.as_ptr().add(i));
        let v2 = _mm256_loadu_pd(vec2.as_ptr().add(i));

        dot_sum = _mm256_fmadd_pd(v1, v2, dot_sum);
        norm1_sum = _mm256_fmadd_pd(v1, v1, norm1_sum);
        norm2_sum = _mm256_fmadd_pd(v2, v2, norm2_sum);

        i += 4;
    }

    // Horizontal sum of the SIMD accumulators
    let mut dot_product = horizontal_sum_avx2(dot_sum);
    let mut norm1 = horizontal_sum_avx2(norm1_sum);
    let mut norm2 = horizontal_sum_avx2(norm2_sum);

    // Handle remaining elements
    while i < len {
        dot_product += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
        i += 1;
    }

    (dot_product, norm1, norm2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_sum_avx2(v: __m256d) -> f64 {
    let arr = [0.0f64; 4];
    _mm256_storeu_pd(arr.as_ptr() as *mut f64, v);
    arr[0] + arr[1] + arr[2] + arr[3]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn compute_dot_and_norms_avx512(vec1: &[f64], vec2: &[f64]) -> (f64, f64, f64) {
    use std::arch::x86_64::*;

    let len = vec1.len();
    let simd_len = len / 8 * 8;

    let mut dot_sum = _mm512_setzero_pd();
    let mut norm1_sum = _mm512_setzero_pd();
    let mut norm2_sum = _mm512_setzero_pd();

    // Process 8 f64 values at a time
    let mut i = 0;
    while i < simd_len {
        let v1 = _mm512_loadu_pd(vec1.as_ptr().add(i));
        let v2 = _mm512_loadu_pd(vec2.as_ptr().add(i));

        dot_sum = _mm512_fmadd_pd(v1, v2, dot_sum);
        norm1_sum = _mm512_fmadd_pd(v1, v1, norm1_sum);
        norm2_sum = _mm512_fmadd_pd(v2, v2, norm2_sum);

        i += 8;
    }

    // Horizontal sum of the SIMD accumulators
    let mut dot_product = _mm512_reduce_add_pd(dot_sum);
    let mut norm1 = _mm512_reduce_add_pd(norm1_sum);
    let mut norm2 = _mm512_reduce_add_pd(norm2_sum);

    // Handle remaining elements
    while i < len {
        dot_product += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
        i += 1;
    }

    (dot_product, norm1, norm2)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn compute_dot_and_norms_neon(vec1: &[f64], vec2: &[f64]) -> (f64, f64, f64) {
    use std::arch::aarch64::*;

    let len = vec1.len();
    let simd_len = len / 2 * 2;

    unsafe {
        let mut dot_sum = vdupq_n_f64(0.0);
        let mut norm1_sum = vdupq_n_f64(0.0);
        let mut norm2_sum = vdupq_n_f64(0.0);

        // Process 2 f64 values at a time
        let mut i = 0;
        while i < simd_len {
            let v1 = vld1q_f64(vec1.as_ptr().add(i));
            let v2 = vld1q_f64(vec2.as_ptr().add(i));

            dot_sum = vfmaq_f64(dot_sum, v1, v2);
            norm1_sum = vfmaq_f64(norm1_sum, v1, v1);
            norm2_sum = vfmaq_f64(norm2_sum, v2, v2);

            i += 2;
        }

        // Horizontal sum of the SIMD accumulators
        let mut dot_product = vaddvq_f64(dot_sum);
        let mut norm1 = vaddvq_f64(norm1_sum);
        let mut norm2 = vaddvq_f64(norm2_sum);

        // Handle remaining elements
        while i < len {
            dot_product += vec1[i] * vec2[i];
            norm1 += vec1[i] * vec1[i];
            norm2 += vec2[i] * vec2[i];
            i += 1;
        }

        (dot_product, norm1, norm2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_vectors() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![1.0, 2.0, 3.0];
        let result = cosine_similarity(&vec1, &vec2).unwrap();
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_scaled_vectors() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![2.0, 4.0, 6.0];
        let result = cosine_similarity(&vec1, &vec2).unwrap();
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_orthogonal_vectors() {
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![0.0, 1.0, 0.0];
        let result = cosine_similarity(&vec1, &vec2).unwrap();
        assert_eq!(result, 0.5);
    }

    #[test]
    fn test_opposite_vectors() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![-1.0, -2.0, -3.0];
        let result = cosine_similarity(&vec1, &vec2).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_different_lengths_error() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![1.0, 2.0];
        let result = cosine_similarity(&vec1, &vec2);
        assert!(matches!(
            result,
            Err(CosineSimilarityError::DifferentLengths { len1: 3, len2: 2 })
        ));
    }

    #[test]
    fn test_zero_vector_error() {
        let vec1 = vec![0.0, 0.0, 0.0];
        let vec2 = vec![1.0, 2.0, 3.0];
        let result = cosine_similarity(&vec1, &vec2);
        assert!(matches!(result, Err(CosineSimilarityError::ZeroVector)));
    }

    #[test]
    fn test_both_zero_vectors_error() {
        let vec1 = vec![0.0, 0.0, 0.0];
        let vec2 = vec![0.0, 0.0, 0.0];
        let result = cosine_similarity(&vec1, &vec2);
        assert!(matches!(result, Err(CosineSimilarityError::ZeroVector)));
    }

    #[test]
    fn test_empty_vectors_error() {
        let vec1: Vec<f64> = vec![];
        let vec2: Vec<f64> = vec![];
        let result = cosine_similarity(&vec1, &vec2);
        assert!(matches!(result, Err(CosineSimilarityError::ZeroVector)));
    }

    #[test]
    fn test_partial_overlap() {
        let vec1 = vec![1.0, 1.0, 0.0];
        let vec2 = vec![1.0, 0.0, 1.0];
        let result = cosine_similarity(&vec1, &vec2).unwrap();
        assert_eq!(result, 0.75);
    }

    #[test]
    fn test_mixed_signs() {
        let vec1 = vec![1.0, -1.0, 1.0];
        let vec2 = vec![1.0, -1.0, 1.0];
        let result = cosine_similarity(&vec1, &vec2).unwrap();
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_large_vectors() {
        let vec1: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let vec2: Vec<f64> = (0..1000).map(|i| i as f64 * 2.0).collect();
        let result = cosine_similarity(&vec1, &vec2).unwrap();
        assert_eq!(result, 1.0);
    }
}
