use std::fmt;

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

    // Compute dot product and norms in a single pass
    let mut dot_product = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;

    for i in 0..vec1.len() {
        let v1 = vec1[i];
        let v2 = vec2[i];

        dot_product += v1 * v2;
        norm1 += v1 * v1;
        norm2 += v2 * v2;
    }

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
