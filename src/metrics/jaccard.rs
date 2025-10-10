//! Jaccard similarity metric for n-grams.

use crate::{
    hash_iterators::{CharHasher, HashIterator, NgramHashIterator, NgramHashIteratorBuilder},
    utils::assert_n_le_32,
};
use nohash_hasher::BuildNoHashHasher;
type NoHasherHashSet = std::collections::HashSet<u64, BuildNoHashHasher<u64>>;

/// Computes Jaccard similarity between two text strings using character n-grams.
///
/// Measures the similarity between sets of n-grams by computing |A ∩ B| / |A ∪ B|.
///
/// # Arguments
///
/// * `N` - The n-gram size (must be ≤ 32)
/// * `expected` - First text string
/// * `actual` - Second text string
/// * `case_sensitive` - Whether to perform case-sensitive comparison
///
/// # Returns
///
/// Similarity score in [0, 1]: 1.0 = identical, 0.0 = no overlap
///
/// # Example
///
/// ```
/// use text_similarity_metrics::jaccard_n_similarity;
///
/// let score = jaccard_n_similarity::<1>("hello world", "hello rust", false);
/// assert_eq!(score, 1.0 / 3.0); // Common: "hello", " " | Union: all unique chars
/// ```
pub fn jaccard_n_similarity<const N: usize>(
    expected: &str,
    actual: &str,
    case_sensitive: bool,
) -> f64 {
    assert_n_le_32(N);

    if case_sensitive {
        let expected_ngram_iter = NgramHashIteratorBuilder::from_str_case_sensitive::<N>(expected);
        let actual_ngram_iter = NgramHashIteratorBuilder::from_str_case_sensitive::<N>(actual);
        jaccard_n_similarity_impl::<_, N>(expected_ngram_iter, actual_ngram_iter)
    } else {
        let expected_ngram_iter =
            NgramHashIteratorBuilder::from_str_case_insensitive::<N>(expected);
        let actual_ngram_iter = NgramHashIteratorBuilder::from_str_case_insensitive::<N>(actual);
        jaccard_n_similarity_impl::<_, N>(expected_ngram_iter, actual_ngram_iter)
    }
}

fn jaccard_n_similarity_impl<'a, H: CharHasher, const N: usize>(
    expected: NgramHashIterator<HashIterator<'a, H>, N>,
    actual: NgramHashIterator<HashIterator<'a, H>, N>,
) -> f64 {
    assert_n_le_32(N);

    let mut expected_ngram_iter = expected;
    let mut actual_ngram_iter = actual;

    let expected_set = create_hash_set(&mut expected_ngram_iter);
    let actual_set = create_hash_set(&mut actual_ngram_iter);

    let mut intersect = 0;
    let mut union = 0;

    for h in expected_set.iter() {
        if actual_set.contains(h) {
            intersect += 1;
        }

        union += 1;
    }

    for h in actual_set.iter() {
        if !expected_set.contains(h) {
            union += 1;
        }
    }

    let jaccard = if union != 0 {
        (intersect as f64) / (union as f64)
    } else {
        0f64
    };
    jaccard
}

fn create_hash_set<'a, H: CharHasher, const N: usize>(
    iter: &mut NgramHashIterator<HashIterator<'a, H>, N>,
) -> NoHasherHashSet {
    let mut set = NoHasherHashSet::default();

    while let Some(h) = iter.next() {
        set.insert(h);
    }

    set
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_texts_unigrams() {
        let result = jaccard_n_similarity::<1>("hello world", "hello world", false);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_identical_texts_bigrams() {
        let result = jaccard_n_similarity::<2>("hello world rust", "hello world rust", false);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_no_overlap_unigrams() {
        let result = jaccard_n_similarity::<1>("hello world", "foo bar", false);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_no_overlap_bigrams() {
        let result = jaccard_n_similarity::<2>("hello world", "foo bar baz", false);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_partial_overlap_unigrams() {
        // Jaccard = 1/3 ≈ 0.333...
        let result = jaccard_n_similarity::<1>("hello world", "hello rust", false);
        assert_eq!(result, 1.0 / 3.0);
    }

    #[test]
    fn test_partial_overlap_bigrams() {
        // Jaccard = 1/3
        let result = jaccard_n_similarity::<2>("hello world rust", "world rust test", false);
        assert_eq!(result, 1.0 / 3.0);
    }

    #[test]
    fn test_empty_strings() {
        let result = jaccard_n_similarity::<1>("", "", false);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_one_empty_string() {
        let result = jaccard_n_similarity::<1>("hello world", "", false);
        assert_eq!(result, 0.0);

        let result = jaccard_n_similarity::<1>("", "hello world", false);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_insufficient_words_for_ngrams() {
        let result = jaccard_n_similarity::<2>("hello", "world", false);
        assert_eq!(result, 0.0);

        let result = jaccard_n_similarity::<3>("hello world", "foo bar", false);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_order_sensitivity() {
        // Jaccard = 0/4 = 0.0
        let result1 = jaccard_n_similarity::<2>("hello world rust", "rust world hello", false);
        let result2 = jaccard_n_similarity::<2>("hello world rust", "hello world rust", false);

        assert_eq!(result1, 0.0);
        assert_eq!(result2, 1.0);
        assert!(result1 < result2);
    }

    #[test]
    fn test_symmetry_property() {
        let text1 = "hello world rust test";
        let text2 = "world test hello programming";

        let result1 = jaccard_n_similarity::<1>(text1, text2, false);
        let result2 = jaccard_n_similarity::<1>(text2, text1, false);

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_known_calculation_unigrams() {
        let result = jaccard_n_similarity::<1>("the quick brown", "the brown fox", false);
        assert_eq!(result, 0.5);
    }

    #[test]
    fn test_trigrams() {
        let text1 = "the quick brown fox jumps";
        let text2 = "the quick brown cat runs";

        // Jaccard = 1/5 = 0.2
        let result = jaccard_n_similarity::<3>(text1, text2, false);
        assert_eq!(result, 0.2);
    }

    #[test]
    fn test_repeated_words() {
        let result = jaccard_n_similarity::<1>("hello hello world", "hello world world", false);
        assert_eq!(result, 1.0);
    }

    // Case Sensitivity Tests
    #[test]
    fn test_case_insensitive_basic() {
        let result = jaccard_n_similarity::<1>("Hello World", "hello world", false);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_case_sensitive_basic() {
        let result = jaccard_n_similarity::<1>("Hello", "hello", true);
        assert_eq!(result, 0.0);
    }

    // Unicode Tests
    #[test]
    fn test_unicode_basic() {
        let result = jaccard_n_similarity::<1>("café naïve", "café naïve", false);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_unicode_case_folding() {
        let result = jaccard_n_similarity::<1>("Café", "café", false);
        assert_eq!(result, 1.0);
    }
}
