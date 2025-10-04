use crate::hash_iterators::{CharHasher, HashIterator,
    NgramHashIterator, NgramHashIteratorBuilder,
};
use crate::metrics::shared::{hash_counts, intersection_count};
use crate::utils::assert_n_le_32;

pub fn rouge_n_similarity<const N: usize>(
    expected: &str,
    actual: &str,
    case_sensitive: bool,
) -> f64 {
    assert_n_le_32(N);

    if case_sensitive {
        let expected_ngram_iter = NgramHashIteratorBuilder::from_str_case_sensitive::<N>(expected);
        let actual_ngram_iter = NgramHashIteratorBuilder::from_str_case_sensitive::<N>(actual);
        rouge_n_similarity_impl::<_, N>(expected_ngram_iter, actual_ngram_iter)
    } else {
        let expected_ngram_iter = NgramHashIteratorBuilder::from_str_case_insensitive::<N>(expected);
        let actual_ngram_iter = NgramHashIteratorBuilder::from_str_case_insensitive::<N>(actual);
        rouge_n_similarity_impl::<_, N>(expected_ngram_iter, actual_ngram_iter)
    }
}

fn rouge_n_similarity_impl<'a, H: CharHasher, const N: usize>(
    expected: NgramHashIterator<HashIterator<'a, H>, N>,
    actual: NgramHashIterator<HashIterator<'a, H>, N>,
) -> f64 {
    assert_n_le_32(N);

    let mut expected_ngram_iter = expected;
    let mut actual_ngram_iter = actual;

    let expected_counts = hash_counts(&mut expected_ngram_iter);
    let actual_counts = hash_counts(&mut actual_ngram_iter);

    let intersection_count =
        intersection_count(&expected_counts.hash_counts, &actual_counts.hash_counts);

    let precision = if actual_counts.count > 0 {
        intersection_count as f64 / actual_counts.count as f64
    } else {
        0f64
    };
    let recall = if expected_counts.count > 0 {
        intersection_count as f64 / expected_counts.count as f64
    } else {
        0f64
    };

    let f1 = if precision + recall > 0f64 {
        (2f64 * precision * recall) / (precision + recall)
    } else {
        0f64
    };

    f1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_strings() {
        let result = rouge_n_similarity::<1>("hello world", "hello world", false);
        assert_eq!(result, 1.0);

        let result = rouge_n_similarity::<2>("hello world", "hello world", false);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_completely_different_strings() {
        let result = rouge_n_similarity::<1>("hello world", "goodbye universe", false);
        assert_eq!(result, 0.0);

        let result = rouge_n_similarity::<2>("hello world", "goodbye universe", false);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_partial_overlap() {
        // Precision: 2/3, Recall: 2/3, F1: 2/3
        let result = rouge_n_similarity::<1>("hello world test", "hello world example", false);
        assert_eq!(result, 2.0 / 3.0);

        // For N=2: common bigram is "hello world" (1 out of 2 in each)
        // Precision: 1/2, Recall: 1/2, F1: 1/2 = 0.5
        let result = rouge_n_similarity::<2>("hello world test", "hello world example", false);
        assert_eq!(result, 0.5);
    }

    #[test]
    fn test_empty_strings() {
        let result = rouge_n_similarity::<1>("", "", false);
        assert_eq!(result, 0.0);

        let result = rouge_n_similarity::<2>("", "", false);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_one_empty_string() {
        let result = rouge_n_similarity::<1>("hello", "", false);
        assert_eq!(result, 0.0);

        let result = rouge_n_similarity::<1>("", "hello", false);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_rouge_1() {
        let result = rouge_n_similarity::<1>("the cat sat", "the dog ran", false);

        // Precision: 1/3, Recall: 1/3, F1: 1/3
        assert_eq!(result, 1.0 / 3.0);
    }

    #[test]
    fn test_rouge_2() {
        let result = rouge_n_similarity::<2>("the cat sat on", "ran the cat fast", false);

        // Precision: 1/3, Recall: 1/3, F1: 1/3
        assert_eq!(result, 1.0 / 3.0);
    }

    #[test]
    fn test_rouge_3() {
        let result = rouge_n_similarity::<3>("the cat sat on mat", "the cat sat on floor", false);

        // Precision: 2/3, Recall: 2/3, F1: 2/3
        assert_eq!(result, 2.0 / 3.0);
    }

    #[test]
    fn test_word_order_matters() {
        let result1 = rouge_n_similarity::<2>("the cat sat", "the cat sat", false);
        let result2 = rouge_n_similarity::<2>("the cat sat", "sat the cat", false);

        assert_eq!(result1, 1.0);

        // Precision: 1/2, Recall: 1/2, F1: 1/2 = 0.5
        assert_eq!(result2, 0.5);

        let result3 = rouge_n_similarity::<3>("the cat sat on", "sat on the cat", false);
        // No common trigrams, so should be 0.0
        assert_eq!(result3, 0.0);
    }

    #[test]
    fn test_repeated_words() {
        let result = rouge_n_similarity::<1>("hello hello world", "hello world world", false);

        // Precision: 2/3, Recall: 2/3, F1: 2/3
        assert_eq!(result, 2.0 / 3.0);
    }

    // Case Sensitivity Tests
    #[test]
    fn test_case_insensitive_basic() {
        let result = rouge_n_similarity::<1>("Hello World", "hello world", false);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_case_sensitive_basic() {
        let result = rouge_n_similarity::<1>("Hello", "hello", true);
        assert_eq!(result, 0.0);
    }

    // Unicode Tests
    #[test]
    fn test_unicode_basic() {
        let result = rouge_n_similarity::<1>("café naïve", "café naïve", false);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_unicode_case_folding() {
        let result = rouge_n_similarity::<1>("Café", "café", false);
        assert_eq!(result, 1.0);
    }
}
