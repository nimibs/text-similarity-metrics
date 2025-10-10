//! BLEU (Bilingual Evaluation Understudy) similarity metric.

use crate::{hash_iterators::{
    CaseInsensitiveCharHasher, CaseSensitiveCharHasher, CharHasher, NgramHashIteratorBuilder,
}, metrics::shared::{hash_counts, intersection_count}};

/// Computes BLEU similarity between two text strings.
///
/// BLEU measures n-gram precision with a brevity penalty. Uses up to 4-grams
/// and penalizes candidates shorter than the reference.
///
/// # Arguments
///
/// * `expected` - Reference text
/// * `actual` - Candidate text to evaluate
/// * `case_sensitive` - Whether to perform case-sensitive comparison
///
/// # Returns
///
/// Similarity score in [0, 1]: 1.0 = perfect match, 0.0 = no match
///
/// # Example
///
/// ```
/// use text_similarity_metrics::bleu_similarity;
///
/// let score = bleu_similarity("the cat sat", "the cat sat", false);
/// assert_eq!(score, 1.0);
/// ```
pub fn bleu_similarity(expected: &str, actual: &str, case_sensitive: bool) -> f64 {
    if case_sensitive {
        bleu_internal(expected, actual, CaseSensitiveCharHasher)
    } else {
        bleu_internal(expected, actual, CaseInsensitiveCharHasher)
    }
}

fn bleu_internal<'a, H: CharHasher + Copy>(expected: &str, actual: &str, hasher: H) -> f64 {
    let expected_hashes_iter : Vec<u64> =
        NgramHashIteratorBuilder::from_str_with_hasher::<H, 1>(expected, hasher).collect();
    let actual_hashes_iter: Vec<u64> =
        NgramHashIteratorBuilder::from_str_with_hasher::<H,1>(actual, hasher).collect();

    let expected_len: f64 = expected.len() as f64;
    let actual_len: f64 = actual.len() as f64;

    if expected_len == 0f64 {
        if actual.len() == 0 {
            return 1f64;
        }

        return 0f64;
    }

    let bp = if actual.len() > expected.len() { 1f64 } else { (1f64 - (expected_len/ actual_len)).exp() };

    // Determine how many n-gram levels we can actually use based on text length
    let max_n = expected_hashes_iter.len().min(actual_hashes_iter.len()).min(4);
    
    if max_n == 0 {
        return 0f64;
    }

    let weight = 1.0f64 / max_n as f64;
    let mut bleu = 0f64;
    let mut non_empty_inter = false;

    // verify we have enough ngrams before we compute the ngram term.
    if max_n >= 1 {
        bleu += weight * ln_precision::<1>(&expected_hashes_iter, &actual_hashes_iter, &mut non_empty_inter);
    }
    if max_n >= 2 {
        bleu += weight * ln_precision::<2>(&expected_hashes_iter, &actual_hashes_iter, &mut non_empty_inter);
    }
    if max_n >= 3 {
        bleu += weight * ln_precision::<3>(&expected_hashes_iter, &actual_hashes_iter, &mut non_empty_inter);
    }
    if max_n >= 4 {
        bleu += weight * ln_precision::<4>(&expected_hashes_iter, &actual_hashes_iter, &mut non_empty_inter);
    }

    // if no intersection found in all ngra,s return flat 0.
    bleu = if non_empty_inter {bp * bleu.exp()} else {0f64};
    
    bleu
}

fn ln_precision<const N: usize>(expected_hashes: &Vec<u64>, actual_hashes: &Vec<u64>, found_inter: &mut bool) -> f64 {
    let expected_iter = expected_hashes.iter().copied();
    let actual_iter = actual_hashes.iter().copied();
    
    let mut expected_ngram_hash_iter = NgramHashIteratorBuilder::from_iter::<_,N>(expected_iter);
    let mut actual_ngram_hash_iter = NgramHashIteratorBuilder::from_iter::<_,N>(actual_iter);


    let expected_hash_counts = hash_counts(&mut expected_ngram_hash_iter);
    let actual_hash_counts = hash_counts(&mut actual_ngram_hash_iter);

    // actual_hash_counts.count cannot be 0, as we verified we have enough ngrams before calling this method.
    let intersection_count = intersection_count(&expected_hash_counts.hash_counts, &actual_hash_counts.hash_counts);
    let mut precision = intersection_count as f64 / actual_hash_counts.count as f64;

    // if precision is 0, we penalize according to the length of the actual hash_counts.
    if precision == 0f64 {
        precision = 1.0 / (2.0 * actual_hash_counts.count as f64)
    }

    if intersection_count > 0 {
        *found_inter = true;
    }

    precision.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Basic Matching Tests
    #[test]
    fn test_single_word_match() {
        let result = bleu_similarity("hello", "hello", false);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_single_word_no_match() {
        let result = bleu_similarity("hello", "world", false);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_multi_word_complete_match() {
        let result = bleu_similarity("the quick brown fox jumps", "the quick brown fox jumps", false);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_multi_word_complete_no_match() {
        let result = bleu_similarity("the quick brown fox", "lazy dog jumps high", false);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_both_empty_strings() {
        let result = bleu_similarity("", "", false);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_expected_empty_actual_not() {
        let result = bleu_similarity("", "hello", false);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_actual_empty_expected_not() {
        let result = bleu_similarity("hello", "", false);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_three_words_match() {
        let result = bleu_similarity("hello world test", "hello world test", false);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_brevity_penalty_actual_shorter() {
        let result = bleu_similarity("the quick brown fox jumps", "the quick brown", false);
        assert!(result < 1.0);
        assert!(result > 0.0);
    }

    #[test]
    fn test_brevity_penalty_actual_longer() {
        let result = bleu_similarity("the quick brown", "the quick brown fox jumps", false);
        assert!(result > 0.4);
    }

    #[test]
    fn test_brevity_penalty_significantly_shorter() {
        let result = bleu_similarity("the quick brown fox jumps over", "the", false);
        assert!(result < 0.1);
    }

    #[test]
    fn test_precision_based_extra_content() {
        let result_extra = bleu_similarity("the cat sat", "the cat sat on the mat happily", false);
        let result_missing = bleu_similarity("the cat sat on the mat happily", "the cat sat", false);
        
        assert!(result_extra > result_missing);
    }

    #[test]
    fn test_partial_ngram_match_longer() {
        let result = bleu_similarity("the quick brown fox", "the quick brown dog", false);
        assert!(result > 0.0);
        assert!(result < 1.0);
    }

    #[test]
    fn test_partial_overlap_longer_strings() {
        let result = bleu_similarity("the cat sat on mat", "the dog sat on mat", false);
        assert!(result > 0.0);
        assert!(result < 1.0);
    }

    #[test]
    fn test_word_based_spacing_doesnt_matter() {
        let result = bleu_similarity("hello   world", "hello world", false);
        // Spacing affects character count in brevity penalty
        assert!(result > 0.8);
        assert!(result < 1.0);
    }

    #[test]
    fn test_word_based_different_words() {
        let result = bleu_similarity("helloworld", "hello world", false);
        assert_eq!(result, 0.0);
    }

    // Case Sensitivity Tests
    #[test]
    fn test_case_insensitive() {
        let result = bleu_similarity("The Quick Brown Fox", "the quick brown fox", false);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_case_sensitive() {
        let result = bleu_similarity("Hello World Test", "hello world test", true);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_case_sensitive_match() {
        let result = bleu_similarity("Hello World Test", "Hello World Test", true);
        assert_eq!(result, 1.0);
    }

    // Unicode Tests
    #[test]
    fn test_unicode_basic() {
        let result = bleu_similarity("café naïve résumé", "café naïve résumé", false);
        assert_eq!(result, 1.0);
    }
}
