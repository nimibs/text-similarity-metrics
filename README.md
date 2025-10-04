# Text Similarity Metrics

A high-performance Rust library for computing text similarity using multiple algorithms. Optimized for speed with rolling hash techniques, single-pass processing, and efficient memory usage.

## Features

- 🚀 **Fast**: 
    - Rolling hash algorithm for O(M) n-gram computation (M = text length)
    - Uses a fast FNV-1a hash for word hashing.
    - Memory efficient with iterator based design.
- 📊 **Multiple Metrics**: BLEU, Jaccard, ROUGE-N, and Cosine Similarity algorithms
- 🔤 **Unicode Support**: Full Unicode support with proper case folding
- ⚙️ **Flexible**: Case-sensitive or case-insensitive comparison
- 🎯 **Generic N-grams**: Compile-time n-gram size specification (1-gram to 32-gram)

## Algorithms

### BLEU (Bilingual Evaluation Understudy)

A precision-based metric originally designed for machine translation evaluation. Measures how much of the generated text appears in the reference text.

**Key characteristics:**
- Uses geometric mean of 1-gram through 4-gram precision
- Includes brevity penalty for shorter texts
- Precision-oriented (penalizes extra content less than missing content)
- Range: [0, 1] where 1 = identical, 0 = no overlap


```rust
use text_similarity_metrics::bleu_similarity;

let reference = "the quick brown fox jumps over the lazy dog";
let candidate = "the quick brown fox jumps over a lazy dog";

let score = bleu_similarity(reference, candidate, false); // case-insensitive
println!("BLEU score: {}", score);
```

### Jaccard Similarity

A set-based similarity metric that measures the intersection over union of n-grams.

**Formula:** `|A ∩ B| / |A ∪ B|`

**Key characteristics:**
- Order-insensitive (treats text as a set of n-grams)
- Range: [0, 1] where 1 = identical, 0 = no overlap


```rust
use text_similarity_metrics::jaccard_n_similarity;

let text1 = "the quick brown fox";
let text2 = "the lazy brown dog";

// Unigram (word-level) Jaccard similarity
let score = jaccard_n_similarity::<1>(text1, text2, false);
println!("Jaccard-1 score: {}", score);

// Bigram Jaccard similarity
let score = jaccard_n_similarity::<2>(text1, text2, false);
println!("Jaccard-2 score: {}", score);
```

### ROUGE-N (Recall-Oriented Understudy for Gisting Evaluation)

A recall-based metric that measures how much of the reference text appears in the generated text.

**Formula:** `(# of overlapping n-grams) / (# of n-grams in reference)`

**Key characteristics:**
- Recall-oriented (focuses on coverage of reference text)
- Commonly used with unigrams (ROUGE-1) or bigrams (ROUGE-2)
- Range: [0, 1] where 1 = perfect recall


```rust
use text_similarity_metrics::rouge_n_similarity;

let reference = "the quick brown fox jumps";
let candidate = "the quick brown dog runs";

// ROUGE-1 (unigram recall)
let score = rouge_n_similarity::<1>(reference, candidate, false);
println!("ROUGE-1 score: {}", score);

// ROUGE-2 (bigram recall)
let score = rouge_n_similarity::<2>(reference, candidate, false);
println!("ROUGE-2 score: {}", score);
```

### Cosine Similarity

A vector-based similarity metric that measures the cosine of the angle between two vectors. The algorithm is embedding-agnostic—it simply computes similarity between any two numeric vectors.

**Formula:** `cosine = dot(A, B) / (||A|| × ||B||)`, normalized to [0, 1]

**Key characteristics:**
- Works with any f64 vector embeddings (word2vec, BERT, custom features, etc.)
- Normalized to [0, 1]
- Single-pass computation of dot product and magnitudes
- Returns error if vectors have different lengths or contain zero vectors

```rust
use text_similarity_metrics::cosine_similarity;

let embedding1 = vec![0.5, 0.8, 0.3];
let embedding2 = vec![0.6, 0.7, 0.4];

let score = cosine_similarity(&embedding1, &embedding2).unwrap();
println!("Similarity: {:.3}", score);
```
