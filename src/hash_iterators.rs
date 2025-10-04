use core::str;

const HASH_BASE: u64 = 131;
const HASH_MOD: u64 = 2305843009213693951;

pub(crate) trait CharHasher {
    fn hash(&self, c: char, hasher: &mut Fnv1aHash);
}

#[derive(Copy, Clone)]
pub(crate) struct CaseSensitiveCharHasher;

#[derive(Copy, Clone)]
pub(crate) struct CaseInsensitiveCharHasher;

type CaseSensitiveNgramHashIterator<'a, const N: usize> =
    NgramHashIterator<HashIterator<'a, CaseSensitiveCharHasher>, N>;
type CaseInsensitiveNgramHashIterator<'a, const N: usize> =
    NgramHashIterator<HashIterator<'a, CaseInsensitiveCharHasher>, N>;

pub(crate) struct NgramHashIteratorBuilder;

// Factory for creating NgramHashIterator
impl NgramHashIteratorBuilder {
    pub(crate) fn from_iter<I: Iterator<Item = u64>, const N: usize>(
        iter: I,
    ) -> NgramHashIterator<I, N> {
        NgramHashIterator::from_iter(iter)
    }

    pub(crate) fn from_str_with_hasher<'a, H: CharHasher, const N: usize>(
        s: &'a str,
        char_hasher: H,
    ) -> NgramHashIterator<HashIterator<'a, H>, N> {
        NgramHashIterator::from_str(s, char_hasher)
    }

    pub(crate) fn from_str_case_sensitive<'a, const N: usize>(
        s: &'a str,
    ) -> CaseSensitiveNgramHashIterator<'a, N> {
        let hasher = CaseSensitiveCharHasher;

        NgramHashIterator::from_str(s, hasher)
    }

    pub(crate) fn from_str_case_insensitive<'a, const N: usize>(
        s: &'a str,
    ) -> CaseInsensitiveNgramHashIterator<'a, N> {
        let hasher = CaseInsensitiveCharHasher;

        NgramHashIterator::from_str(s, hasher)
    }
}

pub(crate) struct NgramHashIterator<I: Iterator<Item = u64>, const N: usize> {
    hash_iterator: I,
    window: [u64; N],
    ngram_hash: u64,
    size: usize,
    pos: usize,
}

impl<I: Iterator<Item = u64>, const N: usize> NgramHashIterator<I, N> {
    pub(crate) fn from_iter(iter: I) -> Self {
        NgramHashIterator {
            hash_iterator: iter,
            window: [0; N],
            ngram_hash: 0,
            size: 0,
            pos: 0,
        }
    }
}

impl<'a, H: CharHasher, const N: usize> NgramHashIterator<HashIterator<'a, H>, N> {
    fn from_str(s: &'a str, char_hasher: H) -> Self {
        let hash_iterator = HashIterator::new(s, char_hasher);

        NgramHashIterator {
            hash_iterator,
            window: [0; N],
            ngram_hash: 0,
            size: 0,
            pos: 0,
        }
    }
}

// Const fn for computing (base)^(n-1) that is used multiple times through out the ngram building.
const fn get_hash_power(n: usize) -> u64 {
    let mut power: u64 = 1;
    let mut i = 1;
    while i < n {
        power = (power * HASH_BASE) % HASH_MOD;
        i += 1;
    }

    power
}

// Ngram hash iterator. Generate a U64 hash for each ngram. uses a rolling hash fro speed up, one pass on the input.
// The NgramHashIterator is wrapping another iterator that returns u64 hash for each word in the input.
impl<'a, I: Iterator<Item = u64>, const N: usize> Iterator for NgramHashIterator<I, N> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        // we special case for N=1, so that we just return the inner iterator hashes.
        if N == 1 {
            return self.hash_iterator.next();
        }

        let mut hash: u128 = 0;

        // we first build a window of size N of all the word hashes.
        // And we build the polynomial hash of the ngram hash(word) * BASE^pos % MOD,
        while self.size < N {
            let cur = self.hash_iterator.next()?;
            self.window[self.size] = cur;
            hash =
                (((hash * HASH_BASE as u128) % HASH_MOD as u128) + cur as u128) % HASH_MOD as u128;

            self.size += 1;

            if self.size == N {
                self.ngram_hash = hash as u64;
                return Some(self.ngram_hash);
            }
        }

        // After we return the first Ngram hash, we go through the items at n+1, n+2 etc..
        // for each item we add the current item and remove the last item from the window.
        // the window acts as a circular buffer. where window[pos] where pos is what we need to remove.
        let cur = self.hash_iterator.next()?;
        let old_token_hash = self.window[self.pos];
        self.window[self.pos] = cur;

        let last_power = get_hash_power(N);
        let term_to_remove = (old_token_hash as u128 * last_power as u128) % HASH_MOD as u128;
        let mut hash =
            (self.ngram_hash as u128 + HASH_MOD as u128 - term_to_remove) % HASH_MOD as u128;
        hash = (hash * HASH_BASE as u128 + cur as u128) % HASH_MOD as u128;

        self.pos = (self.pos + 1) % N;
        self.ngram_hash = hash as u64;

        return Some(self.ngram_hash);
    }
}

impl CharHasher for CaseSensitiveCharHasher {
    #[inline(always)]
    fn hash(&self, c: char, hasher: &mut Fnv1aHash) {
        if c.is_ascii() {
            hasher.hash(c as u8);
        } else {
            hasher.hash_u32(c as u32);
        }
    }
}

impl CharHasher for CaseInsensitiveCharHasher {
    #[inline(always)]
    fn hash(&self, c: char, hasher: &mut Fnv1aHash) {
        if c.is_ascii() {
            hasher.hash(c.to_ascii_lowercase() as u8);
        } else {
            for lc in c.to_lowercase() {
                hasher.hash_u32(lc as u32);
            }
        }
    }
}

pub(crate) struct HashIterator<'a, H: CharHasher> {
    chars: std::str::Chars<'a>,
    char_hasher: H,
}

impl<'a, H: CharHasher> HashIterator<'a, H> {
    fn new(src: &'a str, char_hasher: H) -> Self {
        let chars = src.chars();
        HashIterator { chars, char_hasher }
    }
}

// Simple HashIterator impl where we convert each word to a Fnv1aHash
impl<'a, H: CharHasher> Iterator for HashIterator<'a, H> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        let mut first_ch: Option<char> = None;

        while let Some(c) = self.chars.next() {
            if c.is_whitespace() {
                continue;
            }

            first_ch = Some(c);
            break;
        }

        if first_ch == None {
            return None;
        }

        let mut hasher = Fnv1aHash::new();
        self.char_hasher.hash(first_ch.unwrap(), &mut hasher);

        while let Some(c) = self.chars.next()
            && !c.is_whitespace()
        {
            self.char_hasher.hash(c, &mut hasher);
        }

        Some(hasher.get_hash())
    }
}

pub(crate) struct Fnv1aHash {
    hash: u64,
}

impl Fnv1aHash {
    #[inline(always)]
    fn new() -> Fnv1aHash {
        Fnv1aHash {
            hash: 0xcbf29ce484222325,
        }
    }

    #[inline(always)]
    fn hash(&mut self, b: u8) -> u64 {
        self.hash ^= b as u64;
        self.hash = self.hash.wrapping_mul(0x100000001b3);

        self.hash
    }

    #[inline(always)]
    fn hash_u32(&mut self, value: u32) -> u64 {
        self.hash((value & 0xFF) as u8);
        self.hash(((value >> 8) & 0xFF) as u8);
        self.hash(((value >> 16) & 0xFF) as u8);
        self.hash(((value >> 24) & 0xFF) as u8);
        self.hash
    }

    #[inline(always)]
    fn get_hash(&self) -> u64 {
        self.hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_iterator_basic() {
        let text = "hello world rust";
        let hash_iter = HashIterator::new(text, CaseInsensitiveCharHasher);

        let hashes: Vec<u64> = hash_iter.collect();

        assert_eq!(hashes.len(), 3);
    }

    #[test]
    fn test_hash_iterator_empty_string() {
        let text = "";
        let hash_iter = HashIterator::new(text, CaseInsensitiveCharHasher);

        let hashes: Vec<u64> = hash_iter.collect();
        assert_eq!(hashes.len(), 0);
    }

    #[test]
    fn test_hash_iterator_whitespace_only() {
        let text = "   \t\n  ";
        let hash_iter = HashIterator::new(text, CaseInsensitiveCharHasher);

        let hashes: Vec<u64> = hash_iter.collect();
        assert_eq!(hashes.len(), 0);
    }

    #[test]
    fn test_hash_iterator_single_word() {
        let text = "hello";
        let hash_iter = HashIterator::new(text, CaseInsensitiveCharHasher);

        let hashes: Vec<u64> = hash_iter.collect();
        assert_eq!(hashes.len(), 1);
        assert_ne!(hashes[0], 0);
    }

    #[test]
    fn test_hash_iterator_multiple_whitespace() {
        let text = "hello    world\t\trust\n\ntest";
        let hash_iter = HashIterator::new(text, CaseInsensitiveCharHasher);

        let hashes: Vec<u64> = hash_iter.collect();
        assert_eq!(hashes.len(), 4);
    }

    #[test]
    fn test_hash_iterator_consistency() {
        let text = "hello world";

        let hash_iter1 = HashIterator::new(text, CaseInsensitiveCharHasher);
        let hashes1: Vec<u64> = hash_iter1.collect();

        let hash_iter2 = HashIterator::new(text, CaseInsensitiveCharHasher);
        let hashes2: Vec<u64> = hash_iter2.collect();

        assert_eq!(hashes1, hashes2);
    }

    #[test]
    fn test_hash_iterator_different_words_different_hashes() {
        let text1 = "hello";
        let text2 = "world";

        let mut hash_iter1 = HashIterator::new(text1, CaseInsensitiveCharHasher);
        let hash1 = hash_iter1.next().unwrap();

        let mut hash_iter2 = HashIterator::new(text2, CaseInsensitiveCharHasher);
        let hash2 = hash_iter2.next().unwrap();

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_hash_iterator_same_word_same_hash() {
        let text1 = "hello world hello";
        let hash_iter = HashIterator::new(text1, CaseInsensitiveCharHasher);

        let hashes: Vec<u64> = hash_iter.collect();
        assert_eq!(hashes.len(), 3);
        assert_eq!(hashes[0], hashes[2]); // "hello" should have same hash
    }

    #[test]
    fn test_ngram_hash_iterator_basic_bigrams() {
        let text = "hello world rust";
        let ngram_iter = NgramHashIteratorBuilder::from_str_case_insensitive::<2>(text);

        let ngram_hashes: Vec<u64> = ngram_iter.collect();
        assert_eq!(ngram_hashes.len(), 2); // "hello world", "world rust"

        // Each n-gram hash should be non-zero
        for hash in &ngram_hashes {
            assert_ne!(*hash, 0);
        }
    }

    #[test]
    fn test_ngram_hash_iterator_basic_trigrams() {
        let text = "hello world rust test";
        let ngram_iter = NgramHashIteratorBuilder::from_str_case_insensitive::<3>(text);

        let ngram_hashes: Vec<u64> = ngram_iter.collect();
        assert_eq!(ngram_hashes.len(), 2); // "hello world rust", "world rust test"
    }

    #[test]
    fn test_ngram_hash_iterator_empty_input() {
        let text = "";
        let ngram_iter = NgramHashIteratorBuilder::from_str_case_insensitive::<2>(text);

        let ngram_hashes: Vec<u64> = ngram_iter.collect();
        assert_eq!(ngram_hashes.len(), 0);
    }

    #[test]
    fn test_ngram_hash_iterator_input_shorter_than_n() {
        let text = "hello";
        let ngram_iter = NgramHashIteratorBuilder::from_str_case_insensitive::<3>(text);

        let ngram_hashes: Vec<u64> = ngram_iter.collect();
        assert_eq!(ngram_hashes.len(), 0);
    }

    #[test]
    fn test_ngram_hash_iterator_input_exactly_n() {
        let text = "hello world";
        let ngram_iter = NgramHashIteratorBuilder::from_str_case_insensitive::<2>(text);

        let ngram_hashes: Vec<u64> = ngram_iter.collect();
        assert_eq!(ngram_hashes.len(), 1);
        assert_ne!(ngram_hashes[0], 0);
    }

    #[test]
    fn test_ngram_hash_iterator_consistency() {
        let text = "hello world rust test";

        let ngram_iter1 = NgramHashIteratorBuilder::from_str_case_insensitive::<2>(text);
        let hashes1: Vec<u64> = ngram_iter1.collect();

        let ngram_iter2 = NgramHashIteratorBuilder::from_str_case_insensitive::<2>(text);
        let hashes2: Vec<u64> = ngram_iter2.collect();

        assert_eq!(hashes1, hashes2);
    }

    #[test]
    fn test_ngram_hash_iterator_consistency_2() {
        let text = "hello world break hello world";

        let ngram_iter1 = NgramHashIteratorBuilder::from_str_case_insensitive::<1>(text);
        let hashes1: Vec<u64> = ngram_iter1.collect();

        let ngram_iter2 = NgramHashIteratorBuilder::from_str_case_insensitive::<2>(text);
        let hashes2: Vec<u64> = ngram_iter2.collect();

        assert_eq!(hashes1[0], hashes1[3]);
        assert_eq!(hashes1[1], hashes1[4]);
        assert_eq!(hashes2[0], hashes2[3]);
    }
    #[test]
    fn test_ngram_hash_iterator_different_sequences_different_hashes() {
        let text1 = "hello world rust";
        let text2 = "world rust hello";

        let ngram_iter1 = NgramHashIteratorBuilder::from_str_case_insensitive::<2>(text1);
        let hashes1: Vec<u64> = ngram_iter1.collect();

        let ngram_iter2 = NgramHashIteratorBuilder::from_str_case_insensitive::<2>(text2);
        let hashes2: Vec<u64> = ngram_iter2.collect();

        // Different word orders should produce different n-grams
        assert_ne!(hashes1, hashes2);
    }

    #[test]
    fn test_ngram_hash_iterator_rolling_window() {
        let text = "a b c d e";
        let ngram_iter = NgramHashIteratorBuilder::from_str_case_insensitive::<3>(text);

        let ngram_hashes: Vec<u64> = ngram_iter.collect();
        assert_eq!(ngram_hashes.len(), 3); // "a b c", "b c d", "c d e"

        // All hashes should be different (very high probability)
        assert_ne!(ngram_hashes[0], ngram_hashes[1]);
        assert_ne!(ngram_hashes[1], ngram_hashes[2]);
        assert_ne!(ngram_hashes[0], ngram_hashes[2]);
    }

    #[test]
    fn test_ngram_hash_iterator_large_n() {
        let text = "one two three four five six seven";
        let ngram_iter = NgramHashIteratorBuilder::from_str_case_insensitive::<5>(text);

        let ngram_hashes: Vec<u64> = ngram_iter.collect();
        assert_eq!(ngram_hashes.len(), 3); // 7 words - 5 + 1 = 3 n-grams
    }

    #[test]
    fn test_fnv1a_hash_consistency() {
        let mut hasher1 = Fnv1aHash::new();
        let mut hasher2 = Fnv1aHash::new();

        let test_bytes = b"hello";

        for &byte in test_bytes {
            hasher1.hash(byte);
            hasher2.hash(byte);
        }

        assert_eq!(hasher1.get_hash(), hasher2.get_hash());
    }

    #[test]
    fn test_fnv1a_hash_different_inputs() {
        let mut hasher1 = Fnv1aHash::new();
        let mut hasher2 = Fnv1aHash::new();

        for &byte in b"hello" {
            hasher1.hash(byte);
        }

        for &byte in b"world" {
            hasher2.hash(byte);
        }

        assert_ne!(hasher1.get_hash(), hasher2.get_hash());
    }

    // Case Insensitivity Tests
    #[test]
    fn test_hash_iterator_case_insensitive() {
        let text1 = "Hello World";
        let text2 = "hello world";
        let text3 = "HELLO WORLD";

        let hash_iter1 = HashIterator::new(text1, CaseInsensitiveCharHasher);
        let hashes1: Vec<u64> = hash_iter1.collect();

        let hash_iter2 = HashIterator::new(text2, CaseInsensitiveCharHasher);
        let hashes2: Vec<u64> = hash_iter2.collect();

        let hash_iter3 = HashIterator::new(text3, CaseInsensitiveCharHasher);
        let hashes3: Vec<u64> = hash_iter3.collect();

        assert_eq!(hashes1, hashes2);
        assert_eq!(hashes2, hashes3);
    }

    #[test]
    fn test_hash_iterator_case_sensitive() {
        let text1 = "Hello";
        let text2 = "hello";

        let mut hash_iter1 = HashIterator::new(text1, CaseSensitiveCharHasher);
        let hash1 = hash_iter1.next().unwrap();

        let mut hash_iter2 = HashIterator::new(text2, CaseSensitiveCharHasher);
        let hash2 = hash_iter2.next().unwrap();

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_hash_iterator_unicode_case_folding() {
        let text1 = "Café";
        let text2 = "café";
        let text3 = "CAFÉ";

        let mut hash_iter1 = HashIterator::new(text1, CaseInsensitiveCharHasher);
        let hash1 = hash_iter1.next().unwrap();

        let mut hash_iter2 = HashIterator::new(text2, CaseInsensitiveCharHasher);
        let hash2 = hash_iter2.next().unwrap();

        let mut hash_iter3 = HashIterator::new(text3, CaseInsensitiveCharHasher);
        let hash3 = hash_iter3.next().unwrap();

        assert_eq!(hash1, hash2);
        assert_eq!(hash2, hash3);
    }

    // N-gram tests with case insensitivity and Unicode
    #[test]
    fn test_ngram_hash_iterator_case_insensitive() {
        let text1 = "Hello World Rust";
        let text2 = "hello world rust";

        let ngram_iter1 = NgramHashIteratorBuilder::from_str_case_insensitive::<2>(text1);
        let hashes1: Vec<u64> = ngram_iter1.collect();

        let ngram_iter2 = NgramHashIteratorBuilder::from_str_case_insensitive::<2>(text2);
        let hashes2: Vec<u64> = ngram_iter2.collect();

        assert_eq!(hashes1, hashes2);
    }

    #[test]
    fn test_ngram_hash_iterator_unicode() {
        let text = "café naïve résumé";
        let ngram_iter = NgramHashIteratorBuilder::from_str_case_insensitive::<2>(text);

        let ngram_hashes: Vec<u64> = ngram_iter.collect();
        assert_eq!(ngram_hashes.len(), 2); // "café naïve", "naïve résumé"

        // Each n-gram hash should be non-zero
        for hash in &ngram_hashes {
            assert_ne!(*hash, 0);
        }
    }
}
