use std::{collections::HashMap, mem};

use nohash_hasher::BuildNoHashHasher;

use crate::hash_iterators::{NgramHashIterator};

pub(crate) type NoHasherHashMap<V> = HashMap<u64, V, BuildNoHashHasher<u64>>;

pub(crate) fn intersection_count<K, S>(h1: &HashMap<K, i32, S>, h2: &HashMap<K, i32, S>) -> i32
where
    K: Eq + std::hash::Hash,
    S: std::hash::BuildHasher,
{
    let mut intersection = 0;
    let (mut short, mut long) = (h1, h2);

    if short.len() > long.len() {
        mem::swap(&mut short, &mut long);
    }

    for (h, c) in short {
        if let Some(oc) = long.get(h) {
            intersection += c.min(oc);
        }
    }

    intersection
}

pub(crate) fn hash_counts<'a, I: Iterator<Item= u64>, const N: usize>(iter: &mut NgramHashIterator<I, N>) -> HashCounts {
    let mut count = 0;
    let mut hash_counts = NoHasherHashMap::default();

    for h in iter {
        let val = hash_counts.entry(h).or_insert(0);
        *val += 1;
        count += 1;
    }

    HashCounts { hash_counts, count }
}

pub(crate) struct HashCounts {
    pub(crate) hash_counts: NoHasherHashMap<i32>,
    pub(crate) count: u32,
}