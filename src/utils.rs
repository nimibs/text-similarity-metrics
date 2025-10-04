pub(crate) const fn assert_n_le_32(n: usize) {
    assert!(n <= 32, "N cannot be greater than 32");
}