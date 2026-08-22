use std::cmp;

/// FEATURE 1: Reverso Complementario Ultrarrápido
/// El ADN tiene dos cadenas. Si leemos "ATGC", su espejo es "GCAT".
pub fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev() // Invertimos el orden
        .map(|&c| match c {
            b'A' => b'T',
            b'T' => b'A',
            b'C' => b'G',
            b'G' => b'C',
            _ => c, // Si hay ruido (N), lo dejamos igual
        })
        .collect()
}

/// Devuelve la versión "Canónica" del k-mer (la menor alfabéticamente).
/// Esto reduce la RAM a la mitad y mejora la Inteligencia Artificial.
pub fn canonical_kmer(seq: &[u8]) -> Vec<u8> {
    let rev = reverse_complement(seq);
    // Comparamos los bytes de ambas secuencias y nos quedamos con la menor
    if seq < &rev[..] {
        seq.to_vec()
    } else {
        rev
    }
}

/// FEATURE 4: MinHash (Huella dactilar probabilística)
/// Toma una secuencia y devuelve un número (Hash) que la representa.
pub fn hash_kmer(seq: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = ahash::AHasher::default();
    seq.hash(&mut hasher);
    hasher.finish()
}