//! La frontera entre API publica y detalle interno, comprobada por el
//! compilador en vez de por revision.
//!
//! Un test de integracion vive fuera del crate, asi que solo ve lo que un
//! consumidor veria. Que este archivo compile es la prueba de que los
//! modulos publicos siguen siendolo; que los `pub(crate)` no se puedan
//! nombrar aqui lo garantiza el propio `pub(crate)`, y anadir una linea
//! que los nombre haria fallar la compilacion -- que es exactamente el
//! aviso que se quiere.

/// Cada `use` de aqui es una promesa de compatibilidad explicita. Anadir
/// uno significa "esto es API"; quitarlo, "esto deja de serlo".
///
/// Nota: el bloque de codigo que trae H-08 para este test fue escrito
/// contra una version anterior de `KmerCounter` (`new(k)` + `.k()`) y de
/// `GenomeSketch` (`.k()` como metodo). El trabajo en curso en paralelo ya
/// cambio esas firmas (`KmerCounter::new()` sin `k`, `GenomeSketch.k` como
/// campo publico) antes de que este archivo se anadiera; se adapta aqui a
/// la API real sin tocar `src/counter.rs` ni `src/sketch.rs`, que estan
/// fuera de alcance. La intencion del test -- comprobar que cada modulo
/// que H-08 deja publico sigue siendo nombrable desde fuera del crate --
/// no cambia.
#[test]
fn the_documented_public_modules_are_reachable() {
    use fastdna_core::counter::KmerCounter;
    use fastdna_core::error::FastDnaError;
    use fastdna_core::kmer::{canonical_kmer_u64, extract_canonical_kmers};
    use fastdna_core::ktab::KmerTable;
    use fastdna_core::pipeline::CountStrategy;
    use fastdna_core::sketch::GenomeSketch;

    let mut counter = KmerCounter::new();
    let kmers = extract_canonical_kmers(b"ACGT", 4);
    assert_eq!(kmers.len(), 1);
    counter.insert_batch(&kmers);
    assert_eq!(counter.total_kmers(), 1);

    assert_eq!(canonical_kmer_u64(0b00_01_10_11, 4), 0b00_01_10_11);

    let sketch = GenomeSketch::new(10, 21);
    assert_eq!(sketch.k, 21);

    assert_eq!(CountStrategy::InMemory.as_str(), "in-memory");

    let err = FastDnaError::InvalidK { k: 99, max: 32 };
    assert!(err.to_string().contains("99"));

    match KmerTable::open("does-not-exist-anywhere.parquet") {
        Err(FastDnaError::Io { .. }) => {}
        other => panic!("expected an Io error opening a missing table, got {other:?}"),
    }

    // El lector de tablas anchas es API publica por la misma razon que
    // `ktab`: `query` lo alcanza a traves de `ktab::table_key`, y quien
    // consuma el crate necesita poder nombrar el tipo que le devuelve esa
    // decision.
    use fastdna_core::ktab::{table_key, TableKey};
    use fastdna_core::wide_ktab::{encode_query_wide_kmer, WideKmerTable};

    match WideKmerTable::open("does-not-exist-anywhere.parquet") {
        Err(FastDnaError::Io { .. }) => {}
        other => panic!("expected an Io error opening a missing wide table, got {other:?}"),
    }
    match table_key("does-not-exist-anywhere.parquet") {
        Err(FastDnaError::Io { .. }) => {}
        other => panic!("expected an Io error routing a missing table, got {other:?}"),
    }
    let _: fn(&str, usize) -> Result<u128, FastDnaError> = encode_query_wide_kmer;
    assert_ne!(TableKey::Narrow, TableKey::Wide);
}
