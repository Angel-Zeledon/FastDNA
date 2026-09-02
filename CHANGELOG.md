# Changelog

Formato: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versionado: [SemVer](https://semver.org/) -- mientras la version mayor sea 0,
un incremento de la menor puede introducir cambios rompedores.

## Qué cubre este contrato

La superficie con compatibilidad garantizada es:
- Python: los nombres en `fastdna.__all__`, y los submódulos documentados
  (cada uno con su propio `__all__`).
- Rust: los módulos declarados `pub mod` en `src/lib.rs`.
- CLI: los flags y subcomandos documentados en `fastdna --help`.

## [Unreleased]

### Added

- Rust/Python: new pyfunction `sketch_from_kmers(kmers, k, sketch_size)` /
  `fastdna.sketch_from_kmers()`, wrapping the already-existing
  `sketch::GenomeSketch::from_kmers` -- builds a MinHash `Sketch` directly
  from an in-memory k-mer array, with no FASTQ file read. `python/fastdna/
  cv.py`'s `_mash_distance_matrix()`/`lineage_groups_at_thresholds()`/
  `default_threshold_curve()` now accept a `fastdna.CohortCounts` in place
  of paths, sketching each sample from its already-counted k-mers instead
  of re-reading its FASTQ file (`k` is always taken from `counts.k` in that
  case, silently, by design -- see `_mash_distance_matrix()`'s own
  docstring). `fastdna.audit()`'s `paths` argument accepts a `CohortCounts`
  too: `X` becomes `list(counts.sample_ids)` (satisfying
  `KmerVectorizer(counts=...)` automatically) and lineage/leakage-curve
  derivation reads from the same `counts`, closing the previous mutual
  exclusion between the fast, no-reread model path
  (`KmerVectorizer(counts=...)`) and the automatic leakage curve
  (`auto_leakage_curve=True`, the default), which previously required real
  FASTQ paths. Every existing call passing real paths is unaffected -- this
  is purely additive.
- Python: new module `fastdna.datasets` with `load_amr()` and
  `load_hiv_resistance()` -- real, labeled genomic cohorts (BV-BRC
  antimicrobial-resistance metadata; Stanford HIVDB HIV-1 protease
  drug-resistance genotype/phenotype pairs) in one function call, turning
  the manual real-data reproductions this project already did by hand
  (`scratch/amr_repro/`, `scratch/hiv_repro/`) into reusable, cached,
  idempotent loaders. Each returns a small frozen dataclass (`AmrCohort`/
  `HivResistanceCohort`) with `paths` ready for `fastdna.count()`/
  `fastdna.sklearn.KmerVectorizer`, a binary `phenotype` array, and a
  continuous regression target (`mic`/`fold_resistance`) for
  `fastdna.mic.MicRegressor`. Downloads are cached under
  `~/.fastdna/datasets` by default (override via `cache_dir=` or the
  `FASTDNA_DATA_CACHE` env var); a bad `species`/`antibiotic`/`drug` fails
  loudly, listing what is actually available. Not re-exported from
  `fastdna.__init__` (an onboarding/convenience module, not a
  differentiator-level one) -- reachable via `from fastdna.datasets import
  load_amr, load_hiv_resistance`. See `python/fastdna/datasets.py`'s
  module docstring for the full design rationale (cache location,
  supported species, and why HIV back-translation stays private to this
  module rather than living in `fastdna.translate`).
- Rust/Python/CLI: ntCard-style streaming k-mer frequency-spectrum
  estimator (`docs/feature-gap-analysis.md`'s S7(a); Mohamadi, Khan &
  Birol, *Bioinformatics* 2017) -- `src/ntcard.rs`'s `NtCardSketch`, the
  `fastdna spectrum` CLI subcommand (`src/cli.rs::SpectrumArgs`, `src/
  main.rs::run_spectrum`), and `fastdna.estimate_spectrum()` (`src/ffi.rs`,
  `python/fastdna/__init__.py`). Estimates both the distinct-k-mer count
  (F0) and the full frequency spectrum (how many distinct k-mers occur
  exactly once, twice, ...) in one streaming pass, in `2^precision * 16`
  bytes regardless of input size, reusing `hll.rs`'s hashing/bucket-
  assignment convention. Output matches `KmerCounts.spectrum()`'s exact
  `{depth: distinct k-mers}` shape, so it can be handed straight to
  `fastdna.genomescope.profile_genome()` wherever an exact spectrum is too
  expensive to compute first.
  - CLI: `fastdna spectrum --input FILE... -k 31 [--precision 14]
    [--max-frequency N] -o out.json|csv [--format csv|genomescope]`,
    following `count`'s multi-file/stdin input convention and
    `--histogram-format`'s CSV/GenomeScope output convention; `.json`
    output is chosen by `-o`'s own extension.
  - Accuracy characterized empirically, not assumed: on a synthetic
    dataset mixing a large low-depth error class with several deeper
    coverage classes, at the shared default precision 14, f1 (the
    error/noise class, the hardest to estimate) is estimated within 15% of
    the exact count and every other frequency class present in both
    spectra is within 30% -- see `src/ntcard.rs`'s module doc comment and
    its `ntcard_matches_the_exact_spectrum_within_a_measured_tolerance`
    test.
  - Scope note: S7's other half, k > 32 via `u128` k-mers, is **not**
    included here and remains open -- it would require changing the
    2-bit-per-base `u64` k-mer representation in `src/kmer.rs`, cascading
    into `src/counter.rs`, which had unrelated, uncommitted work in flight
    at the time this landed. See `docs/feature-gap-analysis.md`'s S7 entry
    for the full reasoning.
  - Tests: inline `#[cfg(test)]` unit tests in `src/ntcard.rs` (estimator
    arithmetic on hand-constructed inputs, an empty input, an all-unique
    input, a single-k-mer-repeated-many-times input, `--max-frequency`
    folding, and the exact-spectrum comparison test above), `tests/
    ntcard_cli.rs` (real-binary end to end, matching `tests/ktab_cli.rs`'s
    convention), and `python/tests/test_spectrum_estimate.py` (FFI wiring
    and `genomescope.profile_genome` interoperability).

- Rust/Python/CLI: perfiles de k-mers por lectura (`docs/feature-gap-
  analysis.md`'s S3, construido sobre S1). Nuevo `pub mod read_profile`
  (`src/read_profile.rs`): diseño en dos pasadas -- la tabla de referencia
  se construye primero (`count` -> `export.rs`, S1), luego este módulo la
  abre como `ktab::KmerTable` y recorre en streaming las lecturas de
  entrada, calculando para cada una el vector de conteos que sus propios
  k-mers canónicos tienen en esa tabla (posición `i` de la lectura -> conteo
  del k-mer que empieza en `i`) -- la característica insignia de FastK, sin
  equivalente en KMC.
  - **Formato de salida: RLE, no una tabla "tidy" de una fila por base.**
    Una fila por `(read_id, position, count)` sería trivialmente consumible
    desde DuckDB/pandas/polars, pero explota en tamaño a escala real (del
    orden de 90 mil millones de filas para una corrida humana de 30x con
    lecturas de 150pb, antes de que Parquet aplique su propia codificación)
    -- órdenes de magnitud más que las decenas de millones de filas que
    `export_counts_parquet` escribe para la misma entrada (una fila por
    k-mer *distinto*, no por base). En vez de eso, `read_profile.rs` escribe
    `(read_id, start, run_length, count)`: k-mers canónicos consecutivos se
    solapan en `k - 1` bases y, a profundidad de secuenciación típica, están
    respaldados por el mismo conjunto de lecturas superpuestas en un tramo
    no repetitivo y libre de errores del genoma -- por lo que sus conteos de
    referencia son frecuentemente *idénticos*, no solo cercanos. No es una
    observación inventada para este módulo: es la misma regularidad empírica
    que el propio formato `.prof` de FastK explota, y por eso un corte de
    "run" es en sí mismo informativo (marca una transición real de cobertura
    o un error de secuenciación). Sigue siendo tidy/consultable, y
    `read_profile::expand_rle` reconstruye sin pérdida la secuencia completa
    por posición a partir de las filas RLE (ver el doc comment del módulo
    para el razonamiento completo, incluyendo lo que sí y no se midió en
    esta sesión).
  - Junto al perfil RLE, siempre se escribe una segunda tabla pequeña de
    estadísticas por lectura: `read_id`, `n_kmers`, `n_present_kmers`
    (cuántos de los k-mers propios de la lectura aparecen en la tabla de
    referencia), `min_count`/`median_count`/`max_count` sobre la secuencia
    completa de conteos por posición (incluyendo posiciones ausentes de la
    referencia, es decir conteo `0`) -- suficiente para detección de errores
    y estimación de QV sin descomprimir un solo run RLE. `n_kmers`/
    `n_present_kmers` siguen la convención de nombres de columnas que ya usa
    `metagenomics.rs`'s `ReadClassification` (`n_kmers`, `n_classified_
    kmers`) en vez de inventar una segunda convención para una idea
    estructuralmente idéntica.
  - `ProfileIndex` (nuevo, en `read_profile.rs`): índice residente y
    ordenado de `(k-mer, conteo)`, construido una sola vez por ejecución vía
    `KmerTable::iter` -- el mismo truco que `read_filter::ReferenceIndex` ya
    usa para evitar llamar a `KmerTable::get` en un bucle ajustado por
    k-mer, pero manteniendo el conteo (no solo la presencia). Deliberadamente
    un tipo paralelo pequeño, no una generalización de `ReferenceIndex`: ese
    tipo ya está cubierto por las pruebas de `read_filter.rs` y es
    alcanzable desde un método público estable (`ReferenceIndex::
    from_table`) usado tanto por la CLI como por `ffi.rs::filter_reads`;
    ensancharlo para un segundo llamador no relacionado arriesgaba
    comportamiento probado y ya publicado a cambio de ahorrarse una
    definición de struct de pocas líneas.
  - `kmer.rs` gana `extract_canonical_kmers_with_positions_into`: el mismo
    bucle de extracción rodante que `extract_canonical_kmers_into` ya usa,
    con el añadido de la posición de lectura de cada k-mer -- necesario
    porque un perfil debe mapear cada conteo a "esta es la posición `i` de
    la lectura", no solo "estos son los k-mers que contiene la lectura".
  - Streaming, memoria acotada: las lecturas se recorren una a la vez vía
    `fastq::MultiSourceReader`, nunca materializadas -- la misma
    contrapartida que `read_filter.rs` documenta y adopta deliberadamente
    (memoria acotada por el índice de referencia residente, no por el
    tamaño de la entrada).
  - **Alcance: solo single-end**, la misma razón que `read_filter.rs`
    documenta para su propio filtrado sin sincronización de pares R1/R2.
  - CLI: `fastdna profile --input FILE... --table REFERENCE.parquet -o
    PROFILE.parquet [--summary SUMMARY.parquet]` (`cli::ProfileArgs`,
    `main.rs::run_profile`).
  - Python: `KmerTable.profile_reads(inputs, *, output, summary=
    "read_profile_summary.parquet")`, envolviendo `fastdna._core.
    profile_reads` (`src/ffi.rs`), que devuelve un `ProfileStats`
    (`reads_total`, `reads_profiled`).
  - Pruebas: pruebas unitarias inline en `src/read_profile.rs` (perfil de
    una lectura totalmente presente a un conteo uniforme, lectura
    totalmente ausente de la referencia, lectura más corta que `k`, un
    cambio de conteo que rompe un run exactamente en su posición, un run que
    nunca puentea un hueco por base ambigua aunque el conteo coincida a
    ambos lados, round-trip sin pérdida de `expand_rle`, manejo de bases
    ambiguas verificado contra el extractor compartido) y en `src/kmer.rs`
    (la nueva función posicional pinneada contra el extractor plano en cada
    `k`, el hueco alrededor de una base ambigua, una lectura más corta que
    `k`), más `tests/read_profile_cli.rs` (real-binario `count -> profile`
    de punta a punta) y `python/tests/test_read_profile.py`.
- Docs/config: real, runnable Snakemake and Nextflow workflow templates
  under `workflow_templates/`, closing `ml-genomics-roadmap.md`'s feature
  4(c) (the one sub-item its 2026-08-27 audit pass found genuinely missing
  despite the parent bundle being marked dispatched). Both templates run
  the same conceptual pipeline -- count a cohort of paired-end FASTQ
  samples, then fold every sample's k-mer table into one cohort-level table
  -- using only CLI surface that exists today (`fastdna count`'s multi-file
  `--input` for paired mates, `fastdna union` for the combined output;
  `fastdna matrix` does not exist yet in this snapshot, so `union` is the
  real subcommand used for "combine the cohort").
  - `workflow_templates/snakemake/`: `config.yaml` + `workflow/Snakefile`
    (Snakemake's own recommended layout) + `README.md`. Per-sample
    `fastdna_count` rule (wildcard-discovered via `glob_wildcards` on the
    `<sample>_R1.fastq.gz`/`<sample>_R2.fastq.gz` convention) plus a
    `fastdna_union` rule that folds every sample's Parquet table into
    `cohort_union.parquet` (or copies the single table forward when the
    cohort has only one sample, since `fastdna union` requires >= 2 inputs).
  - `workflow_templates/nextflow/`: `main.nf` (DSL2, `FASTDNA_COUNT` +
    `FASTDNA_UNION` processes) + `nextflow.config` + `README.md`, the same
    pipeline shape via `Channel.fromFilePairs`.
  - Superseded/removed the pre-existing single-file stubs
    (`workflow_templates/snakemake/Snakefile.example`,
    `workflow_templates/nextflow/fastdna.nf`, from commit `9d692f8`): those
    only ever shelled out `fastdna count` on one file at a time, with no
    paired-end handling, no config file, no combined/cohort output, and no
    per-directory `README.md`.
  - Verified two ways: (1) built the `fastdna` binary from this same
    snapshot and confirmed every flag used (`count --input/--output/
    --kmer-size/--min-quality/--min-count/--threads/--qc`, `union --input/
    --output/--combine`) against real `--help` output; (2) ran both
    templates end-to-end for real against that built binary -- a real
    `snakemake` (Docker image `snakemake/snakemake`) run and a real
    `nextflow run` (Nextflow installed via its own installer in a
    Docker container) over a small paired-end fixture, both completing all
    rules/processes and producing a genuine `fastdna union`-written
    `cohort_union.parquet`, not just a syntax check.
- Rust/Python/CLI: capa de consulta de acceso aleatorio sobre la tabla
  k-mer ordenada que `count()`/`fastdna count` ya escribe
  (`docs/feature-gap-analysis.md`'s S1). Ningún formato de archivo nuevo:
  `src/ktab.rs`'s `KmerTable` abre el mismo `.parquet` de siempre y usa las
  estadísticas por row-group de Parquet (min/max de `kmer_u64`) para podar
  qué row groups decodificar, en vez de cargar un índice completo en
  memoria. `export::export_counts_parquet` ahora adjunta siempre los
  metadatos de pie de página `fastdna.sorted_by=kmer_u64`/`fastdna.k=<k>`
  que `KmerTable::open` exige -- ningún flag nuevo, ninguna conversión: la
  salida habitual de `count` ya es una tabla consultable.
  - Rust: `KmerTable::open/k/len/is_empty/get/range/iter`,
    `encode_query_kmer` (`src/ktab.rs`, nuevo `pub mod ktab`).
  - CLI: `fastdna query --table FILE --kmer <secuencia-o-entero>`
    (`cli::QueryArgs`, `main.rs::run_query`).
  - Python: `fastdna.KmerTable` (`.open(path)`, `.k`, `.get(kmer)`,
    `__getitem__`, `__contains__`, `__len__`), envolviendo
    `fastdna._core.KmerTable` (`src/ffi.rs`).
  - Alcance: cubre la salida de ambas estrategias de conteo (en memoria y
    disco), que convergen en el mismo Parquet ordenado vía `export.rs`; la
    estrategia `binned` (S5) no está conectada a esa ruta y por tanto queda
    fuera, sin cambio respecto a antes de este ítem.
- Rust/Python/CLI: operaciones de conjunto entre tablas k-mer --
  unión/intersección/diferencia (`docs/feature-gap-analysis.md`'s S2,
  construido sobre S1). Cada operación es un único merge-join lineal en
  streaming sobre `KmerTable::iter` (`src/setops.rs::MultiTableMerge`, el
  mismo algoritmo de heap binario que `disk_spill.rs::merge_sources_into`,
  adaptado a `RangeIter`s respaldados por Parquet en vez de archivos de
  ejecución planos) -- nunca se carga una tabla completa en un hash set,
  precisamente porque eso desperdiciaría la razón de ser de una
  representación ordenada. La salida es el mismo `(kmer_u64, frequency)`
  Parquet ordenado que `count` ya escribe (`export::export_pairs_parquet`,
  nuevo, comparte el chunking/escritor de `export_counts_parquet`), así que
  el resultado de una operación de conjunto ya es una `KmerTable` válida --
  las operaciones componen.
  - Unión: todo k-mer presente en cualquiera de las tablas; `--combine
    sum|min|max` decide cómo se combinan los conteos por tabla en la única
    columna `frequency` de salida ("sum", el default, es la lectura natural
    de "combinar estas muestras en una").
  - Intersección: solo k-mers presentes en *todas* las tablas; expone el
    conteo de cada tabla de entrada (no solo un booleano) a través del
    mismo merge, plegado por `--combine` ("min" por defecto, igual que el
    reductor por defecto de `kmc_tools simple ... intersect`).
  - Diferencia (asimétrica, A-menos-B): k-mers de A ausentes de B, o cuyo
    conteo en B nunca supera `--max-subtract-count` (0 por defecto) --
    pensada para el caso real de sustracción de referencia/eliminación de
    huésped: acepta varias tablas B a la vez (`--subtract FILE...`) y
    descarta un k-mer si *cualquiera* de ellas lo supera. Las filas
    conservadas mantienen el conteo original de A, nunca mezclado con B.
  - Rust: `setops::{CombineOp, union, intersect, diff}` (nuevo `pub mod
    setops`), `export::export_pairs_parquet` (nuevo, en `export.rs`).
  - CLI: `fastdna union --input FILE FILE... -o OUT.parquet [--combine
    sum|min|max]`, `fastdna intersect --input FILE FILE... -o OUT.parquet
    [--combine min|sum|max]`, `fastdna diff --input FILE --subtract
    FILE... -o OUT.parquet [--max-subtract-count N]` (`cli::{UnionArgs,
    IntersectArgs, DiffArgs}`, `main.rs::{run_union, run_intersect,
    run_diff}`).
  - Python: `KmerTable.union(*others, output=..., combine="sum")`,
    `.intersect(*others, output=..., combine="min")`,
    `.difference(*subtract, output=..., max_subtract_count=0)`, cada uno
    devolviendo una nueva `KmerTable` ya abierta -- envolviendo
    `fastdna._core.ktab_union/ktab_intersect/ktab_diff` (`src/ffi.rs`).
- Rust/Python/CLI: filtrado de lecturas por contenido de k-mers
  (`docs/feature-gap-analysis.md`'s S4, construido sobre S1). Nuevo
  `pub mod read_filter` (`src/read_filter.rs`): recorre en streaming un
  archivo FASTQ/FASTA de entrada y conserva o descarta cada *lectura* según
  la fracción de sus propios k-mers canónicos que aparece en una
  `ktab::KmerTable` de referencia -- el filtrado `ref=`/`k=` de
  `kmc_tools filter`/BBDuk (eliminación de huésped/contaminante, o
  enriquecimiento dirigido).
  - Una lectura "coincide" con la referencia cuando la fracción de sus
    propios k-mers encontrados en la tabla es `>= --min-fraction`
    (inclusive; 0.1 por defecto). Una lectura que no produce ningún k-mer
    propio (más corta que el `k` de la tabla, o enteramente ambigua) nunca
    coincide, sin importar `--min-fraction` (ni siquiera con 0.0) --
    distinto de una lectura que calcula genuinamente una fracción de 0.0,
    la cual sí satisface un umbral inclusivo `>= 0.0`.
  - `--mode keep` escribe solo las lecturas que coinciden (enriquecimiento
    dirigido); `--mode discard` escribe solo las que no coinciden
    (eliminación de huésped/contaminante).
  - La tabla de referencia se carga una sola vez por ejecución en un
    `Vec<u64>` ordenado y residente en memoria (vía `KmerTable::iter`, ya
    en streaming), consultado con `binary_search` -- `ktab.rs`'s propio
    `KmerTable::get` reabre el archivo Parquet en cada llamada, lo cual
    sería catastrófico dentro de un bucle por cada k-mer de cada lectura.
    La memoria escala con el tamaño de la tabla de *referencia*, no con el
    de la corriente de lecturas de entrada, que permanece completamente en
    streaming.
  - La salida siempre es FASTQ (incluso para entrada FASTA, reutilizando la
    convención de calidad sintética Q40 que `fastq.rs` ya establece), gzip
    si la extensión de la ruta de salida lo indica, escrita atómicamente
    (`atomic::AtomicFile`).
  - **Alcance: solo single-end.** Cada archivo de `--input` se filtra de
    forma independiente y se concatena en una sola salida, la misma
    convención que ya usa el `--input` multi-archivo de `count`. El
    filtrado sincronizado de pares (R1/R2) -- donde un par se conserva o
    descarta como unidad si cualquiera de las dos mitades coincide -- **no
    está implementado**: pasar ambas mitades por `--input` las filtra de
    forma independiente y puede desincronizarlas. Queda documentado como
    trabajo futuro explícito, no como una implementación silenciosamente
    incompleta.
  - CLI: `fastdna filter --input FILE... --table REFERENCE.parquet --mode
    keep|discard [--min-fraction 0.1] -o OUT.fastq[.gz]` (`cli::FilterArgs`,
    `main.rs::run_filter`).
  - Python: `KmerTable.filter_reads(inputs, *, mode, output,
    min_fraction=0.1)`, envolviendo `fastdna._core.filter_reads`
    (`src/ffi.rs`), que devuelve un `FilterStats` (`reads_total`,
    `reads_written`).
- Rust/Python: jerarquía de excepciones real para `fastdna._core`
  (`src/ffi.rs`). Trece clases hoja con herencia múltiple genuina --
  `MalformedFastqError`, `InvalidKError`, `MismatchedKError`,
  `MismatchedScaleError`, `InvalidConfigError`, `NoSamplesFoundError`,
  `LoadError` (todas `(FastDnaError, ValueError)`), `IoNotFoundError`
  (`(FastDnaError, FileNotFoundError)`), `IoError` (`(FastDnaError,
  OSError)`), `MatrixTooLargeError`, `VocabTooLargeError` (ambas
  `(FastDnaError, MemoryError)`), `ExportError`, `InternalError` (ambas
  `(FastDnaError, RuntimeError)`) -- construidas con un snippet Python
  embebido ejecutado una vez al importar (`pyo3::create_exception!` solo
  admite una base). Permite capturar tanto por el tipo builtin de Python
  (`except ValueError`) como por la familia FastDNA (`except
  fastdna.FastDnaError`), a elección de quien llama.
- Python: nuevo módulo `fastdna.audit` con `audit(estimator, paths,
  phenotype, *, groups=None, covariates=None, n_splits=5, scoring=None,
  k=21, sketch_size=1000, lineage_threshold=0.01, random_state=0) ->
  AuditReport`, que mide la brecha entre CV aleatorio y CV bloqueado por
  linaje (`fastdna.cv.LineageKFold`) sobre un modelo y cohorte concretos
  -- la demostración concreta del hallazgo que `fastdna.cv` ya documentaba
  en abstracto (PLOS Biology 2025, arXiv 2502.07749). Cada `covariates=`
  añade una comparación bloqueada adicional, expuesta como
  `CovariateAudit`. Se importa con `from fastdna.audit import audit,
  AuditReport, CovariateAudit` -- no reexportado en `fastdna/__init__.py`,
  igual que el resto de módulos de la capa ML (`sklearn`, `cv`,
  `evaluation`, `workflow`, ...); `fastdna.CohortCounts`/`count_cohort`
  son la única excepción documentada, por una necesidad de import
  temprano que no aplica aquí.
- Python: nuevo módulo `fastdna.genomic_model` con `GenomicModel`, que
  envuelve un par `(estimator, vectorizer)` ya entrenado junto con las
  rutas FASTQ de la cohorte de entrenamiento, de modo que
  `.predict(sample)` devuelve, además de la predicción, si la muestra cae
  dentro del dominio de entrenamiento (reutiliza
  `fastdna.anomaly.CohortOutlierFlagger`, sin reimplementarlo). Incluye
  `OutOfDistributionWarning` y el classmethod `GenomicModel.fit(...)` para
  entrenar `vectorizer`/`estimator`/wrapper en una sola llamada.
- Python: nuevo módulo `fastdna.validate_generated` con
  `validate_generated(generated, reference, *, k=(3, 6, 11, 21), ...) ->
  GenerativeValidationReport`, que evalúa la plausibilidad de secuencias
  generadas (p. ej. por un modelo generativo) frente a una referencia real
  en tres ejes: composición de k-mers (distancia Jensen-Shannon
  multiescala), estructura de repeticiones/cobertura, y contención contra
  la referencia (`fastdna.sketch()`). Es un chequeo de plausibilidad, no
  una garantía de validez biológica -- ver el docstring del módulo.
- Python: `fastdna.sklearn.KmerVectorizer` acepta un nuevo parámetro
  `chunk_size=None` -- cuando se da, `fit()` construye el vocabulario en
  lotes de `chunk_size` muestras en vez de contar la cohorte completa de
  una vez, acotando el pico de memoria de esa fase al tamaño del lote más
  el propio vocabulario acumulado. Produce exactamente el mismo
  vocabulario que `chunk_size=None` (suma exacta, no aproximada), para
  cohortes de 10 000+ muestras donde el conteo completo no cabe en
  memoria.
- Python: `python/fastdna/py.typed` (marcador PEP 561) y anotaciones de
  tipo en la práctica totalidad de `python/fastdna/` (todos los módulos
  salvo los ya anotados), más `__all__` explícito por módulo donde
  faltaba -- habilita `mypy`/`pyright` para quien consuma el paquete.
- Python: nuevo módulo `fastdna.cohort_counts` con `CohortCounts`
  (dataclass inmutable con las tablas k-mer de un cohorte, concatenadas una
  sola vez; método `.subset(sample_ids)` para slicing O(filas) sin
  recontar) y `count_cohort(samples, *, k=31, min_count=1, threads=None,
  progress=None)`, que cuenta cada muestra del cohorte una única vez.
  Ambos quedan reexportados como `fastdna.CohortCounts` y
  `fastdna.count_cohort`.
- Python: nuevo módulo `fastdna.explain` con `explain(vectorizer,
  importances, paths, phenotype=None, *, top_n=20, lineage_threshold=None,
  annotation=None)`, que para las top-N features de un modelo ya entrenado
  evalúa identificabilidad, atribución por linaje y asociación intra-linaje
  (test CMH), para separar marcadores fenotípicos reales de artefactos de
  estructura poblacional. No reexportado en `fastdna/__init__.py`; se
  importa con `from fastdna.explain import explain`.
- Python: `fastdna.sklearn.KmerVectorizer` acepta un nuevo parámetro
  `counts=None` -- si se le pasa un `fastdna.CohortCounts`, `X` en
  `fit`/`transform` pasa a ser `sample_id` en vez de rutas FASTQ, y no se
  recuenta nada (slicing sobre el cohorte precontado), habilitando
  validación cruzada barata.
- Python: `fastdna.sklearn.KmerVectorizer` acepta un nuevo parámetro
  `representation` (`"presence"`, `"count"`, `"relative"` o `"clr"`) que
  controla qué contiene cada celda de la matriz esparza; nueva excepción
  `DepthConfoundingWarning` (subclase de `UserWarning`) emitida cuando
  `representation="count"` y la profundidad de secuenciación varía más de
  3x entre muestras.
- Python: `fastdna.KmerCounts.with_sequence()` añade/reconstruye la
  columna `kmer_sequence` decodificada desde `kmer_u64` sin releer el
  FASTQ original.
- Rust: `mem_estimate::estimate_binned_peak_bytes(occurrences, threads,
  num_bins, chunk_bytes)`, un modelo estructural (no calibrado con
  mediciones reales) del pico de RSS para la estrategia `binned`.
- CLI: nuevo flag `--with-sequence` para incluir la columna
  `kmer_sequence` decodificada en la salida CSV/Parquet (ver *Changed*:
  antes se incluía siempre).
- CLI: cuatro subcomandos nuevos (`src/cli.rs`, `src/main.rs`) que exponen
  funcionalidad del núcleo Rust que hasta ahora solo llegaba a través del
  binding de Python (feature-gap-analysis.md Q4):
  - `fastdna sketch --input FILE -k 21 --sketch-size 1000 -o out.json`
    -- construye y guarda un `GenomeSketch` MinHash (equivalente a
    `fastdna.sketch()` + `Sketch.save()`).
  - `fastdna dist --input FILE... [--metric jaccard|containment|mash]`
    -- comparación por pares entre dos o más sketches/archivos, análogo a
    `fastdna.compare_all()`. Acepta cualquier mezcla de sketches ya
    guardados (`.json`) y archivos FASTQ/FASTA (que se sketchean al
    vuelo); a diferencia de `compare_all()`, también admite
    `containment`, reportando ambas direcciones de cada par por ser una
    métrica asimétrica. Sin `--output`, la tabla CSV se imprime por
    stdout.
  - `fastdna card --input FILE -k 31 [--precision 14]` -- estimación de
    cardinalidad HyperLogLog (`hll::estimate_cardinality`), equivalente a
    `fastdna.estimate_cardinality()`.
  - `fastdna peek --input FILE [--n-reads 10000]` -- vista previa rápida
    (estadísticas de longitud de lectura, contenido GC, k sugerido),
    equivalente a `fastdna.peek()`.
  - `fastdna count ...` queda como subcomando explícito, idéntico en todo
    a no dar ningún subcomando (el comportamiento por defecto de siempre,
    preservado sin cambios para no romper invocaciones existentes --
    ver "Qué cubre este contrato" arriba). `src/cli.rs::CountArgs` es
    ahora la única definición de los flags de conteo, compartida entre
    ambos caminos vía `#[command(flatten)]`, en vez de duplicarse.
  - No rompedor: ninguna invocación existente cambia de comportamiento.
    `Cli` (en `src/cli.rs`) ahora expone `command: Option<Command>` y
    `count: CountArgs` en vez de los campos de conteo directamente sobre
    `Cli` -- un consumidor Rust externo que construyera/inspeccionara
    `Cli` directamente (no solo la CLI compilada) sí ve ese cambio de
    forma, documentado aquí por transparencia aunque no afecte a ningún
    invocador por línea de comandos.

### Changed

- **Rompedor (Python): `fastdna.audit()`'s `lineage_threshold` ahora se
  deriva de la cohorte en vez de valer 0.01 fijo.** Pasa a ser
  `Optional[float] = None`; `None` significa "léelo del dendrograma de esta
  cohorte" (la mediana de los puntos de `cv.default_threshold_curve`).
  Pasar un float sigue fijándolo exactamente, que es lo que mantiene
  reproducible cada punto de la curva por separado.

  El motivo es que la constante era medible-mente incorrecta sobre datos
  reales, no una preferencia de estilo. Sobre 200 genomas públicos de
  *E. coli* de BV-BRC (`scripts/validation/lineage_leakage_experiment.py`):
  con `0.01` salían 198 linajes de 200 muestras y `gap = -0.011`; agrupando
  por MLST (verdad externa) `gap = +0.165`; con `0.03`, `gap = +0.175`. El
  valor derivado en esa cohorte es 0.029, que cae con las dos respuestas
  correctas. La curva de esa cohorte abarca de 0.019 a 0.044, así que 0.01
  no era solo subóptimo: quedaba fuera del rango que sus propias alturas de
  fusión ofrecen. La distancia Mash no tiene escala universal --
  `cv.default_threshold_curve` ya lo argumentaba, y aun así el corte
  principal de `audit()` usaba una constante.

  Coste: normalmente cero. Cuando se calcula la curva de fuga (el
  comportamiento por defecto) sus puntos ya están disponibles. Solo con
  `auto_leakage_curve=False` se paga una pasada extra de dendrograma.

- **(rompedor)** El esquema de exportación por defecto (CSV/Parquet) ya no
  incluye la columna `kmer_sequence`. Antes: `kmer_u64,kmer_sequence,
  frequency` siempre. Ahora: `kmer_u64,frequency` salvo pedirla
  explícitamente (CLI: `--with-sequence`; Python: `with_sequence=True` en
  `fastdna.count(...)`, o reconstruirla después con `.with_sequence()`).
  En Rust, `export::counts_schema` gana el parámetro obligatorio
  `with_sequence: bool` (antes sin parámetros), igual que
  `export::export_counts_parquet`, `export::export_parquet`,
  `export::export_counts_csv`, `export::export_csv` y
  `cohort::batch::count_paired_samples` -- rompe a cualquier consumidor
  Rust externo de esas firmas.
- **(rompedor)** `fastdna.sklearn.KmerVectorizer` cambia su salida por
  defecto: antes `.transform()`/`.fit_transform()` devolvían siempre
  frecuencias crudas; con `representation="presence"` como nuevo valor por
  defecto, devuelven 0/1 (presencia/ausencia). Pipelines existentes que
  dependan de las frecuencias deben pasar `representation="count"`
  explícitamente.
- **(rompedor)** Superficie pública de módulos Rust reducida en
  `src/lib.rs`: `adaptive_bins`, `binned`, `disk_spill`, `minimizer` y
  `superkmer` pasaron de `pub mod` a `pub(crate) mod` (dejan de ser
  accesibles fuera del crate); eran detalles de implementación que nunca
  debieron ser públicos. `atomic` se mantiene `pub` porque
  `tests/resilience_hardening.rs` ya lo usa directamente.
- CLI: `fastdna --version` ahora lee `env!("CARGO_PKG_VERSION")` de
  `Cargo.toml` en vez de un literal `"0.1.0"` hardcodeado, evitando que
  diverja de `fastdna.__version__` en Python. Se eliminó el campo
  `author = "FastDNA Team"` (valor inventado) de la metadata de `clap`.
- Python: `KmerCounts.filter(min_count=, max_count=)` y `.top(n)` ahora
  validan sus argumentos (`TypeError`/`ValueError` con valores no-`int` o
  negativos) en vez de devolver silenciosamente resultados incorrectos.
  `sort_by(column)` valida el nombre de columna contra las columnas reales
  de la tabla y lanza `ValueError` con un mensaje claro, en vez del error
  críptico de pyarrow.
- Rendimiento interno, sin cambio de firma pública: `counter.rs` usa una
  partición MSD radix antes de ordenar en `compact_raw` (~1.2-1.4x más
  rápido en el benchmark aislado); `python/fastdna/multiomics.py`
  reescribe `kmer_feature_table()` con operaciones vectorizadas de Arrow
  en vez de diccionarios Python por celda.
- Rendimiento interno, sin cambio de firma pública: `counter.rs` (y su
  `mem_estimate.rs` asociado) pasan la tabla de conteo interna de un
  array-de-structs a un layout struct-of-arrays, para que la
  compactación/el merge de cada finalize toquen menos memoria por
  entrada. El backend gzip sigue siendo el de `flate2` por defecto
  (miniz_oxide); cambiarlo a zlib-ng estaba planeado en la misma tarea
  pero no se completó en esta ronda.
- `pyproject.toml`: se excluyen `**/__pycache__/**` y `**/*.pyc` del
  empaquetado (`[tool.maturin] exclude`), evitando bytecode de una versión
  específica de CPython en el wheel.
- Python: `python/fastdna/assembly_qc.py::evaluate_assembly()` cuenta el
  lado del ensamblaje (assembly) a través de `fastdna.count()` (pipeline
  Rust) en vez de un extractor de k-mers hecho a mano en Python
  (`docs/feature-gap-analysis.md`'s B4, ahora cerrado). Ya no hace falta
  el despacho por extensión de archivo que existía antes -- el núcleo Rust
  detecta FASTA vs. FASTQ por el primer byte del contenido, no por la
  extensión (Q1), así que `.fasta`/`.fa`/`.fna`/`.fastq`/`.fq` (`.gz` o no)
  toman la misma ruta, ya sin dispatch. Semántica sin cambios: mismo `k`,
  misma canonicalización (`kmer.rs::canonical_kmer_u64`, ya usada por el
  lado de las lecturas), misma regla de reinicio de ventana ante bases
  ambiguas (`N`) -- confirmado por `tests/fasta_input.rs`'s
  `a_fasta_file_counts_identically_to_the_equivalent_fastq` a nivel Rust y
  por `python/tests/test_assembly_qc.py::TestFastPathMatchesPurePythonFallback`
  a nivel Python (compara el nuevo `_assembly_kmer_counts` contra el
  extractor puro-Python original, `_count_fasta_kmers`, en el mismo
  fixture). El extractor puro-Python (`_count_fasta_kmers`/
  `_canonical_kmers`/`_iter_fasta_sequences`) se conserva, sin usar por
  `evaluate_assembly()`, como utilidad manual de bajo nivel para
  `evaluate_kmers()`. Nota de rendimiento (medida, no asumida): en
  fixtures de un solo contig de 20 kb a 2 Mb, esta ruta fue más lenta en
  reloj de pared que el extractor puro-Python que reemplaza -- el conteo
  en sí es más rápido en Rust, pero decodificar cada k-mer de vuelta a
  string (`with_sequence=True`) y construir el diccionario Python de
  salida domina en esa escala. El valor real de este cambio es un único
  núcleo de conteo con soporte FASTA nativo en vez de dos extractores de
  k-mers mantenidos por separado, no una mejora de velocidad garantizada
  en cualquier tamaño de entrada; ver el docstring del módulo y el propio
  comentario de `TestFastPathMatchesPurePythonFallback` para el detalle.
- Rust/CLI: `fastdna matrix`, el verbo de CLI y la exportación Parquet
  genérica que le faltaban a `cohort::matrix::CohortMatrix`
  (`docs/feature-gap-analysis.md`'s S6/B2, ahora cerrado). El motor
  (`build_cohort_matrix`) y su cableado a Python
  (`python/fastdna/gwas.py::cohort_presence_matrix`) ya existían; lo único
  que faltaba era el artefacto de archivo independiente del módulo GWAS.
  - Rust: `cohort::matrix::CohortMatrix` gana un campo `kmer_u64: Vec<u64>`
    (paralelo a `kmer_sequences`, sin coste real: es el mismo valor ya
    calculado al decodificar cada columna) y dos funciones nuevas de
    cableado -- `build_cohort_matrix_from_directory` (reutiliza
    `cohort::discover_samples`, la variante permisiva de emparejamiento
    R1/R2, no la estricta de `--paired-dir`) y `build_cohort_matrix_from_
    files` (lista explícita de archivos, uno por muestra, sin
    emparejamiento automático).
  - Rust: `export::export_cohort_matrix_parquet` (nuevo, en `export.rs`,
    reutiliza `AtomicFile`/`WriterProperties`/metadatos de pie de página en
    vez de inventar un escritor Parquet nuevo). Formato elegido:
    largo/"tidy"/COO -- una fila por cada entrada no-cero `(sample_id,
    kmer_u64, count)`, con `kmer_sequence` opcional vía `--with-sequence` --
    en vez de una matriz densa `samples x kmers`, porque eso es exactamente
    la representación que `CohortMatrix` ya mantiene en memoria (ver el
    comentario de módulo de `cohort/matrix.rs`) y una matriz densa real de
    cohorte sería casi enteramente ceros. `sample_id` se escribe como
    columna de texto, no como índice de fila, para que el archivo sea
    utilizable directamente desde DuckDB/pandas/polars sin ninguna
    herramienta específica de FastDNA. El pie de página registra
    `fastdna.k`/`fastdna.n_samples`/`fastdna.n_kmers`/
    `fastdna.n_candidates`/`fastdna.truncation_cutoff` (este último solo
    cuando `--max-kmers` realmente truncó algo) -- los mismos números de
    los que se construye la advertencia de truncamiento de `gwas.py`, sin
    reimplementarla. No se declara `fastdna.sorted_by=kmer_u64` como sí
    hace `export_counts_parquet`: este archivo está ordenado por muestra
    primero, no globalmente por k-mer, así que no es una `ktab::KmerTable`
    válida y no pretende serlo.
  - CLI: `fastdna matrix --input DIR|--sample FILE... -o OUT.parquet [-k 31]
    [--min-count 2] [--min-samples 2] [--max-kmers N] [--with-sequence]`
    (`cli::MatrixArgs`, `main.rs::run_matrix`). `--input DIR` reutiliza el
    descubrimiento de muestras de `cohort::discover_samples` (la misma
    convención que ya usa `count --paired-dir`); `--sample FILE...` nombra
    cada muestra explícitamente, derivando el id del archivo con la misma
    regla que `gwas.py::_sample_id_from_path` (se quita `.gz` primero,
    luego la extensión restante), para que los ids coincidan entre ambas
    rutas de entrada. `--min-count`/`--min-samples` por defecto en 2 (no 1),
    igual que `gwas.py::cohort_presence_matrix`, no que `count`.
  - Python: sin cambios en `gwas.py::cohort_presence_matrix()` -- sigue
    siendo la ruta en memoria hacia `scipy.sparse.csr_matrix`, y su
    contrato/tests no se tocan. Un punto de entrada Python dedicado a
    "exportar esta matriz de cohorte a Parquet" queda fuera de alcance de
    este ítem (ver `docs/feature-gap-analysis.md`'s S6 para el detalle):
    el verbo de CLI ya cubre el caso de uso de archivo independiente que
    S6 pedía.
  - Tests: unit tests inline en `src/cohort/matrix.rs` (`kmer_u64` paralelo
    a `kmer_sequences`, `build_cohort_matrix_from_directory`/`_from_files`
    incluyendo el caso de `--min-samples` por encima del tamaño de la
    cohorte y el aviso de huérfano no fatal) y en `src/export.rs` (esquema
    con/sin `--with-sequence`, round-trip exacto de las triples COO,
    rechazo de un `sample_ids` de longitud incorrecta, matriz vacía válida,
    metadatos de pie de página incluyendo truncamiento), más
    `tests/matrix_cli.rs` (extremo a extremo contra el binario real:
    descubrimiento por directorio, `--sample` explícito, `--with-sequence`,
    y los rechazos de validación cruzada de flags).

### Fixed

- Python: `fastdna.audit()` ya no reporta un `gap` numérico cuando el
  agrupamiento fue degenerado. Antes avisaba (`DegenerateLineagesWarning`)
  y devolvía el número igualmente -- y esa es la mitad peligrosa: los avisos
  se filtran, los tragan los notebooks o simplemente no se leen, mientras
  que el número que acaba en un paper es `report.gap`. Ahora `gap` es
  `float("nan")` con un `gap_undefined_reason` nuevo que nombra la causa,
  siguiendo la misma convención de "devuelve una razón, no un número" que
  `Confounding.value`/`undefined_reason` ya usaba (y que, de hecho, ya se
  activaba sobre ese mismo agrupamiento degenerado, `r == n`).
  `score_random`/`score_lineage` siguen poblados: esos sí se midieron, solo
  su diferencia es insegura de interpretar. Todas las vistas legibles
  (`__repr__`, `to_markdown`, `_repr_html_`) muestran "undefined
  (degenerate grouping)" en vez de "+nan".

- Rust: corregidos los tres avisos de clippy que habían dejado el paso del
  ratchet de CI por encima de su baseline de 8 sitios (`chimera_scan.rs` y
  `cohort_vocab.rs`, ambos añadidos después de fijarse ese baseline). Los
  tres son mecánicos y preservan el comportamiento; los 8 preexistentes se
  dejan intactos a propósito.

- Rust: `pipeline::resolve_strategy`, al forzar `--strategy binned`, ahora
  reporta `StrategyDecision.estimated_peak_bytes` usando el nuevo modelo
  `estimate_binned_peak_bytes` en vez de reutilizar la estimación de la
  estrategia in-memory, que sobreestimaba el consumo real en ese caso. El
  selector automático (`auto`) no cambia de comportamiento.
- `src/error.rs`: los mensajes de `FastDnaError::MatrixTooLarge` y
  `VocabTooLarge` ya no nombran flags de CLI inexistentes
  (`--top-features`, `--format sparse`, `--approx-vocab`); ahora nombran
  los parámetros reales (`top_features`, `min_count`).
- `src/error.rs`: `FastDnaError::Export` y `FastDnaError::Load` exponen su
  causa original vía `source: Option<Box<dyn std::error::Error + Send +
  Sync>>` en vez de aplanarla a `String` y devolver `None` desde
  `std::error::Error::source()`.
- Empaquetado del wheel (ver `pyproject.toml` en *Changed*).

### Removed

- `src/cms.rs` eliminado por completo -- el `pub mod cms;` correspondiente
  desaparece de `src/lib.rs`. Rompe a cualquier consumidor externo de
  `fastdna_core::cms`.
- `examples/binned_occupancy_report.rs` eliminado, consecuencia directa de
  que `binned` pasó a `pub(crate)` (el ejemplo ya no podía usar esos tipos
  desde fuera del crate). El diagnóstico se movió a un test interno
  `#[ignore]` (`per_bin_occupancy_report`) en `src/binned.rs`.
