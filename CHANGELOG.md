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

### Changed

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

### Fixed

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
