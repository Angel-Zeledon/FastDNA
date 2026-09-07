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

> **Aviso, 2026-09-05.** Muchas de las entradas de *Added*, *Changed* y
> *Fixed* de abajo describen la capa de ML, que se eliminó el 2026-09-05
> antes de que ninguna versión la publicara (ver *Removed*). Se dejan
> escritas porque son el registro del trabajo y de los defectos que
> encontró, pero **no describen nada que se pueda instalar**: si una
> entrada nombra `fastdna.sklearn`, `fastdna.audit`, `fastdna.cv`,
> `fastdna.gwas`, `fastdna.metagenomics` o cualquier otro módulo de esa
> lista, se refiere a código que ya no existe.

### Changed

- **El merge entre bins ahora es paralelo** (`counter::k_way_merge_sorted_
  counts_parallel`). Era la mayor porción secuencial que quedaba en una
  corrida binned: 1,88 s en un core contra 2,76 s de todo el conteo por bin
  repartido en once. Se parte por **rango de claves**, no por fuente — cada
  rango toma un tramo contiguo de cada tabla ordenada por búsqueda binaria,
  los rangos se mezclan por separado y se concatenan en orden. Así conserva
  la propiedad de una sola pasada que el doc del merge secuencial defiende
  frente al árbol de reducciones por pares.

  Los puntos de corte son cuantiles de una muestra de las claves reales, no
  divisiones iguales del rango `u64`: las k-mers canónicas no se distribuyen
  uniformemente y una muestra de baja complejidad le daría a un worker casi
  todo el trabajo — el mismo modo de fallo que hubo que medir para el
  balance de bins.

  **Paraleliza mal, y ese es el hallazgo**: 1,88 → 1,48 s, un 1,8x con once
  partes, porque el merge lee 645 MB y escribe 645 MB y está limitado por
  ancho de banda, no por comparaciones. Un intento de quitar los 0,40 s de
  concatenación copiando en paralelo sobre rebanadas disjuntas **no mejoró
  nada** (`vec![_; n]` inicializa a cero esos 645 MB), así que se mantuvo la
  copia secuencial, más simple. De punta a punta la diferencia queda dentro
  de la varianza entre corridas, así que no se afirma nada end-to-end.
  Salida byte-idéntica, verificada por SHA-256 sobre el archivo de 840M
  ocurrencias, y un test diferencial fija que las dos versiones coinciden.


- **El contador ancho: 5,6x menos memoria, tras medirlo.** El módulo se
  entregó diciendo "no medido, y por tanto no afirmado". Medirlo dio
  4.438 MiB de pico con un solo hilo sobre 144M ocurrencias, y dos
  hipótesis mías salieron falsas antes de acertar:

  1. *"Escala con hilos × distintas"* — falsa en una corrida: 1 hilo
     costaba 4.438 MiB y 11 costaban 4.713 MiB.
  2. *"El buffer crudo no se está acotando"* — falsa, y la comprobación
     quedó como test de regresión permanente (que la *capacidad* siga cerca
     de su umbral tras 40M inserciones; un buffer que crece en silencio
     devolvería el pico a depender del tamaño de la entrada).

  Lo que era: escalaba con **ocurrencias**, a ~26 bytes cada una — la firma
  de asignar y liberar un buffer grande repetidamente, no de retener uno.
  `compact` pedía un destino de merge nuevo en cada una de ~72
  compactaciones. Reutilizar dos buffers que se alternan lo dejó en
  1.187 MiB; guardar la tabla como arrays paralelos en vez de
  `Vec<(u128, u32)>` (32 bytes de los que 12 son relleno de alineación) lo
  dejó en **793 MiB**, que es aproximadamente lo que cuesta el motor
  estrecho por hilo.

  Con 4 hilos son 2.002 MiB **y 2,54 s**, más rápido que con 11 (4.152 MiB,
  3,36 s). Esa inversión no la predije: el borrador del comentario decía lo
  contrario y la medición lo contradijo, así que el doc dice lo que dice la
  tabla. Todo en `docs/BENCHMARKS.md`.


- **El contador ahora elige la estrategia particionada por minimizers
  (`binned`) por defecto.** Estaba implementada, correcta y probada desde el
  2026-08-25, pero `auto` no podía seleccionarla: faltaba el modelo de
  memoria calibrado y quedaba viva la preocupación R3 (desbalance de bins en
  datos reales). Ambas cosas se resolvieron con medición:

  - El modelo (`mem_estimate::estimate_binned_peak_bytes`) se calibró contra
    ocho corridas reales -- dos tamaños de entrada x cuatro números de hilos.
    Era estructural y nunca medido, y estaba mal en las dos direcciones a la
    vez: **sub-predecía 26-29%** las corridas de 840M ocurrencias y
    **sobre-predecía 11-19%** las de 144M. Dos errores estructurales lo
    causaban (el merge suponía liberado el store de super-k-mers, y la base
    venía del ajuste de la estrategia en memoria) y un factor de calibración
    nombrado cubre el resto. Residuales de 0,0% a +11,1%, sin sub-predecir
    ningún punto: un modelo que dice que una corrida cabe cuando no cabe es
    como una máquina se queda sin memoria a los veinte minutos.
  - R3 **no** lo resolvía el mapa adaptativo por sí solo. En una entrada de
    baja complejidad (tipo amplicón, 734 MB, 330M ocurrencias) `binned`
    resultó **3,4x más lenta y 2,8x más pesada** que contar en memoria: si
    el espacio de firmas es más estrecho que el número de bins, ningún
    empaquetado puede repartirlo. El tamaño de la entrada no sirve de
    criterio, así que `auto` **mide** el balance que el empaquetado logra
    sobre un prefijo acotado de la entrada real y rechaza `binned` por
    encima de `MAX_ACCEPTABLE_BIN_SKEW`. Medido: 1,20x en shotgun contra
    32,71x en amplicón -- las dos poblaciones están a un factor de 27.

  Efecto en el camino por defecto, misma máquina y mismo archivo:
  **24,10 s / 7,31 GB -> 9,88 s / 3,43 GB** sobre 840.000.000 de
  ocurrencias, con salida Parquet byte-idéntica. Forzar `--strategy binned`
  sigue funcionando y ahora salta la comprobación de balance, que es lo que
  significa forzar.

- **`--strategy` cambia de significado en `auto`** (documentado en
  `fastdna --help`), y el reporte final imprime el balance medido cuando lo
  hubo, para que una corrida anunciada como `binned` que terminó siendo otra
  cosa diga por qué.

### Added

- **Python: `fastdna.similarity(tables)`.** La misma similitud exacta que
  `fastdna similarity` en el CLI, devolviendo un `pyarrow.Table` en formato
  largo con `sample_a`, `sample_b`, `shared`, `only_a`, `only_b`,
  `jaccard`, `containment_ab`, `containment_ba` y `bray_curtis`. Las filas
  se etiquetan con las rutas tal como las pasó quien llama, no con índices.

  Cierra una brecha entre lo que `CLAUDE.md` afirmaba ("la API de Python
  refleja el CLI") y lo que existía: `compare_all` compara *sketches*
  MinHash y es aproximado por construcción, y no puede dar Bray-Curtis ni
  en principio porque un sketch descarta los conteos.


- **`k > 32`: un segundo motor, no una generalización del que ya había.**
  `--engine auto|narrow|wide`. `auto` (el default) enruta por `k`: hasta 32
  el motor estrecho (`kmer.rs`/`counter.rs`, dos bits por base en un `u64`),
  y de 33 a 64 el ancho (`wide_kmer.rs`/`wide_counter.rs`, lo mismo en un
  `u128`).

  **Por qué dos motores y no uno genérico.** El camino `u64` es el que está
  medido *exactamente* igual a KMC3 sobre lecturas reales y el que describe
  cada número de `docs/BENCHMARKS.md`. Hacerlo genérico pondría todos esos
  resultados otra vez en duda: la monomorfización *debería* preservar el
  código generado, pero "debería" no es sobre lo que descansa un número
  validado. El motor ancho se sienta al lado, y el estrecho no cambia.

  **Lo que impide que se separen**: `wide_matches_the_narrow_engine_exactly_
  where_they_overlap` compara los dos motores k-mer a k-mer en todo
  `k ≤ 32`, incluyendo alrededor de bases ambiguas, y
  `tests/wide_engine.rs` repite la comparación *de punta a punta por el
  binario real* sobre un FASTQ. Así el motor ancho hereda la validación
  contra KMC3 en el rango solapado. Por eso `--engine wide` se acepta
  también por debajo de 32: es la forma de hacer esa comprobación sobre
  datos propios, no solo en la suite.

  **Forzar se respeta o se explica, nunca se corrige en silencio**:
  `--engine narrow` con `k=41` es un error que nombra la bandera que lo
  arregla, no una promoción callada al motor ancho.

  **Límites, dichos en vez de descubiertos.** Hasta k=64, no los 256 de
  KMC3: dos bits por base en 128 bits son 64 bases, y pasar de ahí exige
  una clave de bytes que cambia el orden, el esquema Parquet y cada
  comparación del contador. Y el motor ancho cuenta **solo en memoria**:
  `disk_spill.rs` y `binned.rs` son maquinaria de claves de 8 bytes de
  principio a fin, así que `--strategy` no aplica por encima de k=32 y el
  CLI lo dice en vez de ignorar la bandera.

  La tabla ancha es Parquet con `kmer_bits` (16 bytes big-endian, de modo
  que el orden de bytes *es* el orden numérico y el fichero sigue estando
  ordenado para quien no lo decodifique) y `fastdna.sorted_by=kmer_bits`.
  Las operaciones con clave `u64` -- `query`, `union`/`intersect`/`diff`,
  `filter`, `similarity`, sketching, la API de Python -- la **rechazan por
  nombre** en vez de malinterpretarla, y el error explica cuál es.

### Changed

- **`FastDnaError::InvalidK` lleva ahora el límite del que habla**
  (`InvalidK { k, max }`). Con dos motores, un mensaje fijo en 32 miente:
  a quien pedía `k=65` se le respondía "k debe estar entre 1 y 32", de lo
  que se sigue razonablemente que `k=41` tampoco está disponible -- cuando
  es exactamente lo que cuenta `--engine wide`. Es un cambio rompedor del
  enum público, que el 0.x permite y esta línea declara.


- **`fastdna similarity`: similitud exacta entre tablas de k-mers contadas.**
  Jaccard, contención asimétrica (las dos direcciones) y la disimilitud de
  Bray-Curtis ponderada por abundancia, entre dos o más tablas `.parquet`.

  Es lo que `dist` no puede dar: `dist` compara *sketches* MinHash, que son
  aproximados por construcción y descartan los conteos, así que Bray-Curtis
  no es computable ahí ni siquiera en principio. Y es lo que KMC3 no da:
  `kmc_tools` calcula las operaciones de conjunto, pero ninguna herramienta
  de esa familia reporta similitud.

  **Una sola pasada de merge para todos los pares.** `setops::MultiTableMerge`
  ya entrega, por cada k-mer distinta, el conteo de cada tabla; todas las
  estadísticas de todos los pares son sumas corrientes sobre ese mismo flujo,
  así que N tablas cuestan un merge, no `N*(N-1)/2`. Solo se tocan los pares
  que **comparten** la k-mer: los conteos exclusivos se derivan
  (`only_a == |A| - shared`), de modo que una k-mer presente en 2 de 200
  tablas cuesta una actualización y no 19.900.

  **Verificado contra KMC3**, no solo contra tests propios: sobre dos mitades
  de un mismo set de lecturas (solapamiento real, Jaccard 0,783),
  `scripts/validation/similarity_vs_kmc_tools.py` compara |A|, |B| y |A∩B|
  contra `kmc_tools intersect` y cada ratio derivado. Coinciden **exactamente**
  — k-mer a k-mer y a nueve decimales. Corre en el workflow de validación.

  Los casos degenerados están decididos y documentados en vez de devolver
  NaN: dos tablas vacías dan jaccard 1,0 y Bray-Curtis 0,0 (idénticas por
  vacuidad), y la contención desde una tabla vacía es 1,0.


- **macOS: detección de memoria del sistema.**
  `mem_estimate::available_system_memory_bytes` no tenía implementación en
  macOS y caía al fijo de 4 GiB, así que **cualquier Mac quedaba presupuestado
  en 4 GiB tuviera 8 GB o 192 GB** -- y eso solo bastaba para mandar al disco
  corridas que caben holgadas en RAM. Ahora lee `hw.memsize` vía
  `sysctlbyname` (binding escrito a mano, sin dependencia nueva, igual que el
  de `GlobalMemoryStatusEx` en Windows). Es memoria **total**, no disponible
  como en Linux y Windows: la diferencia está documentada en la propia
  función en vez de escondida.


- **Python: `fastdna.sklearn.EmptyVocabularyWarning`** -- `fit()` avisa
  cuando la regla de `presence` no deja ningún feature en pie, es decir
  cuando *toda* k-mer de la cohorte está en todas las muestras. Nombrar la
  causa en el momento del `fit()` evita que el síntoma aparezca varias
  llamadas después como una matriz `(n_samples, 0)` y un error de
  scikit-learn que no menciona ni la cohorte ni el motivo. Se exporta en
  `fastdna.sklearn.__all__` (parte del contrato) para que se pueda filtrar
  o convertir en error.

- **Rust: `cohort_vocab::VocabularyRanking`** -- las dos reglas de ranking
  de vocabulario, explícitas como parámetro de `rank_vocabulary` en vez de
  constantes, precisamente para que la ruta en memoria (NumPy) y la ruta
  en disco (Rust) no puedan divergir. `rank_cohort_vocabulary` gana un
  tercer argumento `ranking` (`"prevalence"` por defecto, `"binary"` para
  presencia) que rechaza cualquier otro valor en vez de caer al default.

- **Python: nuevo módulo `fastdna.design` con `check_design()`** -- responde
  si un experimento *puede* producir un resultado, antes de correrlo. Solo
  necesita la forma del diseño (etiquetas, número de features, grupos,
  folds): ni genomas, ni conteo, ni ajuste. Reporta `p/n`, tamaño de la
  clase minoritaria, grupos frente a folds, cuota del grupo dominante,
  tamaño del fold de test más pequeño, y el intervalo de confianza de
  Hanley-McNeil (1982) sobre el AUC que se espera detectar.

  Nace de una pérdida concreta de esta sesión: dos intentos de demostración
  de fuga que no podían funcionar, ambos diagnosticables por aritmética de
  antemano. 80 muestras contra 5.000 features (`p/n = 62`) no puede aprender
  nada generalizable, y el IC al 95% sobre un AUC de 0.75 en esa cohorte es
  ±0.11 -- más ancho que el gap de +0.075 que la corrida acabó reportando.
  Ambas cifras salen en milisegundos.

  Sin veredicto: reporta números y nombra preocupaciones, no devuelve
  PASS/FAIL ni impide correr nada -- misma postura que `audit()` y
  `evaluation`. `auc_standard_error()` está validada **contra simulación
  Monte Carlo**, no contra una constante copiada del paper. No se
  re-exporta en `fastdna.__all__` (sigue el precedente de
  `fastdna.datasets`): `from fastdna.design import check_design`.

- **`scripts/validation/known_truth_checks.py`: 19 módulos comprobados contra
  una verdad construida.** Complementa la capa anterior: aquellos módulos se
  anclan a una herramienta externa (KMC3, Merqury, Mash, GenomeScope2), y
  estos no tienen ninguna disponible -- no existe una implementación de
  referencia de "marca la muestra que no encaja" -- pero su afirmación sigue
  siendo comprobable construyendo un caso cuya respuesta se conoce de
  antemano: `P(y=1|x)` analítica para `calibration`, un contaminante
  inyectado para `anomaly`, una variante causal para `gwas`, un GFF escrito
  a mano para `annotate`, lecturas tomadas literalmente del genoma de
  referencia para `read_profile` y `metagenomics`.

  Dos tienen anclas genuinamente externas: `mic` se comprueba contra la
  convención de *essential agreement* de CLSI, e `interop` importa Biopython
  real, porque el claim es interoperar con esa librería concreta.

  Encontró dos de los defectos corregidos en esta versión (el falso "very
  deviated" de `validate_generated` y el `TypeError` de `uncertainty_score`).
  Lo que **no** establece está dicho en el propio script: los casos
  construidos son deliberadamente limpios, y las cohortes reales son más
  sucias de formas que no reproducen.

- **`scripts/validation/`: comprobación contra herramientas externas, y un
  workflow programado que la ejecuta** (`.github/workflows/validation.yml`).
  Seis scripts que contrastan FastDNA con la implementación establecida de
  cada algoritmo que toma prestado, **sobre datos de secuenciación reales**
  en vez de generados, con las herramientas de referencia como
  biocontenedores fijados por tag:

  | qué | contra | resultado |
  |---|---|---|
  | conteo de k-mers | KMC3 3.2.4 | **coincidencia exacta** (k=31/21/15) |
  | distancias MinHash | Mash 2.3 | r=0,997, sesgo ~0 |
  | tamaño de genoma | GenomeScope2 2.0.1 | −0,29% |
  | QV de ensamblado | Merqury 1.4.1 | 18,4192 vs 18,4205 |
  | cardinalidad HLL | el conteo exacto | −0,26% (límite ~0,8%) |
  | espectro ntCard | el histograma exacto | −0,6% / +2,0% / −1,5% |
  | clasificación de especie | identidad publicada | 12/12 genomas held-out |

  Cada script asserta lo más estricto que su objeto admite: igualdad exacta
  para el conteo (ambas herramientas resuelven el mismo problema una vez
  desactivado el trimming), el límite documentado para los estimadores, y
  ausencia de sesgo más decaimiento `1/sqrt(sketch_size)` para MinHash
  (exigir igualdad ahí sería exigir reproducir el ruido de muestreo de Mash).
  Cubre los huecos que `docs/audit/`ya señalaba: el claim central de
  `genomescope` solo se probaba contra un espectro generado por la misma
  familia de modelos que ajusta, y el de `taxonomy` contra motivos repetidos
  (`"ACGTGGCATCAGT"*n`). Lo que **no** cubre está listado explícitamente en
  `docs/validation-real-data.md`. Los tres bugs corregidos en esta versión
  fueron encontrados por estos scripts, no por los ~1.700 tests.

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
  reales, no una preferencia de estilo -- y lo era **en ambas direcciones**,
  que es la forma más contundente de ese argumento. Sobre ensamblados
  públicos de *E. coli* de BV-BRC
  (`scripts/validation/lineage_leakage_experiment.py`): `0.01` dejaba 198
  linajes de 200 muestras en una cohorte (demasiado fino) y fundía 80
  genomas en 15 grupos en otra cuya estructura real son 38 sequence types
  (demasiado grueso). El valor derivado en la segunda es `0.0041` -> 40
  grupos, recuperando la partición MLST con **ARI 0.931** frente al 0.512 de
  la constante. La distancia Mash no tiene escala universal --
  `cv.default_threshold_curve` ya lo argumentaba, y aun así el corte
  principal de `audit()` usaba una constante.

  (Una versión anterior de esta entrada justificaba el cambio con un gap de
  fuga de `+0.165`. Ese número está **retractado**: era artefacto del bug de
  truncación descrito más abajo. Ver la nota de retractación al inicio del
  Track D de `docs/validation-real-data.md`. La justificación de arriba se
  midió sobre genomas completos y no depende de él.)

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

- **El modelo de memoria de `binned` sub-predecía, y mis propias mediciones
  lo ocultaban.** Los ocho puntos de calibración registrados el 2026-09-06
  eran de **una sola corrida cada uno**. Repitiéndolos tres veces, todos
  resultaron entre un 20% y un 55% bajos — y con las cifras honestas el
  factor de 1,195 **sub-predice** en 144M ocurrencias con 8 y 11 hilos
  (1,6% y 1,3%). Sub-predecir es el único fallo que ese modelo existe para
  hacer imposible.

  El factor pasa a 1,24: el mínimo que no deja ninguna *corrida* por encima
  de la predicción, ajustado contra la **peor de tres** en cada punto, no
  contra la mediana. Una cota ajustada a la mediana la supera la mitad de
  las corridas que acota.

  Se retiran dos afirmaciones que hice el 2026-09-06 a partir de aquellas
  corridas únicas: "la memoria ya no crece con el número de hilos" (con
  144M ocurrencias crece claramente, 785 → 1.008 MiB de 1 a 11 hilos) y
  "sobre-predice hasta un 94%" (el techo real es 33%).


- **`--threads` no acotaba la fase 2 del conteo binned**, y eso hacía que
  el modelo de memoria **sub-predijera**. `BinStore::finish` paralelizaba
  sobre el pool global de rayon (dimensionado por número de cores) mientras
  `--threads` solo dimensionaba los workers de la fase 1, así que `-t 1`
  contaba bins en los once cores. No era solo una bandera que prometía de
  más: el término de fase 2 de `estimate_binned_peak_bytes` es
  `threads × transitorios_por_bin`, de modo que con `-t 1` el modelo contaba
  un bin en vuelo mientras corrían once — sub-predicción, que es el único
  fallo que ese modelo está calibrado para no cometer nunca.

  La fase 2 corre ahora en un pool del tamaño de `--threads`, construido
  solo cuando difiere del global para que la corrida por defecto no pague
  nada. El pico de memoria bajó en todos los puntos medidos (840M
  ocurrencias con 11 hilos: 3.287 → 1.769 MiB), lo que deja el modelo
  conservador en vez de ajustado. **No se re-ajustaron las constantes, a
  propósito**: la memoria ya no crece con el número de hilos, así que la
  forma del modelo no describe los datos, y re-ajustar una estructura que
  no encaja daría un número que interpola en vez de un modelo — que es
  exactamente lo que produjo la versión estructural sin medir. Sobre-predice
  hasta un 94%, no sub-predice en ningún punto, y su propio test lo dice.


- **Python: `KmerVectorizer` con `representation="presence"` (el default)
  seleccionaba exactamente las k-mers constantes.** Dos decisiones bien
  razonadas por separado y degeneradas juntas. `_select_vocabulary` rankeaba
  por prevalencia descendente -- una k-mer presente en todas las muestras es
  una señal más confiable que una con profundidad enorme en una sola --, y
  `presence` codifica 0/1 para ser inmune a la profundidad de secuenciación.
  Juntas garantizan que el tope del ranking sea el conjunto de k-mers
  presentes en *todas* las muestras, cuyo valor codificado es 1 en todas
  ellas: varianza cero, información cero. Medido sobre la primera cohorte
  real del estudio (80 genomas de *S. pneumoniae*, k=31): **460.795 k-mers
  presentes en los 80**. Con `top_features=500` (lo que usa la encuesta) o
  10.000 (el ejemplo de la propia docstring), la matriz que llegaba al
  clasificador era literalmente de unos.

  Lo que producía aguas abajo, y por qué ningún test lo alcanzaba:
  `fastdna.audit()` devolvía `score_random = 0.5000`,
  `score_lineage = 0.5000`, `gap = 0.0000` -- un informe con todos sus
  campos en rango que se lee como "esta cohorte no tiene fuga". Misma firma
  que los otros ocho defectos que encontró esta validación: *la salida
  equivocada estaba bien formada*.

  El arreglo va en la regla de ranking, no en un filtro añadido después:
  descartar solo las constantes promovería las presentes en 79 de 80
  muestras, cuya varianza es 0,0123. Para un feature binario la
  informatividad *es* la varianza, máxima en prevalencia n/2, así que
  `presence` rankea ahora por `min(prevalence, n_samples - prevalence)` y
  descarta de raíz las constantes -- el mismo criterio de *minor sample
  count* (el análogo del conteo de alelo menor) que `fastdna.gwas` ya
  aplicaba al truncar por `max_kmers`. Las representaciones con valor de
  conteo (`count`/`relative`/`clr`) conservan el ranking por prevalencia:
  ahí una k-mer universal sí varía, en profundidad, y el argumento original
  sigue en pie. El desempate de `presence` es por `kmer_u64` ascendente y
  deliberadamente **no** por `total_freq`: la profundidad es justo lo que
  esta codificación existe para ignorar. Con menos de dos muestras no hay
  varianza que rankear y se aplica la regla de conteo.

  Ambos caminos cambian a la vez: la ruta `disk_backed=True` lo implementa
  en Rust (`cohort_vocab::VocabularyRanking`, seleccionado por el argumento
  `ranking` nuevo de `_core.rank_cohort_vocabulary`), y los tests de
  equivalencia disk-backed/in-memory de `test_sklearn.py` fallan si las dos
  implementaciones se separan.

  Efecto medido hoy sobre un fixture independiente
  (`python/tests/test_genomic_model.py`: 12 muestras, marcador de 40 pb,
  `top_features=500`): con la regla anterior la matriz era de unos y
  `P(positivo | positivo retenido)` daba **0,5000 exacto**; con la nueva da
  **0,98**. Fijado por `python/tests/test_review_findings_2026_09_02.py`.

- **Python: `fastdna.datasets.load_amr()` descargaba en silencio el 26% de
  cada genoma.** El endpoint `genome_sequence` de BV-BRC está respaldado por
  Solr y pagina a 25 filas por defecto; una fila es un contig. Sin un
  `limit()` explícito, un ensamblado draft llegaba cortado en sus primeros 25
  contigs -- sin error, sin aviso, y como un FASTA perfectamente bien
  formado. Medido sobre el genoma 562.13671: 25 contigs / 1.261.838 bases sin
  el parámetro, frente a 98 contigs / 4.834.860 bases con él (*E. coli* son
  4,5-5,5 Mb). Toda cohorte devuelta por `load_amr` hasta ahora llevaba un
  cuarto de cada genoma, y lo heredaban todos los conteos de k-mers,
  sketches, asignaciones de linaje y modelos construidos sobre ella. Ningún
  test lo detectaba: `test_load_amr_real_network` comprobaba que el archivo
  empieza por ">", y un archivo truncado también lo hace. El test de
  regresión nuevo compara el número de bases contra el tamaño conocido de la
  especie, que es la única comprobación que los distingue.

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

- **La capa de machine learning entera.** 31 módulos de Python
  (`sklearn`, `audit`, `cv`, `explain`, `interpret`, `gwas`, `mic`,
  `embed`, `design`, `evaluation`, `calibration`, `anomaly`,
  `active_learning`, `genomic_model`, `rules`, `multiomics`, `datasets`,
  `equivalence`, `validate_generated`, `metagenomics`, `taxonomy`,
  `chimeras`, `annotate`, `translate`, `genomescope`, `assembly_qc`,
  `interop`, `plotting`, `report`, `provenance`, `workflow`), 4 módulos de
  Rust (`metagenomics.rs`, `chimera_scan.rs`, `translate.rs`,
  `cohort_vocab.rs`) y todo lo que existía para servirlos: sus bindings en
  `ffi.rs`, sus tests, sus scripts de validación y sus documentos de
  roadmap.

  **Esto rompe el contrato.** Cualquiera que importe alguno de esos módulos
  queda roto, sin período de deprecación. La versión 0.x lo permite; la
  honestidad exige decirlo así y no llamarlo "enfoque". Lo eliminado está
  en la historia de git en `60b5f82` y se puede recuperar de ahí.

  **Por qué.** Los nueve defectos serios que encontró este proyecto entre
  el 2026-08-31 y el 2026-09-05 estaban todos en esa capa, ninguno era
  alcanzable por los ~1.750 tests que existían, y cada uno se encontró
  contrastando un módulo ya entregado contra una verdad calculada fuera de
  él. Comparten una firma: *la salida equivocada estaba bien formada*. En
  el mismo período, el motor de conteo se comparó contra KMC3 sobre
  lecturas reales y dio **igualdad exacta** (19.062.700 distintos,
  163.051.083 totales). Una de las dos mitades está validada; la otra
  producía respuestas bien formadas y equivocadas más rápido de lo que se
  podían cazar. El razonamiento completo, con las cifras, está en
  `docs/goal-fast-kmer-counter.md`.

  Lo que queda es el CLI actual y su equivalente en Python: `count`,
  `sketch`, `dist`, `card`, `peek`, `query`, `union`, `intersect`, `diff`,
  `filter`, `matrix`, `profile`, `spectrum`.

- **Python: `fastdna.audit` y `fastdna.explain` salen de
  `fastdna.__all__`**, junto con la maquinaria de re-exportación perezosa
  que existía para que `fastdna.audit(...)` funcionara pese al choque de
  nombres con el submódulo. `__all__` queda en 18 nombres, todos de conteo.

- **`FastDnaError::VocabTooLarge` y su `VocabTooLargeError` en Python.** Ya
  no había forma de provocarlo: lo lanzaba únicamente el camino de
  vocabulario que se fue con `cohort_vocab.rs`. `MatrixTooLarge` se queda,
  que sí sigue vivo en `cohort/matrix.rs`.

- **Extras de `pyproject.toml`:** `scikit-learn`, `scipy`, `shap`,
  `umap-learn`, `biopython`, `matplotlib` y `pandas` salen del extra
  `test`. Ningún módulo del paquete importa ya ninguno de ellos; `numpy` es
  el único opcional que queda y se importa perezosamente.

- `src/cms.rs` eliminado por completo -- el `pub mod cms;` correspondiente
  desaparece de `src/lib.rs`. Rompe a cualquier consumidor externo de
  `fastdna_core::cms`.
- `examples/binned_occupancy_report.rs` eliminado, consecuencia directa de
  que `binned` pasó a `pub(crate)` (el ejemplo ya no podía usar esos tipos
  desde fuera del crate). El diagnóstico se movió a un test interno
  `#[ignore]` (`per_bin_occupancy_report`) en `src/binned.rs`.
