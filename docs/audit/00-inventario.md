# 00 — Inventario real

Auditoría del 2026-08-26. **Todo lo de este documento fue ejecutado o leído**,
no inferido. Lo que no pude verificar va marcado `[ASUNCIÓN]`.

## Estado del árbol auditado

| Dato | Valor |
|---|---|
| Commit base | `7daa41f` "Add docs/PERFORMANCE_PLAN.md" |
| Archivos modificados sin commitear | 5 (`docs/PERFORMANCE_PLAN.md`, `python/fastdna/multiomics.py`, `src/counter.rs`, `src/mem_estimate.rs`, `src/pipeline.rs`) |
| Tags git | **ninguno** (`git tag -l` vacío) |
| CHANGELOG | **no existe** |
| Versión declarada | `0.1.0` (`Cargo.toml:3`) |

> **Precondición del plan.** La auditoría se hizo contra el árbol de trabajo
> *con* esas 5 modificaciones aplicadas. Todas las referencias
> `archivo:línea` de `01-auditoria.md` corresponden a ese estado. La tarea
> `00-preflight` del plan exige commitear esos cambios antes de empezar, para
> que las líneas no se muevan bajo el ejecutor.

## Tamaño

| Métrica | Valor | Cómo se obtuvo |
|---|---:|---|
| Líneas Rust (`src/`) | 20 974 | `cat src/*.rs src/cohort/*.rs \| wc -l` |
| Líneas tests integración (`tests/`) | 3 688 | `cat tests/*.rs \| wc -l` |
| Líneas Python (`python/fastdna/`) | 11 967 | `cat python/fastdna/*.py \| wc -l` |
| Líneas tests Python (`python/tests/`) | 12 058 | `cat python/tests/*.py \| wc -l` |
| Módulos Rust | 30 archivos `.rs` en `src/` | `find src -name '*.rs'` |
| Módulos Python | 26 archivos `.py` en `python/fastdna/` | `ls python/fastdna/*.py` |
| Ítems `pub` en Rust | **296** | `grep -rhE "^\s*pub (fn\|struct\|enum\|trait\|const\|type) " src/ \| wc -l` |

## Build y artefactos (medidos, `rust:1-slim-bookworm`, contenedor Linux)

```
rustc 1.98.0 (88d9e12ae 2026-08-18)
cargo 1.98.0 (797e8a9bc 2026-08-05)
```

| Artefacto | Bytes | Nota |
|---|---:|---|
| `cargo build --release` | **205 s** | con `lto = "fat"`, `codegen-units = 1` |
| `target/release/fastdna` | 7 872 664 (7,6 M) | binario CLI, **sin strip** |
| `target/release/fastdna` tras `strip` | 6 719 296 (6,5 M) | −14,6 % |
| `target/release/libfastdna_core.so` | 1 024 520 (1004 K) | cdylib |
| Wheel `cp38-abi3-manylinux_2_34_x86_64` | 1 007 127 (984 K) | `maturin build --release --features python` |
| Wheel descomprimido | 2 606 043 | |

Entradas más grandes del wheel:

```
   1 692 776  fastdna/_core.abi3.so
     201 731  fastdna-0.1.0.dist-info/sboms/fastdna.cyclonedx.json
      60 093  fastdna/gwas.py
      43 310  fastdna-0.1.0.dist-info/METADATA
      38 325  fastdna/rules.py
      34 848  fastdna/genomescope.py
      34 090  fastdna/__pycache__/__init__.cpython-312.pyc   <-- ver H-01
      32 977  fastdna/annotate.py
      32 029  fastdna/multiomics.py
      29 430  fastdna/__pycache__/multiomics.cpython-312.pyc <-- ver H-01
```

## Dependencias

**Directas de runtime (10)** — `cargo tree --depth 1 --edges normal`:

```
arrow 53.4.1 · clap 4.6.6 · crossbeam-channel 0.5.16 · flate2 1.1.9
indicatif 0.17.11 · parquet 53.4.1 · rayon 1.12.0 · rustc-hash 2.1.3
serde 1.0.229 · serde_json 1.0.151
```

**Total de crates en el grafo de runtime: 109.**
Opcionales: `pyo3 0.22` (feature `python`), `wasm-bindgen 0.2` +
`serde-wasm-bindgen 0.6` (feature `wasm`). Dev: `tempfile 3.14`.

Runtime Python: `pyarrow>=14` únicamente (`pyproject.toml:12`).

## Tests — ejecutados, no citados

`cargo test --all-targets` (contenedor Linux, sin features):

| Suite | Tests | Resultado |
|---|---:|---|
| lib (`src/`) | **358** | ok |
| 16 binarios de integración (`tests/`) | **138** | ok |
| **Total** | **496** | **0 fallos** |

`pytest python/tests` contra extensión real (`maturin develop --features python`):
**629 passed, 40 skipped, 1 xfailed, 0 fallos.**

### Cobertura Rust (`cargo llvm-cov --all-targets --summary-only`)

**TOTAL: 93,00 % regiones · 92,96 % líneas · 91,08 % funciones.**

Los cinco peores por regiones:

| Archivo | Regiones | Líneas | Funciones ejecutadas |
|---|---:|---:|---:|
| `src/error.rs` | 65,71 % | 70,31 % | 100 % |
| `src/main.rs` | 67,44 % | 64,52 % | 63,64 % |
| `src/cli.rs` | 81,82 % | 86,21 % | 66,67 % |
| `src/pipeline.rs` | 84,79 % | 87,98 % | 80,00 % |
| `src/sketch.rs` | 86,87 % | 85,02 % | 86,25 % |

`src/export.rs` tiene 88,20 % de regiones pero solo **59,38 % de funciones
ejecutadas** (13 de 32 nunca se llaman en ningún test).

**`src/ffi.rs` (1 570 líneas) y `src/wasm.rs` (84 líneas) no aparecen en la
tabla**: no se compilan sin `--features`, así que su cobertura es
**desconocida y no medida**. Ver H-14.

### Cobertura Python (`coverage run -m pytest`, con extras `[test]` completos)

Ver `01-auditoria.md` H-15 para el desglose; total **83,8 %** con los extras
mínimos, y el detalle por módulo está en esa sección.

## Lint y documentación

`cargo clippy --all-targets`: **8 avisos**, todos preexistentes, ninguno
`deny`. Ubicaciones exactas:

```
src/metagenomics.rs:1118:44   src/binned.rs:553:21    src/counter.rs:905:9
src/minimizer.rs:682:21       src/superkmer.rs:388:21 src/superkmer.rs:480:21
src/translate.rs:424:27 (x2)  tests/dual_strategy.rs:75:17
```

`Cargo.toml:56-60` declara `unwrap_used`, `expect_used`, `print_stdout`,
`print_stderr` como `deny` — se cumplen en producción; los módulos de test
los relajan con `#[allow(...)]` explícito (20 sitios).

`cargo doc --no-deps`: **1 aviso** —
`public documentation for 'Taxonomy' links to private item 'Taxonomy::from_taxa'`.

## Features de Cargo — todas compiladas y verificadas

| Comando | Resultado |
|---|---|
| `cargo check` (default) | ok |
| `cargo check --no-default-features` | ok |
| `cargo check --features wasm` | ok |
| `cargo check --features wasm --target wasm32-unknown-unknown` | ok |
| `cargo check --all-features` | ok |

`src/wasm.rs` contiene **0 módulos de test** (`grep -c "mod tests"` → 0).

## Distribución — verificado por HTTP

| Registro | URL consultada | Resultado |
|---|---|---|
| PyPI | `https://pypi.org/pypi/fastdna/json` | **HTTP 404 — no existe** |
| crates.io | `https://crates.io/api/v1/crates/fastdna` | **HTTP 404 — no existe** |
| Bioconda | `recipe/meta.yaml` presente, `sha256` = 64 ceros (placeholder) | **no enviado** |

El README instruye `pip install fastdna` en su línea 13. Ese comando falla
hoy para cualquier lector. Ver H-02.

CI (`.github/workflows/wheels.yml`): un solo job, `Wheels`. Construye y
prueba wheels en 5 plataformas. **No ejecuta `cargo test`, `cargo clippy`,
ni `pytest` fuera del contexto del wheel.** Sin job de lint, sin cobertura,
sin `cargo audit`/`cargo deny`. Ver H-19, H-20.

## API pública Rust

`src/lib.rs` declara **24 `pub mod`** más dos re-exports
(`FastDnaError`, `Result`, `Progress`, `ProgressFn`). No hay ninguna
distinción entre "API estable" y "detalle interno": todo módulo es público.

Ítems `pub` por módulo (los diez mayores):

```
40 metagenomics · 28 binned · 27 minimizer · 24 fastq · 19 translate
17 superkmer · 17 sketch · 16 counter · 13 adaptive_bins · 11 disk_spill
```

**`src/cms.rs` (247 líneas, Count-Min Sketch) es código muerto**: `grep -rn
"cms::"` fuera del propio archivo no devuelve nada. Sigue siendo
`pub mod cms` en `src/lib.rs:7`. Ya estaba señalado como pasivo en
`docs/feature-gap-analysis.md` (ítem S7, 2026-08-24) y no se ha resuelto.
Ver H-08.

## API pública Python

`python/fastdna/__init__.py` — 13 funciones y 3 clases de nivel superior:

```
class KmerCounts          (:85)
def   count(...)          (:262)
def   peek(path, *, n_reads=10_000)             (:334)
def   build_info()                              (:347)
class Sketch                                    (:355)
def   sketch(path, *, k=21, sketch_size=1000)   (:453)
def   load_sketch(path)                         (:465)
class FracSketch                                (:470)
def   frac_sketch(path, *, k=21, scale=1000)    (:531)
def   load_frac_sketch(path)                    (:547)
def   compare(path_a, path_b, *, k=21, sketch_size=1000)          (:552)
def   compare_all(paths, *, k=21, sketch_size=1000, metric=...)   (:564)
def   estimate_cardinality(path, *, k=31, precision=14)           (:614)
```

Privadas: `_column_as_array` (:20), `_pair_positions` (:41),
`_fallback_table_html` (:68).

### Tipado: ausente

| Métrica | Valor |
|---|---:|
| `py.typed` | **no existe** (`find python -name py.typed` vacío) |
| Ficheros `.pyi` | **0** |
| `def` de nivel superior en el paquete | 159 |
| … con retorno anotado (`-> `) | **23** (14,5 %) |
| Métodos (`    def `) | 103 |
| … con retorno anotado | **2** (1,9 %) |
| `def` de nivel superior en `__init__.py` | 13 |
| … con retorno anotado | **0** |

Ver H-03.

### `__all__`: inconsistente

**18 de 26 módulos lo definen; 8 no**, incluido `__init__.py`:

```
sin __all__: __init__, _progress, assembly_qc, embed, interop,
             interpret, rules, sklearn, spectrum
```

Ver H-04.

### Módulos Python que no tocan Rust (11 de 26, puro Python)

```
active_learning · anomaly · calibration · equivalence · interpret · mic
plotting · report · spectrum · taxonomy · workflow
```

## Superficie CLI vs. superficie de librería

`src/cli.rs` define **17 flags y 0 subcomandos** (`grep -c "Subcommand"` → 0).
El CLI hace **una sola cosa: contar**. La librería expone además, vía FFI,
`peek`, `sketch`, `load_sketch`, `frac_sketch`, `load_frac_sketch`,
`estimate_cardinality`, `translate_sequences`, `translate_file`,
`protein_kmers`, `build_database`, `build_info` — ninguna alcanzable desde
la línea de comandos. Ver H-05.

`docs/feature-gap-analysis.md` ya identificó esto como ítem **Q4** el
2026-08-24 y sigue sin hacerse.

## Documentación existente

| Archivo | Bytes |
|---|---:|
| `README.md` | 42 354 (790 líneas) |
| `docs/ARCHITECTURE.html` | 104 144 |
| `docs/design-minimizer-counting.md` | 77 143 |
| `docs/BENCHMARKS.md` | 18 774 |
| `docs/PERFORMANCE_PLAN.md` | 18 185 |
| `docs/ml-genomics-roadmap.md` | 13 699 |
| `docs/validation-real-data.md` | 10 845 |
| `docs/audit-2026-08-24.md` | 8 093 |
| `docs/CHECKPOINT-2026-08-25.md` | 6 591 |
| `docs/philosophy-narrow-not-broad.md` | 5 449 |
| `docs/feature-gap-analysis.md` | 5 156 |
| `docs/ml-differentiation-roadmap.md` | 5 299 |

**No hay sitio de documentación publicado** (no readthedocs, no GitHub
Pages, no `mkdocs.yml`/`docs/conf.py`). Toda la referencia de API vive
dentro del README, que documenta **5 de las 13 funciones públicas**
(`count`, `peek`, `build_info`, `sketch`, `estimate_cardinality`).
`compare`, `compare_all`, `frac_sketch`, `load_sketch`,
`load_frac_sketch`, `KmerCounts`, `Sketch`, `FracSketch` no tienen entrada
de referencia. Ver H-16.

## TODOs en el código

Solo uno real, y está obsoleto:

```
src/metagenomics.rs:669  TODO: delete this and call
                         `kmer::extract_canonical_kmers_into` once that
                         lands in `src/kmer.rs`
```

Ya aterrizó (`src/kmer.rs:171`). Ver H-06.

## Decisiones estratégicas ya tomadas por el proyecto (vinculantes para este plan)

`docs/philosophy-narrow-not-broad.md` (2026-08-25) rechaza explícitamente,
con evidencia: alineamiento/WFA, grafos pangenoma, álgebra de intervalos
genómicos y un puente DLPack/PyTorch. **Ninguna propuesta de
`03-ideas.md` contradice ese documento**; las que rozan el límite lo dicen.

Territorio propio declarado en ese mismo documento: capa de consulta sobre
el Parquet ordenado, operaciones de conjuntos entre tablas de k-mers,
filtrado de lecturas por contenido de k-mers, perfiles de k-mer por lectura,
y la capa ML (`python/fastdna/cv.py`).
