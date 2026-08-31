# PLAN — De contador de k-mers a la referencia de genética + ML

Fecha: 2026-08-26. Este documento fija la estrategia y el orden de ejecución.
Los datos que lo sustentan están medidos en
[`00-inventario.md`](00-inventario.md); los defectos concretos, en
[`01-auditoria.md`](01-auditoria.md).

---

## 0. La decisión estratégica

**FastDNA deja de venderse como contador de k-mers rápido y pasa a ser la
librería que va de FASTQ crudo a un modelo genómico en el que se puede
confiar.**

### Por qué no "el contador más rápido"

No es una preferencia, es aritmética que ya está en este repositorio:

| Evidencia | Fuente en el repo |
|---|---|
| FastK: 38,3 s · KMC3: 110,4 s · FastDNA: ~101 s (WSL2, misma entrada) | `docs/design-minimizer-counting.md:117-118` |
| El objetivo declarado del plan de rendimiento es *alcanzar a KMC3*, que ya va 2,9× detrás de FastK | `docs/PERFORMANCE_PLAN.md` |
| Predicción propia: "muy probablemente seguiremos ~2× detrás de FastK" | `docs/design-minimizer-counting.md:1239` |
| k máximo 32; KMC llega a 256, FastK a ~128 | `README.md:523`, verificado contra ambos repos |
| biotite migró su núcleo a Rust en jun-2026 y tiene PR abierto (#859) para `KmerTable` en Rust | `docs/philosophy-narrow-not-broad.md` |

La conclusión de ese último documento es literal: *"«un kernel Rust rápido
bajo una API Python» no es un diferenciador que FastDNA posea en solitario —
hay que asumir la convergencia, y competir en ese eje no basta."*

Competir en velocidad de conteo es correr contra Gene Myers en su propio
terreno. **Se sigue optimizando la velocidad** (fase 3 de este plan, con
ganancias concretas ya identificadas) — pero como *cómo* se consigue el
producto, no como *qué* se vende.

### Por qué sí "ML genómico honesto"

El problema no resuelto del campo es que los resultados de ML genómico no
replican, porque las poblaciones bacterianas y virales son clonales y un
split aleatorio de validación cruzada reparte casi-copias del mismo clon
entre train y test. El modelo aprende el linaje, no el fenotipo.
`python/fastdna/cv.py:1-50` ya documenta esto con la literatura: el análisis
de *PLOS Biology* 2025 sobre 24 000+ genomas, el benchmark de *Briefings in
Bioinformatics* 2024, y arXiv 2502.07749 nombrando la validación cruzada
filogenética como *"la herramienta estándar que le falta al campo"*.

**Territorio verificado como libre** (búsqueda del 2026-08-26):

| Herramienta | Qué cubre | Qué le falta |
|---|---|---|
| `trustcv` | CV leakage-aware genérico, incl. estructura espacial/filogenética | Hay que **traerle los grupos ya calculados** |
| `bioLeak` (R) | Detección y auditoría de fuga en ML biomédico | Es R, y parte de una matriz de features |
| `pyseer`, `kmersGWAS` | Asociación con corrección por estructura (LMM) | Asociación, no predicción; las features las traes tú |
| KMC3 / FastK / Jellyfish | Contar k-mers | Nada de ML, y nunca lo tendrán |
| `sourmash` | Sketching, búsqueda, `gather` | No predice |
| biotite | Toolkit general (convergiendo en Rust) | No es un pipeline opinionado |

**Ninguna va de lecturas crudas a modelo honesto en un solo proceso.**
FastDNA sí puede: `compare_all()` da distancias Mash en segundos,
`lineage_groups()` las vuelve clusters, `LineageKFold` un splitter de
sklearn — sin alineador, sin constructor de árboles, sin referencia.

**El foso no es el splitter. Es el camino vertical completo.**

### El riesgo que se ataca de frente, no se esconde

Un [bioRxiv de 2026 sobre pipelines robustos de predicción de AMR](https://www.biorxiv.org/content/10.64898/2026.06.28.734076v1)
encontró que el CV aleatorio a veces predice el rendimiento clínico
**mejor** que los splits filogenéticos. Eso contradice la narrativa simple
"filogenético = correcto".

**Decisión cerrada:** no se ignora, se incorpora. `audit()` reporta *ambos*
números precisamente porque el campo no lo tiene resuelto, y entrega la
herramienta para medir la brecha en cada cohorte concreta en lugar de
imponer una respuesta universal. Es la postura más fuerte *y* la más
honesta, y encaja con la cultura que el repositorio ya tiene.

### La frase

> **FastDNA es la librería que encuentra los confusores de tu ML genómico.**
> Estructura poblacional, profundidad de secuenciación, efecto de lote y
> falsa causalidad — medidos por defecto, desde las lecturas crudas.

(Formulación anterior, que ésta contiene: *"de FASTQ a un modelo en el que
puedes confiar"*. Ver [`ml-gaps.md`](ml-gaps.md) §tesis unificada.)

---

## 1. Fases y hitos

Cada fase termina en un hito comprobable. Ninguna fase empieza sin que la
anterior cumpla el suyo.

### Fase 0 — Existir (semanas) · BLOQUEA TODO LO DEMÁS

Hoy `pip install fastdna` devuelve *"No matching distribution found"*
(HTTP 404 verificado en PyPI **y** en crates.io). Hay ~32 000 líneas de
código con 93 % de cobertura y **cero usuarios posibles**.

| # | Tarea | Hallazgo | Esfuerzo |
|---|---|---|---|
| 00 | Commitear los 5 archivos pendientes (fija las líneas del plan) | — | 10 min |
| 01 | Excluir `.pyc` del wheel | H-01 | 15 min |
| 02 | Corregir la sección Installation del README | H-02 | 20 min |
| 03 | `env!("CARGO_PKG_VERSION")` en el CLI | H-23 | 20 min |
| 04 | `py.typed` + anotaciones de la API pública | H-04, H-05 | 2 h |
| 05 | `__all__` en `__init__.py` y 7 submódulos | H-09 | 1 h |
| 06 | `CHANGELOG.md` + política de compatibilidad | H-24 | 1 h |
| 07 | Jerarquía de excepciones en Python | H-10 | 3 h |
| 08 | CI: job de `cargo test` + `clippy` + `pytest` | H-26 | 3 h |
| 09 | Sitio de documentación (MkDocs + mkdocstrings) | H-20 | 5 h |
| 10 | `#[non_exhaustive]` en `FastDnaError` | H-25 | 30 min |
| 11 | `source` en `Export`/`Load` | H-11 | 2 h |
| 12 | **Tag v0.2.0 + publicar en PyPI y crates.io** | H-02 | 2 h |
| 13 | Enviar el recipe a Bioconda | H-03 | 1 h |

> **Hito 0:** en una máquina limpia, `pip install fastdna` instala,
> `python -c "import fastdna; fastdna.count(...)"` funciona, `mypy` ve los
> tipos, y `https://anzeledon.github.io/fastdna/` está en línea.

> **Actualización 2026-08-26.** Tras leer la capa ML real, la fase 1 crece:
> hay dos defectos que la bloquean (recuento por fold, conteos sin
> normalizar) y una segunda pieza de foso (`explain()`). Ver
> [`ml-gaps.md`](ml-gaps.md). **G-6 y G-7 son prerequisitos de la tarea 14 y
> van antes del release.**

### Fase 1 — El foso (1–3 meses)

| # | Tarea | Esfuerzo |
|---|---|---|
| 14 | `fastdna.audit()` — núcleo del reporte | 3 días |
| 15 | Métrica de confusión fenotipo↔linaje | 2 días |
| 16 | Atribución de features a linaje | 2 días |
| 17 | Bloque de procedencia reproducible | 1 día |
| 18 | `AssociationWorkflow` se niega a reportar el número ingenuo solo | 1 día |
| 19 | Benchmark contra dataset público de AMR con verdad conocida | 1 semana |
| 20 | `audit()` en el CLI y en las plantillas Nextflow/Snakemake | 2 días |
| **33** | **G-6 `CohortCounts`: contar una vez, no una por fold** | **1 semana** |
| **34** | **G-7 `representation="presence"` por defecto** | **2 días** |
| 35 | G-8 `fastdna.explain()` | 3 semanas |
| 36 | G-9 vocabulario en streaming (cohortes 10 000+) | 2 semanas |
| 37 | G-11 dataset público + notebook + tabla de reproducción | 1 semana |
| 38 | G-10 `GenomicModel` con comprobación de dominio | 2 semanas |
| **39** | **G-12 `validate_generated()`: validar salida de modelos fundacionales** | **1 semana** |
| **40** | **G-13 `covariates=` en `audit()`: lote, año, centro** | **1 semana** |

Diseño completo de `audit()` en [`audit-api.md`](audit-api.md); los huecos
de la capa ML y el diseño de `explain()`, en [`ml-gaps.md`](ml-gaps.md).

> **Hito 1:** `fastdna.audit()` sobre un dataset público publicado, con la
> tabla de inflación medida y reproducible por terceros.

### Fase 2 — Credibilidad y escala (3–6 meses)

| # | Tarea | Esfuerzo |
|---|---|---|
| 21 | Paper JOSS (mínimo) o de métodos sobre el audit (mejor) | 1 mes |
| 22 | Base de datos binaria de k-mers + consulta O(log n) — su propio S1 | 3 semanas |
| 23 | Operaciones de conjunto entre tablas — su propio S2 | 2 semanas |
| 24 | Podar la superficie Python (ver §2) | 1 semana |

> **Hito 2:** DOI citable, y una cohorte de 10 000 genomas auditada de
> extremo a extremo sin salirse del presupuesto de memoria.

### Fase 3 — Velocidad (en paralelo desde la fase 1)

Detalle completo, con las mediciones que lo sustentan, en
[`performance-v2.md`](performance-v2.md).

| # | Tarea | Ganancia estimada | Confianza |
|---|---|---|---|
| 25 | `kmer_sequence` deja de ser obligatoria en la salida | −17 % de tiempo total, −50 % de tamaño de archivo | alta |
| 26 | Promover `binned` a la elección automática | el hueco con KMC3 | alta |
| 27 | Arrays paralelos en vez de `Vec<(u64,u32)>` | −25 % de bytes movidos en cada merge | alta |
| 28 | Backend `zlib-ng` para gzip | 2–3× en inflado, que es la ruta normal | media |
| 29 | Minimizers con SIMD | solo la ruta `binned` | media |

> **Hito 3:** un número honesto, medido en build nativo de Windows, frente
> a KMC3 y FastK sobre la misma entrada.

### Fase 4 — Apuestas (6+ meses)

| # | Tarea | Por qué |
|---|---|---|
| 30 | k > 32 vía `u128` | Quita la objeción de paridad más citada |
| 31 | Cerrar el bucle k-mer → gen | Un k-mer significativo no vale nada sin anotación |
| 32 | Puente a foundation models (features como tokens) | Hacia ahí va "genética + ML" y nadie tiene el puente |

---

## 2. Qué se poda

`docs/philosophy-narrow-not-broad.md` argumenta con datos —SAMtools 57 470
citas frente a Biopython 5 675 por ~1 % del alcance; AnnData 1,68 M
descargas/mes frente a scikit-bio 101 280— que hay que ser estrecho.
**Ese principio se aplicó a Rust y luego se ignoró en Python:** 26 módulos,
11 967 líneas.

Ese mismo documento advierte del coste: *"Biopython todavía distribuye
`Bio.pairwise2` — obsoleto y marcado como redundante — cuatro años después
de marcarlo, porque demasiado código de terceros lo importa."* Cada módulo
publicado es un compromiso de varios años para un proyecto mantenido por
una persona.

**Decisión cerrada.** La regla es: *¿sirve al camino FASTQ → modelo honesto?*

| Módulo | Destino | Razón |
|---|---|---|
| `cv`, `sklearn`, `evaluation`, `calibration`, `gwas`, `workflow`, `interpret`, `embed` | **Núcleo** | Son el camino |
| `taxonomy`, `metagenomics`, `genomescope`, `assembly_qc`, `spectrum`, `annotate`, `translate`, `interop`, `equivalence` | **Se mantienen** | Producen o explican features |
| `mic`, `rules`, `anomaly`, `active_learning`, `multiomics` | **Congelar; extraer a `fastdna-contrib` en la fase 2** | Alcance que diluye; ninguno está en el camino |
| `plotting`, `report` | **Se mantienen** | Son la superficie del audit |
| `src/cms.rs` (247 líneas) | **Borrar** | Código muerto verificado; ya señalado como pasivo el 2026-08-24 |

Congelar ≠ borrar: se dejan de ampliar, se documentan como estables-sin-
desarrollo, y se sacan a un paquete aparte cuando exista. Extraer antes de
publicar es gratis; después es un breaking change permanente.

---

## 3. Cómo se posiciona cada competidor tras esto

| | Contar rápido | Sketch/búsqueda | Asociación | **ML honesto de extremo a extremo** |
|---|:---:|:---:|:---:|:---:|
| KMC3 | ✅ | ❌ | ❌ | ❌ |
| FastK | ✅✅ | ❌ | ❌ | ❌ |
| Jellyfish | ✅ | ❌ | ❌ | ❌ |
| sourmash | ➖ | ✅✅ | ❌ | ❌ |
| pyseer | ❌ | ❌ | ✅✅ | ❌ |
| trustcv / bioLeak | ❌ | ❌ | ❌ | ➖ (sin secuencias) |
| biotite | ➖ (convergiendo) | ➖ | ❌ | ❌ |
| **FastDNA** | ✅ | ✅ | ➖ (delega en pyseer) | **✅ único** |

La última columna es la que se defiende. Las otras tres son cómo se llega.

---

## 4. Índice de tareas

Cada tarea vive en `tasks/NN-slug.md`, es autocontenida —lleva dentro el
código ANTES/DESPUÉS y los tests completos, sin apuntar a otro documento— y
cabe en un commit de menos de una hora salvo donde se indique.

Orden topológico: `00` antes que todo. `01`–`11` en cualquier orden entre
sí. `12` exige `01`–`11`. `13` exige `12`. `14`–`20` exigen `12`. `25`–`29`
son independientes de la fase 1 y pueden solaparse.

---

## 5. Las cinco cosas de mayor impacto

1. **Publicar.** Nada de lo demás cuenta mientras `pip install` dé 404.
2. **`fastdna.audit()`.** Es el foso, y es un diagnóstico — los
   diagnósticos se citan porque los revisores los piden.
3. **Reportar los dos números.** La brecha entre CV aleatorio y CV por
   linaje *es* el producto; una cifra sola no lo es.
4. **`kmer_sequence` opcional.** −17 % de tiempo y la mitad de tamaño de
   archivo, por borrar una columna derivable de otra.
5. **Promover `binned`.** El motor de super-k-mers ya está escrito, probado
   y con bins adaptativos; solo está desconectado del selector automático.

### Por dónde empezar mañana

Tarea `00` (commitear lo pendiente) y tarea `01` (excluir `.pyc`): juntas
son 25 minutos y desbloquean toda la fase 0. Luego `03` y `12`, que son lo
que convierte el repositorio en una librería que alguien puede usar.
