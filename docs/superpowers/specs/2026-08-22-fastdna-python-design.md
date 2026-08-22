# FastDNA — Núcleo Rust distribuible como librería Python

**Fecha:** 2026-08-22
**Estado:** Diseño aprobado, pendiente de plan de implementación

---

## 1. Objetivo

Convertir FastDNA en una librería de Python con motor Rust que un bioinformático
pueda instalar con `pip install fastdna` en Linux, Windows o macOS **sin tener
Rust instalado**, y usar desde un notebook para pasar de FASTQ crudos a una
matriz lista para XGBoost.

### Criterios de éxito

1. `pip install fastdna` funciona en las 5 plataformas objetivo sin compilador.
2. `import fastdna; fastdna.count(...)` devuelve datos Arrow sin escribir a disco.
3. Una cohorte de 500 muestras produce una única matriz en una sola llamada.
4. El mismo binario corre en CPUs con y sin AVX2 sin recompilar.
5. El CLI existente sigue funcionando sin cambios en los scripts que ya lo invocan.

---

## 2. Estado actual del código (auditoría)

Hallazgos de la revisión del crate en su estado actual:

| Hallazgo | Detalle |
|---|---|
| **El crate no compila** | `pipeline.rs:109` pasa `&Vec<Vec<u8>>` a `insert_batch(&[u64])` → `error[E0308]` |
| **Lógica canónica duplicada** | `kmer.rs` tiene la versión correcta (bit-twiddling O(1), maneja `N`, con tests). `bio.rs` tiene una copia lenta sobre `Vec<u8>` que asigna en heap por ventana y no maneja `N` |
| **El pipeline usa la versión mala** | `pipeline.rs` llama a `bio::canonical_kmer`, no a `kmer::extract_canonical_kmers` |
| **WASM es correcto, nativo no** | `wasm.rs:26` ya usa `kmer::extract_canonical_kmers`. La divergencia existe porque la lógica biológica se incrustó en `pipeline.rs` en vez de vivir en un núcleo compartido |
| **4 módulos muertos** | `bio.rs` (1 consumidor: la línea rota), `cms.rs`, `sketch.rs`, `simd.rs` (0 consumidores) |
| **SIMD inerte** | La detección de runtime está anidada dentro del gate de compile-time `target_feature = "avx2"`, que es falso sin `.cargo/config.toml`. El bloque no existe en el binario compilado |
| **Repositorio sin commits** | `git log` reporta que la rama `master` todavía no tiene ningún commit. Todo el código fuente está sin versionar |

Nota positiva: `fastq.rs` ya normaliza `\r\n` correctamente, y `Cargo.toml` ya
declara `crate-type = ["cdylib", "rlib"]` — justo lo que PyO3 necesita.

---

## 3. Decisiones cerradas

| Decisión | Elección | Razón |
|---|---|---|
| Binding a Python | **PyO3 nativo, in-process** | Zero-copy vía Arrow, sin subprocess ni archivos intermedios |
| Escala objetivo | **Virus → humano WGS** | El formato de salida se elige explícitamente, nunca se infiere |
| Identidad de muestra | **Detección de pares R1/R2** | Contar R1 y R2 como pacientes distintos es un error silencioso |
| Profundidad de secuenciación | **Conteos crudos + metadatos** | La normalización pertenece al loop de cross-validation, no al binario |
| Motor de cohorte | **Spill a disco + ruta rápida en RAM** | RAM acotada a cualquier escala; sin spill cuando no hace falta |
| CLI | **Se conserva** como cliente delgado | Preserva el benchmark contra Python y los `.bat` existentes |

### Por qué conteos crudos y no normalizados

Con profundidades desiguales (50M lecturas vs 5M), los conteos crudos hacen que
XGBoost aprenda logística de laboratorio en vez de biología. Rust **no** resuelve
esto normalizando: emite crudos más la profundidad por muestra, para que la
normalización ocurra dentro del loop de CV, que es donde estadísticamente
corresponde y donde no filtra información del test set.

### Por qué la selección de features es no supervisada

El ranking de k-mers se hace por **prevalencia** (en cuántas muestras aparece),
nunca por correlación con la etiqueta. Rust no recibe etiquetas, así que cae
naturalmente del lado seguro. Seleccionar features mirando la variable objetivo
sobre la cohorte completa infla la métrica de validación; si alguna vez se hace,
va en Python y dentro del loop de CV.

---

## 4. Arquitectura

Un núcleo con tres clientes:

```
              núcleo Rust puro
    kmer · counter · cms · sketch · fastq
    qc · pipeline · cohort · select · error
                     │
      ┌──────────────┼──────────────┐
      │              │              │
  ffi/python     bin/main.rs     wasm.rs
   (PyO3)      (CLI+indicatif)   (existente)
```

**Reglas duras del núcleo:**

- Sin `println!`, sin `eprintln!`, sin `ProgressBar`, sin `.expect()` ni `.unwrap()`.
- Toda función pública devuelve `Result<T, FastDnaError>`.
- El progreso se emite por callback `Fn(Progress)`; cada cliente decide qué hacer
  con él (CLI → indicatif, Python → tqdm o nada, WASM → ignorar).

Esta separación es la que impide que se repita el bug actual: la lógica biológica
vive en un solo sitio y los tres clientes la comparten.

---

## 5. Fase 0 — Arreglar el build

Cambio mínimo, va solo y primero.

1. En `pipeline.rs`, reemplazar el bucle manual (líneas ~98-111) por:
   ```rust
   let canon_kmers = kmer::extract_canonical_kmers(&record.seq, k);
   local_counter.insert_batch(&canon_kmers);
   ```
2. Borrar `src/bio.rs` y su `pub mod bio;` en `lib.rs`. `hash_kmer` no tiene
   consumidores; no se preserva.
3. Eliminar el `use crate::kmer;` no usado que reporta el warning.

**Efecto:** compila, deja de asignar un `Vec` por ventana, y empieza a manejar
las bases `N` correctamente — la versión de `bio.rs` las trataba como carácter
literal, generando k-mers corruptos en los conteos.

**Verificación:** `cargo test` (los tests de `kmer.rs` ya cubren canónicos,
simetría del reverso complementario, y reinicio ante `N`).

---

## 6. Feature 2 — Filtros de frecuencia

### Semántica

`min_count` y `max_count` se aplican **por muestra**, sobre su propia tabla de
conteos, nunca sobre el agregado de la cohorte.

### Implementación

Nuevo método en `KmerCounter`:

```rust
pub struct PruneStats { pub dropped_min: u64, pub dropped_max: u64, pub kept: u64 }

pub fn prune(&mut self, min: u32, max: Option<u32>) -> PruneStats
```

Se invoca al terminar cada muestra, liberando RAM antes del merge o del spill.
El filtro en tiempo de export se conserva como red de seguridad.

### Limitación conocida y aceptada

Podar después **no baja el pico de RAM de una muestra individual** — la tabla ya
creció hasta su máximo. Bajar el pico de verdad requeriría una pre-pasada con
Count-Min Sketch (contar aproximado, luego retener solo lo que supera el umbral),
como hacen KMC y Jellyfish. **No se construye ahora.** El objetivo declarado es
no reventar el Parquet ni la RAM de Python, y para eso el filtro basta.

### CLI

`--min-count` (ya existe, default 1) y `--max-count` (nuevo, default sin límite).

---

## 7. Feature 3 — Motor de cohorte

### 7.1 Descubrimiento y agrupación de muestras

Se escanea el directorio buscando `*.fastq`, `*.fq`, `*.fastq.gz`, `*.fq.gz`.

Agrupación por `sample_id`: se quita la extensión y luego el sufijo de par
(`_R1`/`_R2`/`_1`/`_2`, también con `.` como separador).

| Situación | Comportamiento |
|---|---|
| `pac_001_R1.fastq.gz` + `pac_001_R2.fastq.gz` | Una muestra, dos archivos, conteos fusionados |
| `pac_001.fastq.gz` sin sufijo | Una muestra single-end |
| `pac_001_R1.fastq.gz` sin su `_R2` | **Warning explícito** de huérfano; se procesa como single-end |
| Directorio sin FASTQ reconocibles | Error, no matriz vacía |

Los `sample_id` se ordenan lexicográficamente para que el orden de filas de la
matriz sea **reproducible** entre ejecuciones y plataformas.

### 7.2 Estrategia de paralelismo

Paralelismo **entre muestras**, no dentro de cada una. Cada muestra se procesa
con una ruta secuencial y rayon reparte las muestras entre hilos.

Razón: 500 muestras son 500 unidades de trabajo independientes; paralelizar
dentro de cada una además del reparto externo sobre-suscribe el pool y añade
sincronización sin ganancia. El escalado queda casi lineal en número de núcleos.

`process_stream_parallel` se conserva para el modo de una sola muestra (`count`),
donde sí es la estrategia correcta.

### 7.3 Pasada 1 — Conteo y prevalencia

Por cada muestra, en paralelo:

1. Stream de su(s) FASTQ → `KmerCounter` (canónicos, trim de calidad).
2. `prune(min_count, max_count)`.
3. Registrar `SampleMeta`.
4. Actualizar la tabla global de prevalencia: `prevalence[kmer] += 1` para cada
   k-mer superviviente (una vez por muestra, no por ocurrencia).
5. Conservar el contador en RAM, o volcarlo a temporal.

```rust
pub struct SampleMeta {
    pub sample_id: String,
    pub files: Vec<PathBuf>,
    pub total_reads: u64,
    pub total_kmers: u64,          // tras trim, antes de prune — base de normalización
    pub distinct_raw: u64,
    pub distinct_pruned: u64,
    pub dropped_min: u64,
    pub dropped_max: u64,
}
```

`total_kmers` es la columna que Python usa para normalizar (CPM, CLR, log).

### 7.4 Ruta rápida vs spill

Se estima la RAM necesaria para mantener los contadores vivos
(`Σ distinct_pruned × 12 bytes`). Por debajo de `--max-ram` (default 4 GB) se
mantiene todo en memoria y **no se toca el disco**. Por encima, se vuelca.

**Formato temporal:** binario crudo, pares `(u64 kmer, u32 count)` en
little-endian. Sin Parquet: los temporales no se inspeccionan, y el overhead de
esquema y compresión no se justifica cuando se van a leer una sola vez.

**Ubicación:** `--temp-dir`, por defecto `std::env::temp_dir()`.

**Limpieza:** un guard con `Drop` borra los temporales también en caso de error o
excepción propagada a Python. No se dejan residuos.

### 7.5 Tabla de prevalencia y su guarda

Implementación v1: `FxHashMap<u64, u32>` exacto. Memoria ≈ `distinct_union × 12 B`.

Si la tabla supera `--max-vocab-ram` (default 2 GB), el programa **aborta con un
mensaje accionable**, sugiriendo subir `--min-count` o usar `--approx-vocab`.

`--approx-vocab` es la escotilla de escape: usa el `CountMinSketch` de `cms.rs`
para la prevalencia en memoria fija. Sobreestima, así que puede colar algún
k-mer espurio en el ranking — aceptable para ordenar por prevalencia, no para
conteos. **Se implementa solo si la guarda salta en uso real.**

### 7.6 Selección de features y formatos de salida

**`format = "wide"`** — matriz densa `n_muestras × top_features`.

Ranking por prevalencia descendente; empates se rompen por conteo total y luego
por `kmer_u64`, para que la selección sea determinista.

Guarda de tamaño: `n_muestras × n_features × 4 bytes`. Si supera
`--max-matrix-bytes` (default 4 GB), aborta con un mensaje que dice el tamaño
estimado y sugiere bajar `--top-features` o cambiar a `sparse`. **Nunca se
intenta escribir una matriz que no cabe.**

```
sample_id | ACGT…(1) | ACGT…(2) | … | ACGT…(N)
pac_001   |   142    |    0     | … |    87
```

**Semántica del cero.** Una celda vale `0` en dos casos distintos que la matriz
no distingue: el k-mer no apareció en esa muestra, o apareció pero por debajo de
`min_count` y fue podado. Es decir, la poda por muestra convierte conteos
sub-umbral en ceros explícitos.

Es el comportamiento correcto para el objetivo — esos conteos bajos son ruido de
secuenciación, que es justo lo que se quería eliminar — pero conviene tenerlo
presente al interpretar dispersión: parte de los ceros son "filtrado", no
"ausente". Quien necesite la distinción debe correr con `min_count=1` y filtrar
en Python.

**`format = "sparse"`** — dos tablas Arrow, sin selección top-N:

```
tripletas:    sample_idx (u32) | kmer_idx (u32) | count (u32)
vocabulario:  kmer_idx (u32)   | kmer_u64 (u64) | kmer_sequence (str)
```

Python reconstruye en una línea:
```python
csr = scipy.sparse.csr_matrix((t.count, (t.sample_idx, t.kmer_idx)))
```
XGBoost y scikit-learn consumen `csr_matrix` directamente, sin paso de
reconstrucción adicional.

### 7.7 Nota sobre Parquet vs Arrow

El techo práctico de ~10–20k columnas es un problema de **archivo Parquet** (el
footer de metadata crece por columna y por row-group), no de los datos. Cuando
la entrega es Arrow en memoria hacia Python, ese techo no aplica. Parquet queda
como formato de exportación opcional, no como mecanismo de transporte.

### 7.8 Proyección sobre un vocabulario fijo

Primitivo adicional, requerido por `KmerVectorizer.transform` (§9.6):

```rust
pub fn count_projected(
    samples: &[SampleFiles],
    vocabulary: &[u64],     // ordenado, define el orden de columnas
    opts: &CountOpts,
) -> Result<CohortMatrix, FastDnaError>
```

Cuenta muestras nuevas y las proyecta sobre un vocabulario **suministrado desde
fuera**, en lugar de derivarlo de las propias muestras. Los k-mers ausentes del
vocabulario se descartan; las columnas del vocabulario que la muestra no tiene
quedan en 0.

Es más simple que `count_cohort`: no hay pasada de prevalencia, no hay selección
de features, y por tanto **no hay spill** — cada muestra se cuenta, se proyecta y
se libera. Una sola pasada, memoria acotada por el tamaño del vocabulario.

Sin este primitivo no hay `transform`, y sin `transform` no hay integración con
sklearn ni prevención estructural de la fuga de datos. Es el habilitador del
nivel 2 completo.

---

## 8. Feature 4 — Comparación MinHash / Jaccard

### API

```python
fastdna.compare("virus1.fastq", "virus2.fastq", k=21, sketch_size=1000)  # -> 0.998
```

### Cambios sobre `sketch.rs`

`sketch.rs` es reutilizable pero necesita dos ajustes:

1. **Construcción por streaming.** `from_kmers(&[u64])` exige todos los k-mers en
   memoria. Se añade `from_stream` que consume el `FastqReader` incrementalmente
   y mantiene solo el bottom-k. Requisito para archivos grandes.
2. **Mejor finalizador de hash.** El hash actual es una multiplicación suelta
   (`wrapping_mul`). Se sustituye por un finalizador tipo splitmix64: el bottom-k
   selecciona por valor numérico, así que la calidad de la mezcla afecta
   directamente la precisión del estimador de Jaccard.

Se añade también `containment` junto a `jaccard`: Jaccard castiga la diferencia
de tamaño entre genomas, y la pregunta clínica frecuente —*¿está este patógeno
presente dentro de esta muestra metagenómica?*— es de contención, no de
similitud. Son ~5 líneas sobre el mismo sketch.

Se conserva el assert de `k` idéntico entre sketches — comparar sketches con `k`
distinto no tiene significado biológico.

### Persistencia del sketch

`Sketch` se serializa a disco (ya deriva `Serialize`/`Deserialize`). Esto elimina
el recálculo: comparar N muestras por pares con `compare(a, b)` haría O(N²)
lecturas de FASTQ; con sketches persistentes son N.

Como consecuencia, **la comparación N×N sale del "fuera de alcance"**: estaba
excluida porque implicaba un motor nuevo en Rust, y con sketches persistentes son
cinco líneas de Python sobre primitivas existentes (`fastdna.compare_all`).

---

## 9. Superficie pública de Python

### 9.0 Estructura del paquete: mixto Rust + Python

```
fastdna/
├── src/                       # núcleo Rust
├── python/fastdna/
│   ├── __init__.py            # API pública
│   ├── _core.{pyd,so}         # extensión PyO3 (privada, prefijo _)
│   ├── sklearn.py             # KmerVectorizer
│   ├── normalize.py           # CPM, CLR, log1p
│   └── spectrum.py            # detección del valle
└── pyproject.toml             # maturin, python-source = "python"
```

**La frontera FFI se mantiene deliberadamente pequeña.** Todo lo que sea
ergonomía —integración con sklearn, normalización, conversión a pandas— vive en
Python puro sobre `_core`. Razones:

1. Cada función que cruza la frontera hay que compilarla y probarla en 5
   plataformas. Cada función que vive en Python se prueba una vez.
2. Iterar sobre la API en Python no requiere recompilar nada.
3. `_core` con guion bajo señala que es privado: si mañana cambia la frontera,
   la API pública no se mueve.

Maturin soporta este layout nativamente vía `python-source`.

---

### 9.1 Nivel 1 — Núcleo (Rust vía PyO3)

```python
import fastdna

fastdna.count(paths, *, k=31, min_count=1, max_count=None,
              min_quality=20.0, threads=None, progress=None)  -> KmerCounts
fastdna.count_cohort(directory, *, k=31, min_count=1, max_count=None,
                     format="wide", top_features=10_000,
                     threads=None, progress=None)             -> Cohort
fastdna.sketch(path, *, k=21, sketch_size=1000)               -> Sketch
fastdna.peek(path, *, n_reads=10_000)                         -> Preview
fastdna.build_info()                                          -> dict
```

`progress` acepta `None` (silencio), `True` (tqdm si está disponible), o un
callable propio. El default es silencio: una librería no escribe en stdout sin
que se lo pidan.

`build_info()` reporta versión, `k` máximo, y **si AVX2 está activo en esta
CPU**. Sin eso, un "me va lento en el Mac" es indiagnosticable a distancia.

---

### 9.2 `KmerCounts`

```python
r = fastdna.count("muestra.fastq.gz", k=31, min_count=5)

r.table                  # pyarrow.Table (kmer_u64, kmer_sequence, frequency)
r.to_pandas()            # DataFrame
r.qc                     # dict de métricas de calidad
r.total_kmers            # base de normalización
r.distinct_kmers
r.top(20)                # los 20 k-mers más frecuentes

r.spectrum()             # {profundidad: nº de k-mers distintos}
r.suggest_min_count()    # -> int, detectado del espectro

r.save("muestra.counts.parquet")
fastdna.load_counts("muestra.counts.parquet")

len(r)                   # k-mers distintos
repr(r)                  # KmerCounts(k=31, distinct=104_882, total=8_931_204)
```

**`suggest_min_count()` merece explicación porque es la que más dolor evita.**
El espectro de frecuencias de una muestra secuenciada tiene dos picos: uno enorme
en frecuencia 1-2 (errores de máquina) y otro en la profundidad real de
cobertura. Entre ambos hay un valle. El umbral correcto está en ese valle, y
**depende de cada muestra** — no es un 5 universal.

Hoy tu usuario elegiría `min_count` a ojo. Con esto lo elige a partir de sus
datos, que es la diferencia entre tirar ruido y tirar señal. El motor ya calcula
el histograma (`KmerCounter::generate_histogram`); solo faltaba exponerlo y
poner la detección del mínimo local encima, en Python.

---

### 9.3 `Cohort`

```python
c = fastdna.count_cohort("./pacientes/", k=31, min_count=5, top_features=10_000)

c.matrix                 # pyarrow.Table
c.samples                # pyarrow.Table de SampleMeta — profundidad por muestra
c.vocabulary             # los k-mers elegidos como columnas
c.dropped                # qué se filtró y por qué

c.to_pandas()            # DataFrame (format="wide")
c.to_numpy()             # ndarray (format="wide")
c.to_scipy()             # csr_matrix (cualquier format)

c.normalize("cpm")       # -> Cohort normalizado; también "clr", "log1p", "relative"

c.save("cohorte.parquet")
fastdna.load_cohort("cohorte.parquet")

repr(c)                  # Cohort(500 muestras × 10_000 k-mers, k=31, format='wide')
```

`c.samples` viaja pegado a `c.matrix` y no como archivo suelto: es más difícil
normalizar mal cuando la profundidad está en el mismo objeto que los conteos. Por
eso `c.normalize()` no necesita argumentos extra — ya tiene lo que necesita.

`save`/`load_cohort` existen porque una cohorte de 500 muestras tarda decenas de
minutos. Perder eso al reiniciar el kernel de Jupyter es inaceptable.

`c.vocabulary` no es decorativo: es lo que permite proyectar muestras nuevas
sobre la misma base de features (§7.8), y es lo que hace posible el nivel 2.

---

### 9.4 `Sketch` — comparación como objeto, no como función

```python
s1 = fastdna.sketch("virus1.fastq", k=21)
s2 = fastdna.sketch("virus2.fastq", k=21)

s1.jaccard(s2)           # 0.998 — similitud simétrica
s1.containment(s2)       # ¿cuánto de s1 está dentro de s2? — asimétrico

s1.save("virus1.sig")
fastdna.load_sketch("virus1.sig")

fastdna.compare("a.fastq", "b.fastq", k=21)   # azúcar sobre lo anterior
fastdna.compare_all(["a.fastq", "b.fastq", "c.fastq"])  # matriz N×N
```

Convertir el sketch en objeto de primera clase resuelve dos cosas de golpe:

**No se recomputa.** `compare(a, b)` recalcula ambos sketches cada vez. Con N
muestras y comparaciones por pares eso es O(N²) lecturas de FASTQ cuando bastan
N. Sketch persistente = se lee cada archivo una vez.

**La comparación N×N deja de ser trabajo de Rust.** Estaba fuera de alcance
porque implicaba un motor nuevo; con sketches persistentes son cinco líneas de
Python sobre primitivas que ya existen. Pasa a estar dentro de alcance gratis.

`containment` responde una pregunta distinta a Jaccard y que en clínica se hace
mucho: *¿está este virus presente dentro de esta muestra metagenómica?* Jaccard
castiga la diferencia de tamaño entre genomas; containment no. Son ~5 líneas
sobre el mismo sketch.

---

### 9.5 `Preview` — inspección antes de comprometerse

```python
p = fastdna.peek("muestra.fastq.gz", n_reads=10_000)

p.n_reads_sampled
p.read_length            # (min, mediana, max)
p.mean_quality_by_position
p.gc_content
p.estimated_distinct_kmers
p.suggest_k()
```

Lee solo las primeras N lecturas: es cuestión de milisegundos.

Su razón de ser: `k=31` es el default de todo el mundo, y está mal para lecturas
cortas — con lecturas de 50 bp, `k=31` deja 20 k-mers por lectura y amplifica el
efecto de cada error. `peek` responde "¿qué k tiene sentido para *estos* datos?"
antes de lanzar un trabajo de 40 minutos, no después.

---

### 9.6 Nivel 2 — Integración con scikit-learn (Python puro)

```python
from fastdna.sklearn import KmerVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

pipe = Pipeline([
    ("kmers", KmerVectorizer(k=31, min_count=5, top_features=10_000)),
    ("clf",   XGBClassifier()),
])

scores = cross_val_score(pipe, fastq_paths, y, cv=5)
```

`KmerVectorizer` implementa la API de estimador de sklearn:

- `fit(paths, y=None)` — cuenta las muestras de entrenamiento, selecciona el
  vocabulario por prevalencia, lo guarda en `self.vocabulary_`.
- `transform(paths)` — proyecta las muestras sobre el vocabulario **ya aprendido**
  (§7.8). Los k-mers no vistos en `fit` se ignoran; los ausentes van a 0.
- `get_feature_names_out()` — devuelve las secuencias de k-mer.

**Esta es la pieza más importante de toda la API**, por una razón que no es de
comodidad sino de corrección estadística.

En §3 quedó anotado que seleccionar features mirando la cohorte completa infla la
métrica de validación, y que había que tener cuidado. Con `KmerVectorizer` ese
cuidado deja de depender de la disciplina de quien lo usa: dentro de un
`Pipeline`, sklearn llama a `fit` **solo sobre el fold de entrenamiento** en cada
iteración de la validación cruzada. El vocabulario nunca ve el fold de test.
**La fuga pasa de ser un riesgo documentado a ser estructuralmente imposible.**

Y `get_feature_names_out()` tiene un premio biológico: emparejado con
`model.feature_importances_`, te da las secuencias de ADN concretas que están
impulsando la predicción. Eso se puede meter en BLAST y preguntar a qué gen
pertenecen. Sin eso, tu modelo es una caja negra que acierta; con eso, es un
resultado publicable.

---

### 9.7 Nivel 3 — Helpers (Python puro)

```python
fastdna.normalize(matrix, samples, method="cpm")   # "cpm" | "clr" | "log1p" | "relative"
fastdna.spectrum_valley(hist)                      # detección del mínimo local
```

Viven en Python porque son aritmética sobre arrays que numpy ya hace bien, y
porque congelarlas en Rust sería justo el error que evitamos al decidir emitir
conteos crudos.

---

### 9.8 Qué queda fuera

- Soporte de manifiesto CSV (se eligió detección R1/R2).
- Iteración perezosa por muestra sobre una cohorte.
- Reanudación de cohortes interrumpidas.
- Alineamiento, ensamblado, o cualquier cosa que no sea conteo de k-mers.

---

### Frontera PyO3

- **El GIL se suelta** con `py.allow_threads` durante todo el trabajo pesado. Sin
  esto, una cohorte de 4 minutos congela el notebook: sin progreso, sin Ctrl-C.
- **Progreso por callback** opcional; sin él, silencio total. `indicatif` no cruza
  la frontera — sus códigos ANSI son basura visual en una celda de Jupyter.
- **Arrow zero-copy** vía la feature `pyarrow` del crate `arrow` (C Data Interface).

---

## 10. CLI

Tres subcomandos, con `count` **por defecto**:

```
fastdna count    --input muestra.fastq.gz --output counts.parquet
fastdna cohort   --input-dir ./pacientes/ --output matriz.parquet
fastdna compare  virus1.fastq virus2.fastq
```

Si no se escribe subcomando, clap cae en `count`. Esto preserva las 3
invocaciones existentes de la forma `fastdna --input X --output Y` en
`experiments/run_covid.bat` y `ml_pipeline/run_benchmark.sh`, que no se tocan.

Implementación: `Option<Commands>` más args aplanados; si `command` es `None`, se
usan los args aplanados como `count`.

---

## 11. Distribución multiplataforma

### Wheels con abi3

`pyo3/abi3-py38` compila contra la ABI estable de CPython: **un wheel por
plataforma cubre Python 3.8+**, en vez de uno por cada combinación
plataforma × versión (≈5 artefactos en lugar de ≈30).

```
manylinux x86_64 · manylinux aarch64 · macOS x86_64 · macOS arm64 · Windows x86_64
```

Construidos en GitHub Actions con `maturin-action` (trae contenedores manylinux y
cross-compile a ARM). `compile.bat` y el copiado manual del `.exe` a `bin/`
desaparecen.

### PyO3 tras feature opcional

`pyo3` va detrás de una feature `python`, activada solo por maturin — mismo
patrón que la feature `wasm` existente.

Razón: `pyo3/extension-module` indica al crate que **no** enlace contra
`libpython`, lo cual es correcto para un módulo de extensión pero rompe la
compilación del `[[bin]]`. Aislarlo tras una feature evita ese conflicto.

### Corrección de `simd.rs` (bloqueante para "cualquier plataforma")

Estructura actual, con la detección de runtime anidada dentro del gate de
compile-time:

```rust
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]   // falso por defecto
{
    if is_x86_feature_detected!("avx2") { ... }                 // nunca se compila
}
```

Consecuencias: hoy el bloque no existe en el binario. Y activar
`-C target-feature=+avx2` para "arreglarlo" sería peor — el wheel de Linux
moriría con `SIGILL` en CPUs sin AVX2, y en Apple Silicon (aarch64) ni compila.

**Corrección:** gate de compilación solo por `target_arch`; detección de runtime
*fuera* de él; `#[target_feature(enable = "avx2")]` sobre la función `unsafe`.
El mismo wheel usa AVX2 donde existe y cae a escalar donde no.

Dado que `simd.rs` no tiene consumidores, la alternativa válida es **borrarlo**.
Se decide al implementar, según si se conecta a `kmer.rs` o no.

---

## 12. Manejo de errores

```rust
pub enum FastDnaError {
    Io { path: PathBuf, source: std::io::Error },
    MalformedFastq { path: PathBuf, record: u64, reason: String },
    InvalidK { k: usize },
    NoSamplesFound { dir: PathBuf },
    MatrixTooLarge { estimated_bytes: u64, limit: u64 },
    VocabTooLarge { estimated_bytes: u64, limit: u64 },
    MismatchedK { left: usize, right: usize },
}
```

Traducción en cada cliente:

| Variante | Python | CLI |
|---|---|---|
| `Io` | `FileNotFoundError` / `OSError` | mensaje + exit ≠ 0 |
| `MalformedFastq` | `ValueError` con ruta y nº de registro | ídem |
| `InvalidK`, `MismatchedK` | `ValueError` | ídem |
| `MatrixTooLarge`, `VocabTooLarge` | `MemoryError` con tamaño y sugerencia | ídem |

Los errores de FASTQ malformado incluyen ruta y número de registro: en una
cohorte de 500 archivos, "parse error" sin ubicación es inservible.

---

## 13. Testing

**Unitarios**
- Canónicos: ya cubiertos en `kmer.rs`; añadir propiedad
  `canonical(revcomp(x)) == canonical(x)` sobre entradas aleatorias.
- Agrupación R1/R2: pares, single-end, huérfanos, nombres con puntos y guiones.
- `prune`: cuentas de descartados por min y por max en los bordes exactos.
- Selección de features: determinismo del desempate con empates provocados.
- Jaccard: conjuntos con solapamiento conocido (idénticos → 1.0, disjuntos → 0.0).

**Integración con fixtures**
- FASTQ diminutos con conteos calculados a mano.
- Cohorte sintética de ~6 muestras: se verifica forma de la matriz, orden de
  filas, y que `wide` y `sparse` codifican los mismos datos.
- Equivalencia de rutas: ruta-rápida-RAM y ruta-spill deben producir salida
  **byte-idéntica** sobre la misma entrada.

**Multiplataforma**
- La suite corre en CI sobre Linux, Windows y macOS.
- Test de FASTQ con terminadores `\r\n` y `\n`.

**Python**
- pytest contra el wheel construido: tipos de las columnas Arrow, round-trip a
  pandas, que las excepciones sean las clases correctas, y que el GIL se libera
  (un hilo de Python sigue avanzando durante una llamada larga).
- Round-trip de persistencia: `save` → `load` devuelve datos idénticos para
  `KmerCounts`, `Cohort` y `Sketch`.
- `suggest_min_count()` sobre un espectro sintético con valle conocido.
- `containment` asimétrico: con A ⊂ B, `A.containment(B) ≈ 1.0` mientras
  `A.jaccard(B)` es bajo. Es el test que demuestra por qué existen las dos.

**`KmerVectorizer` — la garantía anti-fuga**

Es la propiedad más importante que se puede testear, y se puede hacer directamente:

- `check_estimator` de sklearn sobre el transformador.
- `fit` sobre las muestras A,B,C y `transform` sobre D: los k-mers exclusivos de
  D **no** aparecen en la salida, y las columnas son exactamente
  `vocabulary_`, en el mismo orden.
- Dentro de un `Pipeline` con `cross_val_score`, registrar qué rutas recibe
  `fit` en cada fold y afirmar que **ninguna** pertenece al fold de test. Esto
  convierte la garantía de §9.6 en un test ejecutable, no en una nota al pie.
- `get_feature_names_out()` devuelve secuencias ACGT válidas de longitud `k`.

---

## 14. Fuera de alcance (YAGNI)

- Pre-pasada con Count-Min Sketch para bajar el pico de RAM por muestra.
- Normalización **dentro de Rust** (sí existe como helper en Python, §9.7).
- Selección de features supervisada.
- Reanudación / checkpointing de cohortes interrumpidas.
- Manifiesto CSV de muestras.
- Alineamiento, ensamblado, o cualquier análisis que no sea conteo de k-mers.

*Movido a dentro de alcance:* comparación N×N — dejó de requerir trabajo en Rust
al hacer `Sketch` persistente (§8).

---

## 15. Orden de implementación

| Fase | Contenido | Dependencias |
|---|---|---|
| **0** | Arreglar build; borrar `bio.rs` | — |
| **A** | Extraer núcleo; `Result` en todo; callback de progreso | 0 |
| **B** | Filtros de frecuencia (`prune`, `--max-count`) | A |
| **C** | PyO3 + maturin + wheels en CI; `count` + `peek` expuestos | A |
| **D** | `KmerCounts`, espectro, `suggest_min_count`, save/load | C |
| **E** | Motor de cohorte + `Cohort` | A, B, C |
| **F** | `count_projected` + `KmerVectorizer` (sklearn) | E |
| **G** | `Sketch` por streaming, persistencia, `compare_all` | C |
| **H** | Corregir o borrar `simd.rs` | A |

La fase C va deliberadamente pronto: es el criterio de éxito principal, y es
mejor descubrir los problemas de empaquetado multiplataforma con dos funciones
expuestas que con nueve.

La fase F depende de E porque `KmerVectorizer` necesita la selección de features
por prevalencia que vive en el motor de cohorte. Es la fase con más valor
por línea de código: convierte la librería de "genera matrices" en "es un
componente de scikit-learn".
