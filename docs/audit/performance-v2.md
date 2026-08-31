# Contar más rápido — cinco ganancias no reclamadas

Fase 3 de [`PLAN.md`](PLAN.md). Todo lo de aquí está verificado contra el
repositorio o medido; nada es estimación de sobremesa. Donde no hay
medición posible en este entorno, se dice.

---

## Punto de partida, medido

Del propio `docs/BENCHMARKS.md` (2,14 GB FASTQ, k=31, 8 hilos, Windows
nativo):

```
FastDNA CLI de extremo a extremo    133,1 s
  de los cuales, contar             110,1 s   (82,7 %)
  de los cuales, exportar Parquet    23,0 s   (17,3 %)
```

Y del `docs/design-minimizer-counting.md` (WSL2, misma entrada):

```
FastK    38,3 s   2,86 GB
KMC3    110,4 s   9,06 GB
FastDNA ~101   s   8,02 GB (Windows nativo)
```

**Observación que cambia las prioridades:** casi una quinta parte del tiempo
de pared no es contar. Nadie ha mirado ahí.

---

## G-1 · `kmer_sequence` deja de ser obligatoria

**Ganancia: ~17 % del tiempo total y ~50 % del tamaño de archivo. Confianza
alta. Esfuerzo: 3 h.**

### Qué pasa hoy

`src/export.rs:44-50` fija el esquema de toda tabla de conteos:

```rust
pub fn counts_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("kmer_u64", DataType::UInt64, false),
        Field::new("kmer_sequence", DataType::Utf8, false),
        Field::new("frequency", DataType::UInt32, false),
    ]))
}
```

`kmer_sequence` es **enteramente derivable** de `kmer_u64` más `k`:
`kmer::decode_kmer_into(bits, k, &mut out)` la reconstruye exactamente.

### La aritmética

Sobre el archivo de referencia, 53 776 394 k-mers distintos a k=31:

| Columna | Bytes sin comprimir |
|---|---:|
| `kmer_u64` | 53 776 394 × 8 = **430 MB** |
| `frequency` | 53 776 394 × 4 = **215 MB** |
| `kmer_sequence` | 53 776 394 × 31 = **1 667 MB** |

La columna redundante es **2,6× más grande que las otras dos juntas**. Y
cuesta trabajo real en cada exportación: `ChunkBuffers::push`
(`src/export.rs:105-114`) llama a `decode_kmer_into` una vez por k-mer,
escribe 31 bytes, y actualiza el buffer de offsets; luego Snappy la
comprime, que es donde se va el grueso de los 23 s.

### Por qué es la mejor relación ganancia/esfuerzo del documento

No se optimiza nada: **se deja de hacer trabajo que casi nadie necesita**.
Quien trabaja en espacio `u64` —que es todo el camino de ML de FastDNA:
`sklearn.KmerVectorizer`, `cv`, `gwas`, y el propio `audit()`— nunca lee
esa columna. Solo la necesita quien va a mirar secuencias con los ojos o a
hacer un BLAST.

### La decisión

`with_sequence: bool`, **por defecto `false`**, en `count()` y en el
exportador; flag `--with-sequence` en el CLI.

Alternativa descartada: dejarla por defecto y añadir `--no-sequence`. El
default debe ser el caso mayoritario, y el mayoritario es el programático.
Es un cambio rompedor del esquema de salida, y por eso va **antes** del
primer release (fase 0/3), no después — luego sería permanente.

Compensación: `KmerCounts.with_sequence()` reconstruye la columna en Python
sin releer el FASTQ, así que nadie pierde la capacidad, solo el coste por
defecto.

### Verificación

```bash
fastdna --input bench2gb.fastq --output /tmp/a.parquet
fastdna --input bench2gb.fastq --output /tmp/b.parquet --with-sequence
ls -l /tmp/a.parquet /tmp/b.parquet
```

Esperado: `a.parquet` en torno a la mitad de `b.parquet`, y el tiempo total
bajando de ~133 s a ~115 s.

---

## G-2 · Promover `binned` a la elección automática

**Ganancia: el hueco documentado con KMC3. Confianza alta en memoria, media
en velocidad. Esfuerzo: ya está escrito; falta validación.**

### Estado real

El motor completo existe y está probado:

| Pieza | Archivo | Estado |
|---|---|---|
| Partición por minimizer | `src/minimizer.rs` (891 líneas) | ✅ probado |
| Super-k-mers empaquetados 2-bit | `src/superkmer.rs` (859) | ✅ 0,998 B/ocurrencia medido |
| Conteo por bins | `src/binned.rs` (1 065) | ✅ acuerdo exacto con `KmerCounter` |
| Mapa adaptativo de bins (estilo KMC2) | `src/adaptive_bins.rs` (446) | ✅ arregla el sesgo 1516× |
| Modelo de memoria | `src/mem_estimate.rs` | ✅ añadido 2026-08-25 |

**Y sin embargo `resolve_strategy` nunca lo elige** (`src/pipeline.rs:780-784`
solo decide entre `InMemory` y `Disk`). Solo se alcanza con
`--strategy binned`.

Es la diana correcta: el diagnóstico del propio proyecto es que KMC3
comprime la entrada en 70 635 757 super-k-mers desde 840 M de ocurrencias
—11,9× menos elementos que ordenar— y que ese es el mecanismo del hueco.
`binned` implementa exactamente eso, con **7,7× de reducción de bytes**
medida.

### Lo único que falta

`docs/design-minimizer-counting.md:1234` lo dice: el mapa adaptativo se
validó contra una **reproducción sintética** del sesgo de `DRR021372`, nunca
contra la muestra ENA real, por falta de red en el entorno donde se
implementó.

**Decisión: se descarga `DRR021372` y `DRR002015` y se re-mide.** Es una
tarde de trabajo y desbloquea el mayor cambio de rendimiento disponible.
Sin eso, promoverlo sería exactamente la clase de suposición que este
repositorio no se permite en ninguna otra parte.

### Después de validar

```rust
// en resolve_strategy, sustituyendo la eleccion binaria actual
let auto = if policy.estimated_input_bytes.is_none() {
    CountStrategy::InMemory
} else if binned_peak_bytes <= budget_bytes {
    // Preferido cuando cabe: el almacen de super-k-mers es una particion
    // entre workers, no una replica por worker, asi que el termino
    // `threads * occurrences` -- el que hace que una corrida de 8 hilos
    // cueste 8,34 GB -- no se reduce, desaparece.
    CountStrategy::Binned
} else if in_memory_peak_bytes <= budget_bytes {
    CountStrategy::InMemory
} else {
    CountStrategy::Disk
};
```

---

## G-3 · Arrays paralelos en vez de `Vec<(u64, u32)>`

**Ganancia: −25 % de bytes movidos en cada pasada de merge. Confianza alta.
Esfuerzo: 1 día.**

### Qué pasa hoy

`Vec<(u64, u32)>` es la representación de tabla de conteos en todo el crate
(`src/counter.rs`, `src/disk_spill.rs`, `src/binned.rs:291`, `:344`, `:469`).

`size_of::<(u64, u32)>()` es **16, no 12**: el `u64` fuerza alineación a 8 y
el `u32` final se rellena con 4 bytes de padding.

### La aritmética, que es del propio proyecto

`docs/design-minimizer-counting.md:862-870` ya lo calculó y lo dejó sin
hacer:

> `size_of::<(u64,u32)>()` es **16**, no 12 — el padding de alineación
> desperdicia 4 bytes por entrada. Sobre 53 776 394 k-mers distintos son
> **205 MiB tirados**, y el 25 % de los bytes movidos en cada pasada de
> merge. Vectores paralelos de claves `Vec<u64>` y cuentas `Vec<u32>` lo
> recuperan. **Esto es independiente de todo lo demás, son unas horas de
> trabajo, y probablemente debería hacerse primero.**

Se escribió el 2026-08-25 y sigue sin hacerse.

### Por qué importa más de lo que parece

El merge k-way (`counter::k_way_merge_sorted_counts`) está limitado por
ancho de banda de memoria, no por CPU: lee cada fuente una vez y escribe el
resultado una vez. Un cuarto menos de bytes es un cuarto menos de tráfico
en la fase que más memoria mueve de toda la corrida.

Además reduce el pico de RSS: el modelo de `estimate_binned_peak_bytes`
carga `COUNT_TABLE_BYTES_PER_ENTRY = 16`; con arrays paralelos son 12, y
como el término dominante del pico es `2 × tabla_final` durante el merge, el
pico predicho baja de 2,36 GiB a **1,84 GiB**.

### La decisión

Un tipo `CountTable { kmers: Vec<u64>, counts: Vec<u32> }` con los métodos
que hoy usan las tuplas. **No** se expone en la API pública (queda
`pub(crate)`, coherente con H-08); el FFI y el exportador siguen viendo
columnas, que es lo que Arrow quiere de todas formas — de hecho esta
representación es *más* cercana a Arrow que la de tuplas, así que
`build_record_batch` se simplifica en vez de complicarse.

---

## G-4 · Backend `zlib-ng` para gzip

**Ganancia: 2–3× en inflado, en la ruta de entrada normal. Confianza media
(no medible en este entorno). Esfuerzo: 30 min + medición.**

### Qué pasa hoy

`Cargo.toml:33`:

```toml
flate2 = "1.0"
```

Sin feature de backend, `flate2` usa **`miniz_oxide`**: Rust puro, portable,
y notablemente más lento que las implementaciones en C. Y la descompresión
está en el hilo productor, en serie con la lectura:
`src/fastq.rs:373` envuelve `MultiGzDecoder::new(file)` en un `BufReader`.

### Por qué importa aunque el benchmark no lo vea

**El archivo de referencia es FASTQ plano**, así que este coste no aparece
en ninguna cifra publicada. Pero los datos reales vienen comprimidos: el
propio código lo dice en `src/fastq.rs:366` — *"real SRA/ENA downloads
are…"* multi-miembro gzip. Para un usuario real, inflar está en el camino
crítico de cada corrida y hoy usa el backend más lento disponible.

### La decisión

```toml
flate2 = { version = "1.0", features = ["zlib-ng"] }
```

Alternativa descartada: `gzp` para inflado en paralelo. Solo funciona sobre
gzip multi-miembro o BGZF; un `.fastq.gz` de un solo miembro —lo habitual en
ENA— no se puede repartir sin descomprimirlo antes, así que la ganancia es
condicional a la forma del archivo. `zlib-ng` mejora **todos** los casos y
no añade una dependencia con condiciones.

Coste: `zlib-ng` necesita un compilador de C en tiempo de construcción. Los
wheels ya se construyen en contenedores manylinux que lo tienen, y Bioconda
también. **Hay que medirlo antes de darlo por bueno**, con un `.fastq.gz`
real, y comparar contra `miniz_oxide` en la misma máquina.

---

## G-5 · Minimizers con SIMD

**Ganancia: solo la ruta `binned`. Confianza media. Esfuerzo: 1 semana.**

`docs/design-minimizer-counting.md:502` ya cita el trabajo relevante
(`simd-minimizers`, doi:10.1101/2025.01.27.634998), que reporta cálculo de
minimizers un orden de magnitud más rápido con AVX2/NEON.

Se pone el último a propósito: solo importa **después** de G-2, porque hoy
la ruta `binned` no la usa nadie automáticamente. Y choca con una decisión
ya tomada y bien razonada en `Cargo.toml:78-82` — no fijar `target-cpu`
porque el proyecto distribuye wheels y un binario compilado para un CPU
concreto revienta en máquinas más viejas. La salida es **despacho en tiempo
de ejecución** (detectar AVX2 y elegir la implementación), no compilación
específica. Eso es lo que hace el trabajo de la semana.

---

## Lo que NO hay que hacer

| Idea | Por qué no |
|---|---|
| Tocar `lto`/`codegen-units` | `Cargo.toml:72-86` documenta que están sin medir *a propósito*, porque esta máquina oscila entre 25 y 90 s con el mismo binario. Ese razonamiento sigue vigente. |
| `-C target-cpu=native` | Ya rechazado y con razón: se distribuyen wheels. |
| Sustituir `sort_unstable` por radix LSD | Ya se probó y se revirtió con medición: **2,1–3,4× más lento**. Está documentado en `src/counter.rs`. |
| Subir `RAW_FINALIZE_THRESHOLD` | Barrido hecho el 2026-08-25: las dos formas de cobertura apuntan en direcciones opuestas y no hay ganador. |
| k > 32 vía `u128` | Es paridad de features, no velocidad. **Hará el conteo más lento**, no más rápido. Va en la fase 4 y por otra razón. |

---

## Orden y ganancia acumulada

| # | Tarea | Depende de | Ganancia |
|---|---|---|---|
| 25 | G-1 `kmer_sequence` opcional | — | −17 % tiempo, −50 % archivo |
| 27 | G-3 arrays paralelos | — | −25 % bytes en merge, −22 % pico RSS |
| 28 | G-4 `zlib-ng` | — | 2–3× inflado (entrada .gz) |
| 26a | Validar bins adaptativos contra `DRR021372` real | red | desbloquea 26b |
| 26b | G-2 promover `binned` | 26a, 27 | el hueco con KMC3 |
| 29 | G-5 SIMD minimizers | 26b | ruta `binned` |

**Regla, heredada de la cultura del repositorio:** ninguna de estas se da
por buena sin una medición en build **nativo de Windows** con
`Measure-Command`. Las cifras de WSL y Docker no valen aquí — este proyecto
ya documenta oscilaciones de 25 a 90 s con el mismo binario en esos
entornos. G-1 y G-3 son excepciones parciales: reducen trabajo de forma
demostrable por conteo (bytes escritos, bytes movidos), así que la medición
confirma la magnitud, no la dirección.
