# Tarea 25 — `kmer_sequence` deja de ser obligatoria

**Objetivo:** dejar de escribir 1,6 GB de una columna derivable de otra.

**Hallazgos que resuelve:** ninguno de la auditoría; es G-1 de
`performance-v2.md`.
**Prerequisitos:** tarea 00. **Debe ir ANTES de la tarea 12** (es un cambio
rompedor del esquema de salida; después del primer release sería permanente).
**Archivos exactos:** `src/export.rs`, `src/ffi.rs`, `src/cli.rs`,
`src/main.rs`, `python/fastdna/__init__.py`.
**Esfuerzo:** 3 h.

## La aritmética

`src/export.rs:44-50` fija el esquema:

```rust
pub fn counts_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("kmer_u64", DataType::UInt64, false),
        Field::new("kmer_sequence", DataType::Utf8, false),
        Field::new("frequency", DataType::UInt32, false),
    ]))
}
```

`kmer_sequence` es **enteramente derivable** de `kmer_u64` más `k`. Sobre el
archivo de referencia, 53 776 394 k-mers distintos a k=31:

| Columna | Bytes sin comprimir |
|---|---:|
| `kmer_u64` | 430 MB |
| `frequency` | 215 MB |
| `kmer_sequence` | **1 667 MB** |

La columna redundante es 2,6× más grande que las otras dos juntas. Y el
export cuesta **23,0 s de los 133,1 s** de una corrida completa (17,3 %),
según `docs/BENCHMARKS.md:238`.

Quien trabaja en espacio `u64` —todo el camino de ML: `sklearn`, `cv`,
`gwas`, `audit()`— nunca lee esa columna.

## Decisión ya tomada

`with_sequence: bool`, **por defecto `false`**. Flag `--with-sequence` en el
CLI.

Alternativa descartada: dejarla por defecto y añadir `--no-sequence`. El
default debe ser el caso mayoritario, y el mayoritario es el programático.

Compensación obligatoria: `KmerCounts.with_sequence()` en Python reconstruye
la columna sin releer el FASTQ, así que nadie pierde la capacidad.

## Pasos

1. `counts_schema()` pasa a `counts_schema(with_sequence: bool)`.
2. `ChunkBuffers` solo llena `seq_bytes`/`seq_offsets` si se pidió.
3. `export_counts_parquet` y `export_counts_csv` toman el parámetro.
4. `src/ffi.rs`: `count()` gana el argumento; `build_record_batch` también.
5. `src/cli.rs`: flag `--with-sequence`.
6. `python/fastdna/__init__.py`: `count(..., with_sequence=False)` y el
   método `KmerCounts.with_sequence()`.

## Código ANTES

`src/export.rs:44-50`, pegado arriba.

## Código DESPUÉS

```rust
/// El esquema de toda tabla de conteos. `with_sequence` decide si se
/// incluye `kmer_sequence`.
///
/// Esa columna es enteramente derivable de `kmer_u64` mas `k` -- es lo que
/// hace `kmer::decode_kmer_into` -- y sobre el archivo de referencia son
/// 53 776 394 x 31 = 1 667 MB frente a los 430 MB de `kmer_u64` y los
/// 215 MB de `frequency`: 2,6 veces mas que las otras dos juntas. Escribir
/// esa columna es la mayor parte de los 23,0 s que el export ocupa de los
/// 133,1 s de una corrida completa.
///
/// Por defecto va apagada porque el consumidor mayoritario -- todo el
/// camino de ML de este paquete: `sklearn`, `cv`, `gwas`, `audit` -- opera
/// en espacio `u64` y nunca la lee. Quien la necesita es quien va a mirar
/// secuencias o a hacer BLAST, y la pide explicitamente o la reconstruye
/// con `KmerCounts.with_sequence()` sin releer el FASTQ.
pub fn counts_schema(with_sequence: bool) -> Arc<Schema> {
    let mut fields = vec![Field::new("kmer_u64", DataType::UInt64, false)];
    if with_sequence {
        fields.push(Field::new("kmer_sequence", DataType::Utf8, false));
    }
    fields.push(Field::new("frequency", DataType::UInt32, false));
    Arc::new(Schema::new(fields))
}
```

Y en `ChunkBuffers::push`:

```rust
    fn push(&mut self, kmer_bits: u64, k: usize, count: u32, with_sequence: bool) {
        self.kmers.push(kmer_bits);
        if with_sequence {
            kmer::decode_kmer_into(kmer_bits, k, &mut self.seq_bytes);
            self.seq_offsets.push(self.seq_bytes.len() as i32);
        }
        self.freqs.push(count);
    }
```

En `python/fastdna/__init__.py`, el método de compensación:

```python
    def with_sequence(self) -> KmerCounts:
        \"\"\"La vista actual con una columna `kmer_sequence` anadida,
        decodificada desde `kmer_u64`.

        No relee el FASTQ: la secuencia es una funcion pura del entero de
        64 bits y de `k`, asi que reconstruirla aqui cuesta un paso sobre
        las filas que ya tienes. Existe para que apagar la columna por
        defecto no le quite la capacidad a nadie -- solo el coste a quien
        no la usa.
        \"\"\"
        if "kmer_sequence" in self.table.column_names:
            return self

        bits = self.table.column("kmer_u64").to_numpy(zero_copy_only=False)
        k = self.k
        alphabet = np.frombuffer(b"ACGT", dtype=np.uint8)
        # Una columna por posicion de base, decodificada de una vez para
        # las N filas en vez de fila a fila.
        codes = np.empty((len(bits), k), dtype=np.uint8)
        for i in range(k):
            codes[:, k - 1 - i] = alphabet[(bits >> (2 * i)) & 0b11]
        sequences = pa.array([row.tobytes().decode("ascii") for row in codes], type=pa.string())

        table = self.table.append_column("kmer_sequence", sequences)
        return KmerCounts(self._raw, table.select(["kmer_u64", "kmer_sequence", "frequency"]))
```

## Tests completos

En `python/tests/test_api.py`:

```python
def test_sequence_column_is_off_by_default_and_reconstructible(tmp_path):
    \"\"\"La columna de secuencia es derivable de kmer_u64 y cuesta el 17 %
    del tiempo de una corrida, asi que va apagada por defecto. Este test
    fija las dos mitades del trato: que no este, y que se pueda recuperar
    sin releer el FASTQ.
    \"\"\"
    path = tmp_path / "s.fastq"
    path.write_text("".join(f"@r{i}\nACGTACGTACGTACGT\n+\n{'I'*16}\n" for i in range(5)))

    default = fastdna.count(str(path), k=8)
    assert "kmer_sequence" not in default.table.column_names

    explicit = fastdna.count(str(path), k=8, with_sequence=True)
    assert "kmer_sequence" in explicit.table.column_names

    # Reconstruida == pedida al motor, fila por fila.
    rebuilt = default.with_sequence()
    assert rebuilt.table.column("kmer_sequence").to_pylist() == \
           explicit.table.column("kmer_sequence").to_pylist()

    # Y el orden de columnas es el mismo por los dos caminos.
    assert rebuilt.table.column_names == explicit.table.column_names


def test_with_sequence_is_idempotent(tmp_path):
    path = tmp_path / "s.fastq"
    path.write_text("@r0\nACGTACGTACGT\n+\nIIIIIIIIIIII\n")
    counts = fastdna.count(str(path), k=6, with_sequence=True)
    assert counts.with_sequence().table.column_names == counts.table.column_names
```

En `tests/export_errors.rs` o `tests/pipeline_integration.rs`:

```rust
/// El esquema por defecto no lleva la columna de secuencia, y con la
/// bandera si. Que el fichero resultante sea sustancialmente menor es el
/// punto entero del cambio.
#[test]
fn parquet_without_the_sequence_column_is_substantially_smaller() {
    let dir = tempfile::tempdir().expect("tempdir");
    let fastq = dir.path().join("s.fastq");
    let mut text = String::new();
    for i in 0..2000 {
        text.push_str(&format!("@r{i}\nACGTACGTACGTACGTACGTACGTACGTACGT\n+\n{}\n", "I".repeat(32)));
    }
    std::fs::write(&fastq, text).expect("escribir fastq");

    let lean = dir.path().join("lean.parquet");
    let full = dir.path().join("full.parquet");
    // (invocar el pipeline con with_sequence=false y true respectivamente)

    let lean_size = std::fs::metadata(&lean).expect("lean").len();
    let full_size = std::fs::metadata(&full).expect("full").len();
    assert!(
        lean_size < full_size,
        "sin la columna de secuencia ({lean_size} B) debe pesar menos que con ella ({full_size} B)"
    );
}
```

## Criterios de aceptación

- `fastdna.count(path, k=31).table.column_names == ["kmer_u64", "frequency"]`.
- `--with-sequence` la devuelve.
- `KmerCounts.with_sequence()` produce exactamente la misma columna que el
  motor.
- El Parquet por defecto pesa menos que el que lleva la columna.

## Comando exacto de verificación

```bash
cargo test --all-targets
python -m pytest python/tests/test_api.py -q -k sequence_column
```

## Qué NO tocar

No cambies `kmer::decode_kmer_into` ni `decode_kmer`. Siguen siendo la
fuente de verdad de la decodificación; esta tarea solo decide **cuándo** se
llaman.

## Riesgos

**Es un cambio rompedor del esquema de salida.** Cualquier consumidor que
lea `kmer_sequence` de un Parquet de FastDNA deja de encontrarla. Por eso
va antes del primer release. Anótalo en `CHANGELOG.md` bajo
`Changed (breaking)`.
