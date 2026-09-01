# Tarea 33 — `CohortCounts`: contar una vez, no una por fold

**Objetivo:** que una validación cruzada de 5 folds deje de leer cada FASTQ
cinco veces.

**Hallazgos que resuelve:** G-6 de `ml-gaps.md`.
**Prerequisitos:** tarea 00.
**Debe ir ANTES de:** tarea 12 (release) y tarea 14 (`audit()`).
**Archivos exactos:** `python/fastdna/cohort_counts.py` (nuevo),
`python/fastdna/sklearn.py`, `python/fastdna/__init__.py`,
`python/tests/test_cohort_counts.py` (nuevo).
**Esfuerzo:** 1 semana.

---

## El problema, con la aritmética

`python/fastdna/sklearn.py:365-390`:

```python
    def transform(self, X):
        check_is_fitted(self, "vocabulary_")
        paths = _validate_paths(X, "transform")
        return self._project(self._count_cohort(paths), len(paths))
```

`_count_cohort` (`sklearn.py:189-213`) llama a `fastdna.count(path, ...)` una
vez por ruta. **Sin caché de ningún tipo.**

`cross_val_score(pipeline, paths, y, cv=5)` hace, por cada fold:

| Llamada de sklearn | Qué cuenta |
|---|---|
| `Pipeline.fit(X_train)` → `vectorizer.fit_transform` | 80 % de las muestras |
| `Pipeline.score(X_test)` → `vectorizer.transform` | 20 % de las muestras |

= 100 % de la cohorte **por fold**. Con `cv=5`, **cinco pasadas completas**
sobre cada archivo FASTQ. `audit()` (tarea 14) hace CV aleatorio más CV por
linaje más sketching más la línea base solo-linaje: **~11 pasadas**.

Sobre 412 muestras de ~1 GB: **~4,5 TB de E/S y descompresión para auditar
un modelo.**

Ya se arregló la versión ×2 de este mismo bug dentro de un solo `fit` — el
docstring de `fit_transform` (`sklearn.py:342-364`) lo documenta. La
repetición entre folds sigue.

---

## Decisión de diseño (cerrada — no la reabras)

**`X` pasa a ser una lista de `sample_id`, y los conteos se le dan al
vectorizador en el constructor.**

```python
counts = fastdna.count_cohort("cohort/", k=31)        # UNA pasada
vec = KmerVectorizer(counts=counts, top_features=10_000)
cross_val_score(make_pipeline(vec, clf), counts.sample_ids, y, cv=5)
```

**Por qué así y no haciendo que `CohortCounts` sea indexable:** sklearn parte
`X` con `_safe_indexing`, que tiene ramas distintas para pandas, numpy,
sparse y secuencias. Un objeto propio caería en la rama de secuencia genérica
y el comportamiento dependería de detalles internos de sklearn que pueden
cambiar entre versiones. Una lista de cadenas es un tipo que sklearn parte de
forma nativa, documentada y estable. El artefacto pesado viaja por el
constructor, que es donde sklearn espera los hiperparámetros y donde
`clone()` lo preserva.

**La garantía anti-fuga se mantiene, y hay que decir por qué.** Lo que
`KmerVectorizer` impide es que **el vocabulario** se decida viendo el fold de
test. Contar es una operación por muestra que no mira ni las etiquetas ni las
otras muestras, así que contar todo por adelantado no filtra nada. Lo que no
puede salir del fold es `_learn_vocabulary`, y no sale: sigue recibiendo solo
las filas de los `sample_id` de entrenamiento.

**Compatibilidad:** si `counts=None`, `X` se sigue interpretando como rutas y
se cuenta como hoy. No se rompe nada.

---

## Código ANTES

`python/fastdna/sklearn.py:152-163` (constructor):

```python
    def __init__(self, k=31, min_count=1, top_features=10_000, threads=None):
        self.k = k
        self.min_count = min_count
        self.top_features = top_features
        self.threads = threads
```

---

## Código DESPUÉS

### Archivo nuevo `python/fastdna/cohort_counts.py`

```python
"""fastdna.cohort_counts -- los conteos de una cohorte, contados una vez.

## Por que existe

`KmerVectorizer` contaba cada FASTQ dentro de `fit()` y otra vez dentro de
`transform()`. Como scikit-learn llama a `fit` sobre el fold de
entrenamiento y a `transform` sobre el de test en *cada* particion, una
validacion cruzada de 5 folds leia la cohorte entera cinco veces, y
`fastdna.audit()` -- que corre dos esquemas de CV mas el sketching --
llegaba a once. Sobre 412 muestras de 1 GB son ~4,5 TB de E/S para evaluar
un modelo.

Contar es la operacion mas cara del paquete y su resultado es
determinista: la misma muestra con la misma `k` y el mismo `min_count` da
la misma tabla siempre. No hay ninguna razon para repetirla.

## Por que esto no reintroduce fuga de datos

La fuga que `KmerVectorizer` existe para impedir es que el *vocabulario* --
que k-mers se convierten en features -- se decida mirando muestras que
luego se evaluan. Contar no es eso: es una funcion de una sola muestra que
no mira ni las etiquetas ni las demas muestras. Contar toda la cohorte por
adelantado no puede filtrar nada, porque no hay ninguna decision tomada en
ese paso.

Lo que sigue estrictamente dentro del fold es `_learn_vocabulary`, que solo
ve las filas de los `sample_id` que `fit()` recibio.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa

import fastdna
from fastdna import _column_as_array

__all__ = ["CohortCounts", "count_cohort"]


@dataclass(frozen=True)
class CohortCounts:
    """Las tablas de conteo de una cohorte, apiladas y contadas una vez.

    `kmers` y `frequencies` son la concatenacion de las tablas por muestra,
    en el orden de `sample_ids`; `row_counts[i]` dice cuantas filas aporto
    la muestra `i`, que es lo que permite recortar por muestra sin volver a
    contar.

    `frozen=True` a proposito: es un artefacto de datos compartido entre
    todos los folds de una validacion cruzada, y que un fold pudiera
    mutarlo seria una fuente de fuga silenciosa.
    """

    sample_ids: tuple[str, ...]
    kmers: pa.Array
    frequencies: pa.Array
    row_counts: tuple[int, ...]
    k: int
    min_count: int

    def __len__(self) -> int:
        return len(self.sample_ids)

    @property
    def offsets(self) -> np.ndarray:
        """Indice de inicio de cada muestra dentro de `kmers`."""
        return np.concatenate([[0], np.cumsum(self.row_counts)]).astype(np.int64)

    def subset(self, sample_ids: Sequence[str]) -> "CohortCounts":
        """Los conteos de un subconjunto de muestras, sin recontar nada.

        Es lo que usa `KmerVectorizer` para quedarse con el fold que le
        toca. Levanta `KeyError` nombrando el id que falta en vez de
        devolver un subconjunto silenciosamente incompleto: un fold al que
        le faltan muestras produce un score plausible y equivocado.
        """
        position = {sid: i for i, sid in enumerate(self.sample_ids)}
        missing = [s for s in sample_ids if s not in position]
        if missing:
            raise KeyError(
                f"estos sample_id no estan en la cohorte contada: {missing[:5]}"
                f"{' ...' if len(missing) > 5 else ''}. "
                f"La cohorte tiene {len(self.sample_ids)} muestras."
            )

        starts = self.offsets
        take = np.concatenate(
            [np.arange(starts[position[s]], starts[position[s] + 1]) for s in sample_ids]
        ) if sample_ids else np.empty(0, dtype=np.int64)

        indices = pa.array(take, type=pa.int64())
        return CohortCounts(
            sample_ids=tuple(sample_ids),
            kmers=self.kmers.take(indices),
            frequencies=self.frequencies.take(indices),
            row_counts=tuple(
                self.row_counts[position[s]] for s in sample_ids
            ),
            k=self.k,
            min_count=self.min_count,
        )


def _sample_id_from_path(path) -> str:
    """El stem del archivo, sin la doble extension de FASTQ.

    Misma convencion que `multiomics._sample_id_from_filename`: `Path.stem`
    solo quita el ultimo sufijo, asi que el stem de `s.fastq.gz` es
    `s.fastq`, y un `.fastq` colgando de cada sample_id romperia cualquier
    join contra una hoja clinica.
    """
    name = os.path.basename(str(path))
    for suffix in (".gz", ".fastq", ".fq", ".fasta", ".fa"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return name


def count_cohort(
    samples,
    *,
    k: int = 31,
    min_count: int = 1,
    threads: int | None = None,
    progress=None,
) -> CohortCounts:
    """Cuenta cada muestra de la cohorte exactamente una vez.

    `samples` es un directorio, un `Mapping[sample_id, ruta]` o una
    secuencia de rutas (los ids salen del nombre del archivo).
    """
    if isinstance(samples, Mapping):
        pairs = [(str(sid), str(p)) for sid, p in samples.items()]
    elif isinstance(samples, (str, os.PathLike)) and os.path.isdir(samples):
        entries = sorted(
            os.path.join(samples, f)
            for f in os.listdir(samples)
            if f.lower().endswith((".fastq", ".fq", ".fastq.gz", ".fq.gz", ".fasta", ".fa"))
        )
        if not entries:
            raise ValueError(f"no se encontro ningun FASTQ/FASTA en {samples}")
        pairs = [(_sample_id_from_path(p), p) for p in entries]
    else:
        paths = [str(p) for p in samples]
        if not paths:
            raise ValueError("la cohorte esta vacia")
        pairs = [(_sample_id_from_path(p), p) for p in paths]

    seen: dict[str, str] = {}
    for sid, path in pairs:
        if sid in seen:
            raise ValueError(
                f"dos rutas producen el sample_id {sid!r} ({seen[sid]!r} y {path!r}); "
                "pasa un dict {sample_id: ruta} explicito para desambiguar."
            )
        seen[sid] = path

    kmer_arrays, freq_arrays, row_counts, ids = [], [], [], []
    for i, (sid, path) in enumerate(pairs):
        table = fastdna.count(path, k=k, min_count=min_count, threads=threads).table
        kmer_arrays.append(_column_as_array(table.column("kmer_u64")))
        freq_arrays.append(_column_as_array(table.column("frequency")))
        row_counts.append(table.num_rows)
        ids.append(sid)
        if progress is not None:
            progress(i + 1, len(pairs), sid)

    return CohortCounts(
        sample_ids=tuple(ids),
        kmers=pa.concat_arrays(kmer_arrays),
        frequencies=pa.concat_arrays(freq_arrays),
        row_counts=tuple(row_counts),
        k=k,
        min_count=min_count,
    )
```

### Cambios en `python/fastdna/sklearn.py`

Constructor:

```python
    def __init__(self, k=31, min_count=1, top_features=10_000, threads=None, counts=None):
        self.k = k
        self.min_count = min_count
        self.top_features = top_features
        self.threads = threads
        # Artefacto de cohorte ya contado. Cuando se da, `X` son
        # `sample_id` y no rutas, y ni `fit` ni `transform` vuelven a leer
        # un FASTQ: una validacion cruzada de 5 folds pasaba de 5 pasadas
        # completas sobre la cohorte a ninguna. Viaja por el constructor y
        # no por `X` porque sklearn parte `X` con `_safe_indexing`, cuyas
        # ramas para tipos propios dependen de detalles internos; una lista
        # de cadenas es un tipo que parte de forma nativa y estable.
        self.counts = counts
```

Y `_count_cohort` gana la rama de artefacto:

```python
    def _count_cohort(self, keys):
        """Las tablas de la cohorte para `keys`.

        Con `self.counts`, `keys` son `sample_id` y esto es un recorte del
        artefacto ya contado -- coste O(filas del subconjunto), sin E/S.
        Sin el, `keys` son rutas y se cuenta como siempre.
        """
        if self.counts is not None:
            subset = self.counts.subset(list(keys))
            return _CohortCounts(
                kmers=subset.kmers,
                sequences=None,
                frequencies=subset.frequencies,
                row_counts=list(subset.row_counts),
            )

        kmer_arrays, sequence_arrays, frequency_arrays, row_counts = [], [], [], []
        for path in keys:
            table = fastdna.count(path, k=self.k, min_count=self.min_count, threads=self.threads).table
            kmer_arrays.append(_column_as_array(table.column("kmer_u64")))
            sequence_arrays.append(_column_as_array(table.column("kmer_sequence")))
            frequency_arrays.append(_column_as_array(table.column("frequency")))
            row_counts.append(table.num_rows)
        return _CohortCounts(
            kmers=pa.concat_arrays(kmer_arrays),
            sequences=pa.concat_arrays(sequence_arrays),
            frequencies=pa.concat_arrays(frequency_arrays),
            row_counts=row_counts,
        )
```

> **Nota:** `sequences=None` en la rama del artefacto es deliberado —
> `CohortCounts` no guarda la columna de secuencia (ver tarea 25, que la
> apaga por defecto). `get_feature_names_out()` la reconstruye desde
> `kmer_u64` con `kmer.decode_kmer_into`. Si el ejecutor encuentra que
> `_learn_vocabulary` la usa, hay que cambiar esa ruta para decodificar en
> vez de leer; **no** volver a guardar 1,6 GB de cadenas.

---

## Tests completos

Archivo nuevo `python/tests/test_cohort_counts.py`:

```python
"""El artefacto de cohorte: mismo resultado, sin recontar.

Los dos tests que importan son la equivalencia (el atajo no puede cambiar
la respuesta) y el conteo de lecturas (el atajo tiene que ser realmente un
atajo). Sin el segundo, este modulo podria no estar ahorrando nada y los
tests pasarian igual.
"""
from __future__ import annotations

import numpy as np
import pytest

import fastdna

pytest.importorskip("sklearn")
from sklearn.linear_model import LogisticRegression       # noqa: E402
from sklearn.model_selection import cross_val_score        # noqa: E402
from sklearn.pipeline import make_pipeline                 # noqa: E402

from fastdna.sklearn import KmerVectorizer                 # noqa: E402


def _cohort(tmp_path, n=12):
    rng = np.random.default_rng(0)
    bases = np.array(list("ACGT"))
    paths = []
    for i in range(n):
        seq = "".join(rng.choice(bases, size=600))
        reads = [seq[j : j + 100] for j in range(0, 500, 50)]
        p = tmp_path / f"S{i:02d}.fastq"
        p.write_text("".join(f"@r{j}\n{r}\n+\n{'I' * len(r)}\n" for j, r in enumerate(reads)))
        paths.append(str(p))
    return paths


def test_artifact_gives_the_same_matrix_as_counting_from_paths(tmp_path):
    """El atajo no puede cambiar la respuesta."""
    paths = _cohort(tmp_path)
    counts = fastdna.count_cohort(paths, k=11)

    from_paths = KmerVectorizer(k=11, top_features=50).fit_transform(paths)
    from_artifact = KmerVectorizer(
        k=11, top_features=50, counts=counts
    ).fit_transform(list(counts.sample_ids))

    assert from_paths.shape == from_artifact.shape
    np.testing.assert_array_equal(from_paths.toarray(), from_artifact.toarray())


def test_cross_validation_does_not_reread_any_fastq(tmp_path, monkeypatch):
    """El punto entero de la tarea, medido.

    Con el artefacto, una CV de 5 folds tiene que hacer CERO llamadas a
    `fastdna.count`. Sin el, hace una por muestra y por fold.
    """
    paths = _cohort(tmp_path)
    counts = fastdna.count_cohort(paths, k=11)
    y = np.array([0, 1] * (len(paths) // 2))

    calls = {"n": 0}
    real_count = fastdna.count

    def counting_spy(*args, **kwargs):
        calls["n"] += 1
        return real_count(*args, **kwargs)

    monkeypatch.setattr(fastdna, "count", counting_spy)
    monkeypatch.setattr("fastdna.sklearn.fastdna.count", counting_spy)

    pipeline = make_pipeline(
        KmerVectorizer(k=11, top_features=50, counts=counts),
        LogisticRegression(max_iter=1000),
    )
    cross_val_score(pipeline, list(counts.sample_ids), y, cv=3)

    assert calls["n"] == 0, (
        f"la validacion cruzada leyo {calls['n']} FASTQ; con el artefacto "
        "no debe leer ninguno"
    )


def test_subset_preserves_per_sample_rows(tmp_path):
    paths = _cohort(tmp_path, n=6)
    counts = fastdna.count_cohort(paths, k=11)

    picked = [counts.sample_ids[0], counts.sample_ids[3]]
    sub = counts.subset(picked)

    assert sub.sample_ids == tuple(picked)
    assert len(sub.kmers) == sum(sub.row_counts)
    assert sub.row_counts == (counts.row_counts[0], counts.row_counts[3])


def test_subset_names_a_missing_sample_instead_of_returning_less(tmp_path):
    """Un fold al que le faltan muestras produce un score plausible y
    equivocado. Tiene que fallar, no encogerse."""
    paths = _cohort(tmp_path, n=4)
    counts = fastdna.count_cohort(paths, k=11)
    with pytest.raises(KeyError, match="no_existe"):
        counts.subset([counts.sample_ids[0], "no_existe"])


def test_duplicate_sample_ids_are_rejected(tmp_path):
    a = tmp_path / "sub_a"
    b = tmp_path / "sub_b"
    a.mkdir()
    b.mkdir()
    for d in (a, b):
        (d / "S1.fastq").write_text("@r\nACGTACGTACGT\n+\nIIIIIIIIIIII\n")
    with pytest.raises(ValueError, match="S1"):
        fastdna.count_cohort([str(a / "S1.fastq"), str(b / "S1.fastq")], k=6)


def test_counts_artifact_is_immutable():
    """Un fold que pudiera mutar el artefacto compartido seria una fuente
    de fuga silenciosa entre folds."""
    import dataclasses

    assert dataclasses.fields(fastdna.CohortCounts)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c = fastdna.CohortCounts(
            sample_ids=("a",), kmers=None, frequencies=None,
            row_counts=(0,), k=31, min_count=1,
        )
        c.k = 21
```

---

## Criterios de aceptación

- `cross_val_score` con `counts=` hace **cero** llamadas a `fastdna.count`.
- La matriz por artefacto es idéntica, elemento a elemento, a la de rutas.
- `fastdna.count_cohort` y `fastdna.CohortCounts` están en `__all__`.
- Los tests existentes de `python/tests/test_sklearn.py` siguen en verde sin
  modificarlos (la ruta de rutas no cambia).

## Comando exacto de verificación

```bash
python -m pytest python/tests/test_cohort_counts.py python/tests/test_sklearn.py -q
```

Salida esperada: `6 passed` de los nuevos, y los de `test_sklearn.py` en
verde.

## Qué NO tocar

- **No muevas `_learn_vocabulary` fuera del fold.** Es la garantía anti-fuga
  entera. Esta tarea mueve el *conteo*, que no toma ninguna decisión; el
  aprendizaje de vocabulario se queda donde está.
- No cambies la firma de `fastdna.count`.
- No añadas persistencia a disco en esta tarea (`CohortCounts.save()` es la
  tarea 36, junto con el vocabulario en streaming).

## Riesgos

- Si `_learn_vocabulary` resulta depender de `counts.sequences`, hay que
  hacer que decodifique desde `kmer_u64`. **No** vuelvas a materializar la
  columna de secuencias: son 1,6 GB en una cohorte real (ver tarea 25).
- El test del espía (`monkeypatch`) es frágil frente a cómo `sklearn.py`
  importa `fastdna`. Si el parcheo no intercepta, el test pasaría por el
  motivo equivocado. Verifica que **falla** sin `counts=` antes de darlo por
  bueno.
