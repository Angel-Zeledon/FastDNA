# Tarea 34 — `representation="presence"` por defecto

**Objetivo:** que un modelo entrenado con `KmerVectorizer` deje de poder
aprender profundidad de secuenciación en lugar de biología.

**Hallazgos que resuelve:** G-7 de `ml-gaps.md`.
**Prerequisitos:** tarea 00.
**Debe ir ANTES de:** tarea 12 (es un cambio rompedor del default).
**Archivos exactos:** `python/fastdna/sklearn.py`,
`python/tests/test_sklearn.py`.
**Esfuerzo:** 2 días.

---

## El problema

`python/fastdna/sklearn.py:301-311` devuelve frecuencias crudas:

```python
        data = np.asarray(counts.frequencies)[kept].astype(np.float64)
        return sparse.csr_matrix(
            (data, (rows, column[kept].astype(np.int64))),
            shape=(n_samples, len(self.vocabulary_)),
            dtype=np.float64,
        )
```

No hay normalización por profundidad en ninguna parte de `sklearn.py` ni de
`cv.py` (verificado con
`grep -niE "normaliz|depth|cpm|tpm|rarefac|relative.abundance"`).

### Por qué importa

Una muestra secuenciada a 100× tiene aproximadamente 5× los conteos de k-mer
de una secuenciada a 20×, **por razones puramente técnicas**. Un
`LogisticRegression` sobre conteos crudos puede separar las clases usando la
magnitud global de la fila, que es profundidad, no biología.

En cohortes clínicas reales la profundidad correlaciona con el lote, el año y
el centro de secuenciación — que a su vez correlacionan con el fenotipo. Es
exactamente la estructura del problema que `cv.py` existe para atacar, en un
eje distinto: un confusor técnico que el modelo aprende y el score premia.

### La inconsistencia interna

Los dos caminos del paquete ya discrepan:

| Camino | Qué hace | Dónde |
|---|---|---|
| GWAS (`workflow.py`) | **binariza** a presencia/ausencia | `workflow.py:419-432` |
| sklearn (`KmerVectorizer`) | conteos crudos | `sklearn.py:301` |

`workflow.py:423` lo comenta: *"cohort_presence_matrix() stores raw
per-sample depth, not presence… setting every stored value to 1 recovers
plain 0/1 presence"*. El camino de asociación sabe que hay que hacerlo; el de
predicción no lo hace.

---

## Decisión de diseño (cerrada)

**El default pasa a `representation="presence"`.**

Razones:
1. Es lo que hace el otro camino de este mismo paquete.
2. Es lo que hacen pyseer, kmersGWAS y DBGWAS.
3. Es inmune a la profundidad por construcción.
4. Para la mayoría de preguntas genómicas —¿tiene este aislado este gen?—
   presencia *es* la pregunta.

Alternativa descartada: mantener `count` por defecto y documentar el riesgo.
La documentación no impide el fallo; el default sí. Y este paquete ya eligió
la vía estructural en `KmerVectorizer` («cierra la puerta estructuralmente en
vez de por documentación o disciplina», dice su propio docstring de módulo).

**`"count"` sigue disponible**, porque en metagenómica la abundancia es la
señal — pero emite `DepthConfoundingWarning` si la profundidad varía más de
3× dentro de la cohorte, **con la cifra medida en el mensaje**.

**Se añade `"clr"`** (log-ratio centrado), que es la transformación correcta
para datos composicionales de metagenoma y hoy no existe en el paquete.

Es un cambio rompedor y por eso va antes del primer release.

---

## Código ANTES

`python/fastdna/sklearn.py:152-163`:

```python
    def __init__(self, k=31, min_count=1, top_features=10_000, threads=None):
        self.k = k
        self.min_count = min_count
        self.top_features = top_features
        self.threads = threads
```

`python/fastdna/sklearn.py:287-311` (`_project`), pegado arriba.

---

## Código DESPUÉS

Constructor:

```python
    def __init__(
        self,
        k=31,
        min_count=1,
        top_features=10_000,
        threads=None,
        counts=None,
        representation="presence",
    ):
        self.k = k
        self.min_count = min_count
        self.top_features = top_features
        self.threads = threads
        self.counts = counts
        # "presence" por defecto, no "count".
        #
        # Una muestra secuenciada a 100x tiene ~5x los conteos de k-mer de
        # una a 20x por razones puramente tecnicas, y en cohortes clinicas
        # la profundidad correlaciona con lote, ano y centro -- que
        # correlacionan con el fenotipo. Un modelo lineal sobre conteos
        # crudos puede separar las clases por la magnitud de la fila, que
        # es profundidad y no biologia: el mismo tipo de confusor tecnico
        # que `fastdna.cv` ataca en el eje de estructura poblacional.
        #
        # Es ademas lo que ya hace el otro camino de este paquete:
        # `workflow.py` binariza la matriz antes de asociar, y lo comenta.
        # Y es lo que hacen pyseer, kmersGWAS y DBGWAS.
        #
        # `count` sigue disponible porque en metagenomica la abundancia es
        # la senal, pero avisa cuando la profundidad varia lo bastante como
        # para dominar. `clr` es la transformacion correcta para datos
        # composicionales.
        self.representation = representation
```

Aviso y `_project`:

```python
class DepthConfoundingWarning(UserWarning):
    """La profundidad de secuenciacion varia lo bastante entre muestras
    como para que un modelo entrenado sobre conteos crudos pueda estar
    aprendiendola en vez de aprender biologia."""


_VALID_REPRESENTATIONS = ("presence", "count", "relative", "clr")


    def _project(self, counts, n_samples):
        """La matriz dispersa `(n_samples, len(vocabulary_))` para una
        cohorte ya contada, en la representacion pedida.

        `index_in` resuelve el `kmer_u64` de cada fila contra el
        vocabulario en una pasada de hash en C++ (los k-mers fuera de el
        vuelven nulos, el caso "se ignoran en silencio" que documenta
        `transform()`).
        """
        if self.representation not in _VALID_REPRESENTATIONS:
            raise ValueError(
                f"representation debe ser uno de {list(_VALID_REPRESENTATIONS)}, "
                f"se recibio {self.representation!r}"
            )

        column = np.asarray(
            pc.fill_null(pc.index_in(counts.kmers, value_set=pa.array(self.vocabulary_)), -1)
        )
        kept = column >= 0
        rows = np.repeat(np.arange(n_samples, dtype=np.int64), counts.row_counts)[kept]
        data = np.asarray(counts.frequencies)[kept].astype(np.float64)

        if self.representation == "presence":
            # Toda entrada almacenada es por construccion >= 1, asi que
            # ponerlas todas a 1 recupera presencia 0/1 sin tocar el patron
            # de dispersion. Mismo razonamiento que `workflow.py:419-432`.
            data = np.ones_like(data)
        else:
            # Profundidad por muestra: la suma de conteos de las filas de
            # esa muestra dentro del vocabulario.
            depth = np.bincount(rows, weights=data, minlength=n_samples)
            nonzero = depth[depth > 0]
            if nonzero.size and nonzero.max() / nonzero.min() > 3.0:
                warnings.warn(
                    f"la profundidad varia {nonzero.max() / nonzero.min():.1f}x entre "
                    f"muestras ({nonzero.min():.0f} a {nonzero.max():.0f} conteos dentro "
                    f"del vocabulario) y representation={self.representation!r} conserva "
                    "esa magnitud. Un modelo puede separar las clases por profundidad en "
                    "vez de por biologia. Usa representation='presence' salvo que la "
                    "abundancia sea la senal que buscas.",
                    DepthConfoundingWarning,
                    stacklevel=2,
                )

            if self.representation == "relative":
                with np.errstate(divide="ignore", invalid="ignore"):
                    data = data / np.where(depth[rows] > 0, depth[rows], 1.0)
            elif self.representation == "clr":
                # Log-ratio centrado sobre las entradas presentes de cada
                # muestra. Los ceros estructurales se quedan fuera de la
                # matriz dispersa, que es lo correcto: el CLR se define
                # sobre las partes observadas, y materializar los ceros
                # convertiria una matriz dispersa de 10 000 columnas en una
                # densa.
                with np.errstate(divide="ignore", invalid="ignore"):
                    logs = np.log(data)
                    mean_log = np.bincount(rows, weights=logs, minlength=n_samples)
                    per_row = np.bincount(rows, minlength=n_samples)
                    mean_log = mean_log / np.where(per_row > 0, per_row, 1)
                    data = logs - mean_log[rows]

        return sparse.csr_matrix(
            (data, (rows, column[kept].astype(np.int64))),
            shape=(n_samples, len(self.vocabulary_)),
            dtype=np.float64,
        )
```

Añadir `import warnings` a la cabecera y `DepthConfoundingWarning` a
`__all__`.

---

## Tests completos

Añadir a `python/tests/test_sklearn.py`:

```python
def _uneven_depth_cohort(tmp_path):
    """Dos grupos con el MISMO contenido biologico y profundidad muy
    distinta. Cualquier separacion que un modelo encuentre aqui es
    profundidad, porque no hay otra cosa que encontrar."""
    import numpy as np

    rng = np.random.default_rng(0)
    bases = np.array(list("ACGT"))
    genome = "".join(rng.choice(bases, size=800))
    reads = [genome[i : i + 100] for i in range(0, 700, 25)]

    paths, depth_group = [], []
    for i in range(12):
        deep = i % 2 == 0
        repeats = 10 if deep else 1          # 10x de diferencia de profundidad
        p = tmp_path / f"S{i:02d}.fastq"
        p.write_text(
            "".join(
                f"@r{j}\n{r}\n+\n{'I' * len(r)}\n"
                for j, r in enumerate(reads * repeats)
            )
        )
        paths.append(str(p))
        depth_group.append(int(deep))
    return paths, np.array(depth_group)


def test_presence_is_the_default_representation(tmp_path):
    paths, _ = _uneven_depth_cohort(tmp_path)
    matrix = KmerVectorizer(k=11, top_features=100).fit_transform(paths)
    values = set(np.unique(matrix.data).tolist())
    assert values <= {1.0}, (
        f"el default debe ser presencia 0/1; se encontraron valores {sorted(values)[:5]}"
    )


def test_presence_makes_depth_unlearnable_where_counts_do_not(tmp_path):
    """El sintoma exacto. Dos grupos de muestras con contenido biologico
    identico y 10x de diferencia de profundidad: sobre conteos crudos un
    modelo separa los grupos perfectamente, sobre presencia no puede.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    paths, depth_group = _uneven_depth_cohort(tmp_path)

    counts_matrix = KmerVectorizer(
        k=11, top_features=100, representation="count"
    ).fit_transform(paths)
    presence_matrix = KmerVectorizer(
        k=11, top_features=100, representation="presence"
    ).fit_transform(paths)

    on_counts = cross_val_score(
        LogisticRegression(max_iter=1000), counts_matrix, depth_group, cv=3
    ).mean()
    on_presence = cross_val_score(
        LogisticRegression(max_iter=1000), presence_matrix, depth_group, cv=3
    ).mean()

    assert on_counts > 0.9, (
        "sobre conteos crudos la profundidad deberia ser trivialmente "
        f"aprendible; dio {on_counts:.2f} y el test no prueba nada"
    )
    assert on_presence < 0.8, (
        f"sobre presencia la profundidad no deberia ser aprendible; dio {on_presence:.2f}"
    )


def test_count_representation_warns_about_uneven_depth(tmp_path):
    import pytest

    from fastdna.sklearn import DepthConfoundingWarning

    paths, _ = _uneven_depth_cohort(tmp_path)
    with pytest.warns(DepthConfoundingWarning, match=r"\d+\.\dx"):
        KmerVectorizer(k=11, top_features=100, representation="count").fit_transform(paths)


def test_presence_does_not_warn(tmp_path):
    import warnings

    from fastdna.sklearn import DepthConfoundingWarning

    paths, _ = _uneven_depth_cohort(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DepthConfoundingWarning)
        KmerVectorizer(k=11, top_features=100).fit_transform(paths)


def test_unknown_representation_lists_the_valid_ones(tmp_path):
    import pytest

    paths, _ = _uneven_depth_cohort(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        KmerVectorizer(k=11, representation="binary").fit_transform(paths)
    message = str(excinfo.value)
    assert "binary" in message and "presence" in message


def test_relative_rows_sum_to_one(tmp_path):
    paths, _ = _uneven_depth_cohort(tmp_path)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matrix = KmerVectorizer(
            k=11, top_features=100, representation="relative"
        ).fit_transform(paths)
    sums = np.asarray(matrix.sum(axis=1)).ravel()
    np.testing.assert_allclose(sums, 1.0, rtol=1e-9)
```

---

## Criterios de aceptación

- `KmerVectorizer().fit_transform(paths)` produce una matriz cuyos valores
  almacenados son todos `1.0`.
- Sobre la cohorte de profundidad desigual, un clasificador separa los grupos
  con `representation="count"` y **no** con `"presence"`.
- `representation="count"` emite `DepthConfoundingWarning` con la cifra.
- Las filas de `"relative"` suman 1.

## Comando exacto de verificación

```bash
python -m pytest python/tests/test_sklearn.py -q
```

Salida esperada: todos en verde, incluidos los 6 nuevos.

## Qué NO tocar

- No toques `workflow.py`: ya binariza y su comportamiento no cambia.
- No añadas `representation` a `cohort_presence_matrix` (gwas.py). Esa
  función tiene su propio contrato documentado — almacena profundidad y los
  consumidores binarizan — y cambiarlo es otra tarea.

## Riesgos

- **Cambio rompedor del default.** Cualquier resultado publicado con la
  versión anterior usaba conteos. Anótalo en `CHANGELOG.md` bajo
  `Changed (breaking)` con la migración: `representation="count"` recupera
  el comportamiento anterior.
- Los tests existentes de `test_sklearn.py` que comprueben valores concretos
  de la matriz fallarán. Actualízalos a `representation="count"` explícito si
  lo que prueban es la aritmética de conteo, o al nuevo default si lo que
  prueban es la forma.
- El umbral de 3× para el aviso es una elección, no una constante física.
  Documentado como tal en el mensaje.
