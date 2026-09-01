# Tarea 14 — `fastdna.audit()`, núcleo del reporte

**Objetivo:** una llamada que mida la brecha entre CV aleatorio y CV
bloqueado por linaje sobre una cohorte, y la reporte.

**Hallazgos que resuelve:** ninguno (feature nueva; es el foso del proyecto).
**Prerequisitos:** tarea 12 (publicado), y `python/fastdna/cv.py` ya existente.
**Archivos exactos:** `python/fastdna/audit.py` (nuevo),
`python/fastdna/__init__.py`, `python/tests/test_audit.py` (nuevo).
**Esfuerzo:** 3 días (excede 1 h a propósito: es la tarea grande de la fase 1).

## Contexto

Las poblaciones bacterianas son clonales. Un split aleatorio de CV reparte
casi-copias del mismo clon entre train y test, y el modelo puede puntuar
alto reconociendo el linaje en vez del fenotipo. `python/fastdna/cv.py` ya
documenta la literatura y ya tiene las piezas: `lineage_groups()`,
`LineageKFold`, `permutation_importance_pvalues()`.

Lo que falta es el producto: una sola llamada que las orqueste y produzca
un reporte pegable en un paper.

**Alcance de ESTA tarea:** los dos scores, la estructura de linajes y las
dos líneas base. La confusión fenotipo↔linaje es la tarea 15, la
atribución de features la 16 y la procedencia la 17. Esta tarea deja
`Confounding` y `FeatureAttribution` como `None` en el dataclass.

## Firma

```python
def audit(
    samples,
    phenotype,
    *,
    estimator=None,
    k=31,
    min_count=None,
    top_features=100_000,
    sketch_size=1000,
    lineage_threshold=None,
    n_splits=5,
    scoring=None,
    random_state=0,
    n_jobs=None,
) -> AuditReport:
```

## Decisiones ya tomadas (no las reabras)

- **`estimator=None` usa `LogisticRegression(penalty="l2", max_iter=1000)`.**
  Lineal a propósito: es interpretable, no sobreajusta 2 M de features tan
  agresivamente como un bosque, y hace legible la atribución de la tarea 16.
- **`scoring=None` detecta:** `roc_auc` si el fenotipo es binario, `r2` si
  es continuo. Detectado, no adivinado.
- **`min_count=None` es por muestra**, vía `suggest_min_count()` sobre su
  propio espectro — no global. La profundidad varía entre muestras y un
  umbral global descarta k-mers reales de las poco cubiertas.
- **`lineage_threshold=None` se elige por el codo** de la curva de alturas
  de fusión del dendrograma (máxima distancia a la recta que une el primer
  y el último punto). Determinista, y **el valor elegido se reporta
  siempre**.
- **`AuditReport` es `frozen=True`.** Un reporte es evidencia; mutarlo
  después de generarlo es justo lo que no debe poder hacerse.
- **Los dos scores se calculan sobre la misma matriz y la misma semilla.**
  La única variable que cambia es el splitter — si cambiara algo más, la
  brecha no sería atribuible a la fuga.

## Esqueleto a implementar

```python
\"\"\"fastdna.audit -- mide cuanta de tu senal es linaje y no fenotipo.

No es un modelo, es un diagnostico. La pregunta que responde es la que un
revisor no puede formular hoy: "de tu AUC de 0.94, cuanto queda cuando el
modelo no puede ver casi-copias de las muestras de entrenamiento".

Reporta DOS scores a proposito. Un bioRxiv de 2026 sobre pipelines
robustos de prediccion de AMR encontro que el CV aleatorio a veces predice
el rendimiento clinico mejor que los splits filogeneticos: el campo no lo
tiene resuelto. Este modulo no impone una respuesta universal, mide la
brecha en esta cohorte concreta y deja el juicio a quien la conoce.

`scipy` y `scikit-learn` se importan de forma perezosa dentro de las
funciones que los usan, siguiendo el patron que ya usan `fastdna.cv` y
`fastdna.embed`: importar este modulo nunca exige ninguno de los dos.
\"\"\"
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = ["audit", "AuditReport", "FoldScores", "LineageStructure", "Verdict"]


class Verdict(enum.Enum):
    CLEAN = "clean"
    INFLATED = "inflated"
    CONFOUNDED = "confounded"
    UNDERPOWERED = "underpowered"


@dataclass(frozen=True)
class FoldScores:
    per_fold: tuple[float, ...]

    @property
    def mean(self) -> float:
        return float(np.mean(self.per_fold))

    @property
    def std(self) -> float:
        return float(np.std(self.per_fold, ddof=1)) if len(self.per_fold) > 1 else 0.0

    @property
    def ci95(self) -> tuple[float, float]:
        \"\"\"Intervalo normal al 95 %. Con 5 folds es ancho, y eso es
        informacion: un intervalo que cruza el clasificador trivial
        significa que no hay resultado.\"\"\"
        half = 1.96 * self.std / max(len(self.per_fold) ** 0.5, 1.0)
        return (self.mean - half, self.mean + half)


@dataclass(frozen=True)
class LineageStructure:
    labels: np.ndarray
    n: int
    sizes: tuple[int, ...]
    threshold: float
    singletons: int

    @property
    def largest_fraction(self) -> float:
        return max(self.sizes) / sum(self.sizes) if self.sizes else 0.0


@dataclass(frozen=True)
class AuditReport:
    score_random: FoldScores
    score_lineage_blocked: FoldScores
    score_majority_class: float
    score_lineage_only: float
    n_samples: int
    n_features: int
    lineages: LineageStructure
    scoring_name: str
    verdict: Verdict

    @property
    def inflation(self) -> float:
        return self.score_random.mean - self.score_lineage_blocked.mean

    def to_markdown(self) -> str: ...
    def __repr__(self) -> str: ...
    def _repr_html_(self) -> str: ...
```

## Pasos

1. Crear `python/fastdna/audit.py` con el esqueleto de arriba completo.
2. Implementar `_resolve_samples()`: acepta directorio (usa
   `cohort.discover_samples` vía el FFI), `Mapping`, `Sequence` o
   `pyarrow.Table` ya vectorizada.
3. Implementar `_align_phenotype()`: si `phenotype` es `Mapping`/`Series`, se
   alinea **por sample_id**, no por posición. Si es `Sequence`, se exige el
   mismo orden y se valida la longitud con un mensaje que nombre ambas.
4. Implementar `_build_features()` usando `sklearn.KmerVectorizer`.
5. Implementar `_lineages()` usando `compare_all(metric="mash_distance")` y
   `cv.lineage_groups()`, con la selección por codo si el umbral es `None`.
6. Calcular los cuatro scores.
7. Implementar el veredicto según la tabla de abajo.
8. Implementar `__repr__` con el formato de reporte.
9. Reexportar `audit` y `AuditReport` en `__init__.py` y añadirlos a
   `__all__`.

## La tabla del veredicto

| Veredicto | Condición |
|---|---|
| `UNDERPOWERED` | < 3 linajes con ambas clases, o < 20 muestras |
| `CONFOUNDED` | confusión ≥ 0,6 (tarea 15; hasta entonces, nunca) |
| `INFLATED` | inflación ≥ 0,05 |
| `CLEAN` | el resto |

Se evalúa en ese orden.

## Tests completos

Archivo nuevo `python/tests/test_audit.py`:

```python
\"\"\"Los tres controles que hacen creible el audit.

Un modulo que mide fuga tiene que demostrar que la detecta cuando la hay
(control positivo) y que no la inventa cuando no la hay (control
negativo). Sin esos dos, el numero que reporta no significa nada.
\"\"\"
from __future__ import annotations

import numpy as np
import pytest

import fastdna

sklearn = pytest.importorskip("sklearn")
pytest.importorskip("scipy")


def _write_lineage_cohort(tmp_path, n_lineages=4, per_lineage=8, read_len=120):
    \"\"\"Cohorte sintetica con estructura de linaje real: cada linaje es una
    secuencia base distinta, y sus miembros son mutaciones puntuales de
    ella. Eso reproduce lo que hace clonal a una poblacion bacteriana --
    los miembros de un linaje comparten casi todos sus k-mers.\"\"\"
    rng = np.random.default_rng(0)
    bases = np.array(list("ACGT"))
    paths, lineage_of = [], []

    for lineage in range(n_lineages):
        root = "".join(rng.choice(bases, size=read_len * 4))
        for member in range(per_lineage):
            seq = list(root)
            for pos in rng.choice(len(seq), size=max(1, len(seq) // 200), replace=False):
                seq[pos] = str(rng.choice(bases))
            seq = "".join(seq)
            reads = [seq[i : i + read_len] for i in range(0, len(seq) - read_len, read_len // 2)]
            path = tmp_path / f"L{lineage}_S{member}.fastq"
            path.write_text(
                "".join(f"@r{i}\n{r}\n+\n{'I' * len(r)}\n" for i, r in enumerate(reads))
            )
            paths.append(str(path))
            lineage_of.append(lineage)

    return paths, np.array(lineage_of)


def test_positive_control_a_phenotype_that_is_lineage_is_caught(tmp_path):
    \"\"\"CONTROL POSITIVO. El fenotipo se asigna POR LINAJE, asi que no hay
    ninguna senal biologica que aprender: todo lo que un modelo pueda
    puntuar es linaje. Si el audit no lo detecta, el modulo no sirve.\"\"\"
    paths, lineage = _write_lineage_cohort(tmp_path)
    phenotype = (lineage < 2).astype(int)      # el fenotipo ES el linaje

    report = fastdna.audit(paths, phenotype, k=21, n_splits=3, random_state=0)

    assert report.score_random.mean > 0.8, (
        "el CV aleatorio deberia puntuar alto reconociendo el linaje; "
        f"dio {report.score_random.mean:.3f}"
    )
    assert report.inflation > 0.15, (
        f"la inflacion deberia ser grande y fue {report.inflation:.3f}"
    )
    assert report.score_lineage_only > 0.8, (
        "usar solo la etiqueta de linaje deberia bastar para este fenotipo"
    )


def test_negative_control_a_shuffled_phenotype_scores_at_chance(tmp_path):
    \"\"\"CONTROL NEGATIVO. Fenotipo barajado: no hay nada que aprender, ni
    biologico ni de linaje. Ambos scores deben caer al azar. Si el audit
    reporta inflacion aqui, esta inventando estructura.\"\"\"
    paths, _ = _write_lineage_cohort(tmp_path)
    rng = np.random.default_rng(1)
    phenotype = rng.integers(0, 2, size=len(paths))

    report = fastdna.audit(paths, phenotype, k=21, n_splits=3, random_state=0)

    assert report.score_lineage_blocked.mean < 0.75, (
        "un fenotipo aleatorio no puede predecirse; "
        f"dio {report.score_lineage_blocked.mean:.3f}"
    )


def test_lineage_structure_recovers_the_planted_clusters(tmp_path):
    \"\"\"Los linajes que detecta tienen que ser los que se plantaron. Sin
    esto, todo lo demas del reporte podria estar bien por casualidad.\"\"\"
    paths, lineage = _write_lineage_cohort(tmp_path, n_lineages=4, per_lineage=8)
    phenotype = np.zeros(len(paths), dtype=int)
    phenotype[::2] = 1

    report = fastdna.audit(paths, phenotype, k=21, n_splits=3, random_state=0)

    from sklearn.metrics import adjusted_rand_score

    ari = adjusted_rand_score(lineage, report.lineages.labels)
    assert ari > 0.7, f"los clusters no recuperan los linajes plantados (ARI={ari:.2f})"
    assert report.lineages.threshold > 0, "el umbral elegido debe reportarse"


def test_phenotype_given_as_a_mapping_aligns_by_sample_id(tmp_path):
    \"\"\"Un dict se alinea por sample_id, no por posicion. Alinear por
    posicion en silencio seria el peor bug posible en este modulo: daria
    un numero plausible sobre etiquetas equivocadas.\"\"\"
    paths, lineage = _write_lineage_cohort(tmp_path, n_lineages=2, per_lineage=6)
    import pathlib

    by_id = {pathlib.Path(p).stem: int(l < 1) for p, l in zip(paths, lineage)}

    shuffled = list(reversed(paths))
    report = fastdna.audit(shuffled, by_id, k=21, n_splits=2, random_state=0)
    assert report.n_samples == len(paths)


def test_a_phenotype_of_the_wrong_length_names_both_numbers(tmp_path):
    paths, _ = _write_lineage_cohort(tmp_path, n_lineages=2, per_lineage=4)
    with pytest.raises(ValueError) as excinfo:
        fastdna.audit(paths, [0, 1, 0], k=21)
    message = str(excinfo.value)
    assert str(len(paths)) in message and "3" in message
```

## Criterios de aceptación

- Los cinco tests pasan.
- `print(report)` produce el reporte de texto con las secciones
  RENDIMIENTO y ESTRUCTURA POBLACIONAL.
- `fastdna.audit` está en `fastdna.__all__`.

## Comando exacto de verificación

```bash
python -m pytest python/tests/test_audit.py -q
```

Salida esperada: `5 passed`.

## Qué NO tocar

- No modifiques `python/fastdna/cv.py`. Esta tarea lo **usa**, no lo cambia.
- No implementes la confusión fenotipo↔linaje ni la atribución de features
  aquí. Son las tareas 15 y 16.
- No implementes inferencia estadística (modelos mixtos, heredabilidad).
  `pyseer` y `kmersGWAS` existen y están validados.

## Riesgos

- Los controles positivo y negativo son la parte que hace creíble el
  módulo. **Si el control positivo no detecta la fuga plantada, para y
  repórtalo** — significa que la lógica está mal, no que el umbral del test
  haya que aflojarlo.
- El test genera cohortes sintéticas; son lentos. Si tardan más de 60 s,
  baja `per_lineage`, no `n_lineages` (necesitas ≥ 3 linajes para que
  `LineageKFold` tenga folds).
