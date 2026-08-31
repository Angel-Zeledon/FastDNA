# `fastdna.audit()` — diseño de API completo

Fase 1 de [`PLAN.md`](PLAN.md). Este documento fija la API, qué mide cada
número y cómo se calcula. Es el foso del proyecto: no es un modelo, es un
**diagnóstico**, y los diagnósticos se citan porque los revisores los piden.

---

## 1. Qué problema resuelve, exactamente

Un investigador entrena un clasificador de resistencia antimicrobiana sobre
una cohorte bacteriana y reporta AUC 0,94. Ese número casi siempre está
inflado: las poblaciones bacterianas son clonales, un split aleatorio de
CV reparte casi-copias del mismo clon entre train y test, y el modelo puede
puntuar alto reconociendo el linaje en vez del fenotipo.

Hoy, un revisor que sospecha eso **no tiene qué pedir**. Después de esto,
pide: *"corre `fastdna.audit()` y enseña la tabla."*

`python/fastdna/cv.py` ya tiene las piezas (`lineage_groups`,
`LineageKFold`, `permutation_importance_pvalues`). Lo que falta es el
producto: una sola llamada que las orqueste, mida la brecha y la reporte de
forma que se pueda pegar en un paper.

---

## 2. La API

### 2.1 Entrada

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
    n_permutations=1000,
    random_state=0,
    n_jobs=None,
    cache=None,
    progress=None,
) -> AuditReport:
```

| Parámetro | Tipo | Decisión de diseño |
|---|---|---|
| `samples` | `str \| PathLike \| Mapping[str, PathLike] \| Sequence[PathLike] \| pyarrow.Table` | Un directorio se descubre con `cohort.discover_samples` (pareo R1/R2 ya implementado). Un `Mapping` da IDs explícitos. Una `Table` ya vectorizada salta el conteo — permite auditar features que no vienen de FastDNA. |
| `phenotype` | `Sequence \| Mapping[str, Any] \| pandas.Series` | Si es `Mapping`/`Series` se alinea **por sample_id**, no por posición. Un `Sequence` exige el mismo orden que `samples` y se valida la longitud. |
| `estimator` | estimador sklearn o `None` | `None` usa `LogisticRegression(penalty="l2", max_iter=1000)`. **Decisión:** el default es lineal a propósito — es interpretable, no sobreajusta 2 M de features tan agresivamente como un bosque, y hace que la atribución a linaje (§3.4) sea legible. |
| `k` | `int` | 31 por defecto, igual que `count()`. |
| `min_count` | `int \| None` | `None` = por muestra, vía `suggest_min_count()` sobre su propio espectro. **Decisión:** por muestra y no global, porque la profundidad varía entre muestras y un umbral global descarta k-mers reales de las poco cubiertas. |
| `top_features` | `int \| None` | Cota de memoria. `None` = sin límite, y entonces se aplica el límite de bytes de `MatrixTooLarge`. |
| `lineage_threshold` | `float \| None` | Distancia Mash que define "mismo linaje". `None` = se elige por el método del codo sobre la distribución de distancias (§3.2), y el valor elegido se reporta. |
| `scoring` | `str \| callable \| None` | `None` = `roc_auc` si el fenotipo es binario, `r2` si es continuo. Detectado, no adivinado. |
| `n_permutations` | `int` | Para los p-valores de importancia. 0 los desactiva (es la parte cara). |
| `cache` | `str \| PathLike \| None` | Directorio donde persistir sketches y matriz de conteos. **Decisión:** sin caché por defecto; auditar dos veces la misma cohorte no debe recontar 400 FASTQ en silencio, pero escribir en disco sin que lo pidan tampoco. |

### 2.2 Uso mínimo

```python
import fastdna

report = fastdna.audit("cohort/", phenotype=resistencia)
print(report)
```

### 2.3 La salida

```
FastDNA leakage audit — 412 muestras, k=31, 2 138 447 features

  RENDIMIENTO
    AUC, CV aleatorio 5-fold        0.94  [0.91, 0.96]
    AUC, CV bloqueado por linaje    0.71  [0.64, 0.78]
    Inflación                      +0.23   ALTA
    AUC, clasificador trivial       0.58          <- solo la clase mayoritaria
    AUC, solo-linaje                0.89          <- linaje como unica feature

  ESTRUCTURA POBLACIONAL
    linajes detectados                14   (umbral Mash d < 0.0100, por codo)
    linaje mayor                     38%   de la cohorte
    linajes con una sola muestra       3   (excluidos de los folds)
    confusion fenotipo/linaje       0.81   ALTA  (V de Cramer, p < 0.001)

  ATRIBUCION DE FEATURES
    de los 20 k-mers mas importantes:
      17 estan restringidos a <= 2 linajes
       3 son invariantes de linaje   <- los unicos candidatos creibles
    señal atribuible a linaje         62%

  VEREDICTO
    Este diseño no puede separar fenotipo de linaje. El AUC de 0.94
    obtenido con CV aleatorio no es un estimador de rendimiento sobre
    aislados nuevos. Reporta 0.71, o amplía la cohorte con aislados
    resistentes de linajes hoy no representados.

  PROCEDENCIA
    fastdna 0.2.0 · k=31 · min_count=auto · seed=42
    cohorte sha256 a3f1c8...  ·  2026-08-26T14:22:03Z
```

### 2.4 El objeto de retorno

```python
@dataclass(frozen=True)
class AuditReport:
    # Rendimiento
    score_random: FoldScores          # .mean, .std, .ci95, .per_fold
    score_lineage_blocked: FoldScores
    score_majority_class: float
    score_lineage_only: float
    inflation: float                  # random - lineage_blocked

    # Estructura
    n_samples: int
    n_features: int
    lineages: LineageStructure        # .labels, .n, .sizes, .threshold, .singletons
    confounding: Confounding          # .cramers_v, .p_value, .verdict

    # Features
    feature_attribution: FeatureAttribution   # .top, .lineage_restricted, .lineage_invariant
    signal_attributable_to_lineage: float

    # Meta
    verdict: Verdict                  # enum: CLEAN | INFLATED | CONFOUNDED | UNDERPOWERED
    provenance: Provenance

    def to_table(self) -> "pyarrow.Table": ...
    def to_markdown(self) -> str: ...      # para pegar en un paper
    def to_html(self) -> str: ...
    def _repr_html_(self) -> str: ...      # Jupyter
    def plot(self, ax=None): ...           # requiere matplotlib
```

**Decisión:** `frozen=True`. Un reporte es evidencia; mutarlo después de
generarlo es exactamente lo que no debe poder hacerse.

---

## 3. Qué mide cada número, y cómo

### 3.1 Los dos scores

- **CV aleatorio:** `sklearn.model_selection.cross_val_score` con
  `StratifiedKFold(n_splits, shuffle=True, random_state=...)`.
- **CV bloqueado por linaje:** el mismo estimador con `cv.LineageKFold`.

Ambos sobre **la misma matriz de features y la misma semilla**. La única
variable que cambia es el splitter — si cambiara algo más, la brecha no
sería atribuible a la fuga.

> **Se reportan los dos a propósito.** Un [bioRxiv de 2026](https://www.biorxiv.org/content/10.64898/2026.06.28.734076v1)
> encontró que el CV aleatorio a veces predice el rendimiento clínico mejor
> que los splits filogenéticos. El campo no lo tiene resuelto. `audit()` no
> impone una respuesta: mide la brecha en *esta* cohorte y deja el juicio a
> quien la conoce.

### 3.2 Los linajes

`compare_all(metric="mash_distance")` → matriz de distancias →
`scipy.cluster.hierarchy` con enlace `average` → corte en
`lineage_threshold`.

Si `lineage_threshold is None`, se elige por **el codo de la curva de
distancias de fusión**: se ordenan las alturas de fusión del dendrograma, se
toma el punto de máxima curvatura (máxima distancia a la recta que une el
primer y el último punto). Es determinista, no necesita que el usuario sepa
qué es una distancia Mash, y **el valor elegido se reporta siempre** para
que sea auditable y sobreescribible.

Los linajes de una sola muestra se excluyen de los folds y se cuentan
aparte: no pueden aparecer en train y test a la vez, así que no aportan
información sobre generalización entre linajes.

### 3.3 Confusión fenotipo ↔ linaje

**La pregunta:** ¿está el fenotipo repartido entre linajes, o cada linaje es
casi puro en su fenotipo?

**Cómo se calcula:** tabla de contingencia `linaje × fenotipo` → **V de
Cramer**, con su p-valor por prueba exacta de permutación (`n_permutations`
barajadas del fenotipo dentro de la cohorte).

**Por qué V de Cramer y no información mutua:** está normalizada a `[0,1]`
independientemente del número de linajes, así que un valor es comparable
entre cohortes de 5 y de 50 linajes. La información mutua cruda no lo está,
y compararla entre estudios llevaría a error.

Umbrales reportados como texto:

| V de Cramer | Etiqueta | Significado |
|---|---|---|
| < 0,3 | BAJA | El fenotipo cruza linajes; el diseño puede separar |
| 0,3–0,6 | MODERADA | Separable con cuidado |
| > 0,6 | ALTA | El diseño probablemente no puede separar |

Para fenotipo continuo, se usa **eta cuadrado** (η²) del ANOVA de una vía
del fenotipo sobre el linaje, que ocupa el mismo rango `[0,1]` y admite los
mismos umbrales.

### 3.4 Atribución de features a linaje

Para cada uno de los `n` k-mers más importantes del modelo (por coeficiente
absoluto en un modelo lineal, por importancia por permutación si no):

1. **Restringido a linaje:** ¿en cuántos linajes distintos aparece? Un k-mer
   presente en ≤ 2 linajes de 14 es un marcador de linaje, no de fenotipo.
2. **Invariante de linaje:** ¿aparece en múltiples linajes *y* su presencia
   correlaciona con el fenotipo **dentro** de cada uno? Estos son los únicos
   candidatos biológicamente creíbles, y son los que hay que enseñar.

**`signal_attributable_to_lineage`** = la caída relativa de score al
reentrenar usando **solo** las features restringidas a linaje, dividida por
el score completo. Es una atribución operativa, no una descomposición de
varianza — y se documenta así, sin pretender más.

### 3.5 Las dos líneas base

- **Clasificador trivial:** `DummyClassifier(strategy="prior")`. Si tu AUC
  de 0,94 baja a 0,71 pero el trivial da 0,68, no tienes casi nada.
- **Solo-linaje:** entrenar usando **únicamente la etiqueta de linaje** como
  feature (one-hot). Si eso da 0,89, el modelo no necesitaba los k-mers.

Esta segunda es la más contundente del reporte y no la calcula ninguna otra
herramienta.

### 3.6 El veredicto

| Veredicto | Condición |
|---|---|
| `CLEAN` | inflación < 0,05 **y** confusión < 0,3 |
| `INFLATED` | inflación ≥ 0,05, confusión < 0,6 |
| `CONFOUNDED` | confusión ≥ 0,6 |
| `UNDERPOWERED` | < 3 linajes con ambas clases, o < 20 muestras |

**Decisión:** el veredicto es una regla determinista y publicada, no un
juicio. Un usuario puede discrepar y leer los números.

---

## 4. Contra qué se valida

**Decisión: no se acepta este módulo sin una validación de extremo a extremo
sobre datos reales públicos.** Tarea 19 del plan.

1. Cohorte pública de AMR con linajes conocidos (p. ej. las colecciones de
   *M. tuberculosis* con linajes 1–7 asignados).
2. **Control positivo:** un fenotipo *sintético* asignado por linaje. El
   audit debe dar `CONFOUNDED`, inflación alta y ~100 % de señal atribuible
   a linaje. Si no lo detecta, el módulo no sirve.
3. **Control negativo:** fenotipo barajado al azar. Ambos scores deben caer
   al nivel del clasificador trivial y la confusión debe ser baja.
4. **Caso real:** un fenotipo de resistencia auténtico, donde se espera algo
   intermedio.

Los tres se ejecutan en CI con una submuestra pequeña, y completos en el
paper.

---

## 5. Superficie en CLI

```bash
fastdna audit --input cohort/ --phenotype pheno.csv --output audit.html
```

Escribe el HTML, imprime el resumen y **sale con código 1 si el veredicto
es `CONFOUNDED`** — para que un pipeline pueda fallar en vez de publicar
un número inflado.

---

## 6. Lo que `audit()` NO hace

Se declara explícitamente, igual que hace `cv.py` con sus límites:

- **No arregla nada.** Da un estimador honesto, no uno mejor. Los scores
  bajan; esa caída es el hallazgo, no una regresión.
- **No hace inferencia estadística.** Sin modelos mixtos, sin heredabilidad.
  `pyseer` y `kmersGWAS` existen y están validados; rehacerlos aquí sería
  una versión peor de software en el que la gente ya confía.
- **No sustituye a una filogenia.** Las distancias Mash son un sustituto
  a la resolución "¿son estos dos aislados casi clonales?", que es la que
  este problema necesita. Para tiempos de divergencia hace falta un árbol
  de verdad.
- **No rescata un diseño confundido.** Si todos los resistentes están en un
  clon, ningún split separa las dos señales. `audit()` lo dice en vez de
  fingir.

---

## 7. Por qué esto es defendible

| Competidor | Por qué no cierra este hueco |
|---|---|
| `trustcv` | Necesita los grupos ya calculados; no toca secuencias |
| `bioLeak` | R, y parte de una matriz de features |
| `pyseer` | Asociación, no predicción; las features las traes tú |
| KMC3 / FastK / Jellyfish | Son contadores; nunca tendrán ML |
| `sourmash` | Búsqueda, no predicción |
| biotite | Toolkit general; no es un pipeline opinionado |

La pieza que nadie más tiene es **el camino completo desde FASTQ**: para
calcular los linajes hace falta comparar todos los pares, y para eso hace
falta sketchear cada muestra, y para eso hace falta leer los FASTQ. FastDNA
ya hace las tres cosas en un solo proceso, en segundos.
