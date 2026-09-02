# Ser #1 en ML genómico — los huecos que quedan

Continuación de [`PLAN.md`](PLAN.md). Este documento sale de leer la capa ML
real (`sklearn.py`, `cv.py`, `gwas.py`, `workflow.py`, `equivalence.py`) y de
contrastarla con lo que el campo declara como problema abierto.

---

## 0. El hallazgo que reordena la estrategia

El paper que `python/fastdna/cv.py` ya cita —
[James, Williamson, Tino & Wheeler, *Whole-Genome Phenotype Prediction with
Machine Learning: Open Problems in Bacterial Genomics*, arXiv 2502.07749](https://arxiv.org/abs/2502.07749)
— dice esto en su resumen:

> *"Los primeros esfuerzos que confiaron a modelos de machine learning la
> tarea de predecir fenotipo a partir de genotipo devuelven puntuaciones de
> alta precisión. Sin embargo, **los intentos de extraer cualquier
> significado de esos modelos predictivos resultan corrompidos por features
> falsamente identificadas como 'causales'**. Confiar solo en
> reconocimiento de patrones y correlaciones no es fiable, y lo es
> especialmente poco en genómica bacteriana, donde la alta dimensionalidad
> y las asociaciones espurias son la norma."*

**El problema #1 del campo no es predecir mejor. Es que las features que el
modelo señala como causales son mentira.**

Eso reencuadra el producto. `audit()` (fase 1 del plan) ataca una mitad — la
inflación del score por estructura poblacional. La otra mitad —¿en qué de lo
que el modelo dice puedo creer?— es un territorio más grande, y **ya tienes
la maquinaria y no la estás usando para eso**.

---

## G-6 · La validación cruzada recuenta cada FASTQ en cada fold

**Severidad: crítica. Está justo en el camino que quieres poseer.**

> **Estado: cerrado.** `fastdna.count_cohort()` / `CohortCounts` eliminan el
> recuento entre folds (medido en
> `python/tests/test_cohort_counts.py::test_cross_validation_with_the_artifact_never_recounts`:
> **cero** llamadas a `fastdna.count()` en una CV de 3 folds), y
> `CohortCounts.save()`/`.load()` lo extienden entre procesos, que era la
> mitad que faltaba: un segundo pase del mismo estudio —otro
> `--top-features`, un arreglo en `audit()`— ya no vuelve a contar.
> `scripts/validation/leakage_survey.py` lo usa vía `--cache-dir`.

### Qué pasa hoy

`python/fastdna/sklearn.py:365-390`:

```python
    def transform(self, X):
        check_is_fitted(self, "vocabulary_")
        paths = _validate_paths(X, "transform")
        return self._project(self._count_cohort(paths), len(paths))
```

y `_count_cohort` (`:189-213`) hace `fastdna.count(path, ...)` para cada
ruta, sin caché ni memoización de ningún tipo.

### La aritmética

`cross_val_score(pipeline, paths, y, cv=5)` hace, por fold:

| Llamada | Qué cuenta |
|---|---|
| `Pipeline.fit(X_train)` → `vectorizer.fit_transform` | 80 % de las muestras |
| `Pipeline.score(X_test)` → `vectorizer.transform` | 20 % de las muestras |

= **100 % de la cohorte, una vez por fold**. Con 5 folds, **5 pasadas
completas** sobre cada FASTQ.

Y `audit()` (tarea 14) hace CV aleatorio *más* CV bloqueado por linaje, más
el sketching de `compare_all`, más la línea base solo-linaje:
**~11 pasadas completas**.

Sobre una cohorte de 412 muestras de ~1 GB: **~4,5 TB de E/S y
descompresión para auditar un modelo.**

### Por qué duele especialmente aquí

Ya encontraste y arreglaste la versión ×2 de este mismo bug. El docstring
de `fit_transform` (`:342-364`) lo dice:

> *"el `TransformerMixin.fit_transform` por defecto es
> `self.fit(X, y).transform(X)`, que cuenta cada FASTQ de entrenamiento dos
> veces… una cohorte de 20 muestras hacía 20 pasadas en vez de 40."*

Arreglaste la repetición **dentro** de un fit. La repetición **entre folds**
es 5–11× y sigue ahí.

Toda la ventaja de tener un motor Rust rápido se evapora si el camino de ML
lo invoca once veces sobre los mismos bytes.

### El arreglo

Un artefacto de cohorte contado una vez y reusado por todos los folds.

```python
# Nuevo: fastdna.CohortCounts
counts = fastdna.count_cohort("cohort/", k=31)   # UNA pasada, en paralelo
counts.save("cohort.fdna")                        # opcional, persistente

# El vectorizador acepta el artefacto en vez de rutas
vec = KmerVectorizer(top_features=10_000)
scores = cross_val_score(make_pipeline(vec, LogisticRegression()),
                         counts, y, cv=5)          # cero recuentos
```

**Decisión de diseño:** `KmerVectorizer` acepta **o** rutas **o** un
`CohortCounts`. Las rutas siguen funcionando —no se rompe nada— pero la
documentación empuja al artefacto, y `audit()` lo exige.

La garantía anti-fuga se mantiene intacta y es importante decir por qué:
la fuga que `KmerVectorizer` previene es que **el vocabulario** se decida
viendo el fold de test. Contar es una operación por muestra que no mira
etiquetas ni otras muestras; contar todo antes no filtra nada. Lo que no
puede moverse fuera del fold es `_learn_vocabulary`, y no se mueve.

**Ganancia: 5× en CV normal, ~11× en `audit()`.** Es la mayor ganancia de
rendimiento del proyecto entero, más grande que cualquier cosa de
[`performance-v2.md`](performance-v2.md), y no toca ni una línea de Rust.

---

## G-7 · `KmerVectorizer` devuelve conteos crudos: el modelo aprende profundidad

**Severidad: alta. Es un confusor técnico de la misma clase que el linaje.**

### Qué pasa hoy

`python/fastdna/sklearn.py:301-311`:

```python
        data = np.asarray(counts.frequencies)[kept].astype(np.float64)
        return sparse.csr_matrix(
            (data, (rows, column[kept].astype(np.int64))),
            shape=(n_samples, len(self.vocabulary_)),
            dtype=np.float64,
        )
```

Frecuencias crudas. No hay normalización por profundidad en ninguna parte de
`sklearn.py` ni de `cv.py` (verificado por grep sobre
`normaliz|depth|cpm|tpm|rarefac|relative.abundance`).

### El problema

Una muestra secuenciada a 100× tiene ~5× los conteos de k-mer de una
secuenciada a 20×, **por razones puramente técnicas**. Un
`LogisticRegression` sobre conteos crudos aprende profundidad de
secuenciación, no biología.

Y en la práctica la profundidad correlaciona con el lote, el año y el
centro de secuenciación — que a su vez correlacionan con el fenotipo en
cualquier cohorte clínica real. Es exactamente la estructura del problema
que `cv.py` existe para atacar, en otro eje.

### La inconsistencia interna

Los dos caminos del paquete tratan esto de forma distinta:

| Camino | Qué hace | Dónde |
|---|---|---|
| GWAS (`workflow.py`) | **Binariza** a presencia/ausencia | `workflow.py:419-432` |
| sklearn (`KmerVectorizer`) | Conteos crudos, sin normalizar | `sklearn.py:301` |

`workflow.py:423` incluso lo comenta: *"cohort_presence_matrix() stores raw
per-sample depth, not presence… setting every stored value to 1 recovers
plain 0/1 presence"*. El camino de asociación sabe que hay que hacerlo. El
de predicción no lo hace.

### El arreglo

```python
KmerVectorizer(
    k=31,
    top_features=10_000,
    representation="presence",   # NUEVO. presence | count | clr | relative
)
```

**Decisión: el default pasa a `"presence"`.** Razones:

1. Es lo que hace el camino GWAS del propio paquete, y lo que hacen pyseer,
   kmersGWAS y DBGWAS.
2. Es inmune a la profundidad por construcción.
3. Para la mayoría de las preguntas genómicas —¿tiene este aislado este
   gen?— presencia *es* la pregunta.

`"count"` sigue disponible para quien mida abundancia (metagenómica), pero
**emitiendo un `DepthConfoundingWarning`** si la profundidad varía más de
3× entre muestras de la cohorte, con la cifra medida en el mensaje.

`"clr"` (log-ratio centrado) es la opción correcta para datos
composicionales de metagenoma, y hoy no existe.

Es un cambio rompedor y por eso va **antes** del primer release.

---

## G-8 · `fastdna.explain()` — atacar el problema abierto del campo

**Esta es la propuesta grande, y es la que te haría el estándar.**

### Qué pregunta responde

No *"¿cuál es mi AUC?"* — eso es `audit()`. Sino:

> **"Este k-mer que mi modelo dice que es la variante causal, ¿lo es?"**

Que es literalmente el problema abierto que nombra arXiv 2502.07749.

### La pieza que ya tienes y no usas para esto

`python/fastdna/equivalence.py` colapsa k-mers con patrón de
presencia/ausencia idéntico en la cohorte. Hoy se usa **solo para
comprimir** el espacio de features en `workflow.py`.

Pero el tamaño de la clase de equivalencia **es una medida directa de
no-identificabilidad**:

> Si 847 k-mers tienen exactamente el mismo patrón de presencia en tu
> cohorte, ninguno de los 847 es más causal que los otros 846. Reportar uno
> como "la variante causal" es precisamente la falsa causalidad que el paper
> describe. **El número 847 es el resultado honesto.**

Nadie reporta eso. Y tú puedes calcularlo hoy.

### La API

```python
explanation = fastdna.explain(report, top_n=20)
print(explanation)
```

```
FastDNA feature explanation — 20 k-mers principales

  #1  ACGTTGCAAACCGGTTGATTACAGATTACAG      coef +2.31
      ┌ identificabilidad
      │  clase de equivalencia          847 k-mers   NO IDENTIFICABLE
      │  se extiende sobre              ~26 kb del genoma
      │  → esto no es una variante, es un bloque ligado
      ├ atribución
      │  linajes en los que aparece      2 de 14     RESTRINGIDO A LINAJE
      │  asociación dentro de linaje     p = 0.71    SIN EVIDENCIA
      │  → indistinguible de un marcador de linaje
      └ anotación
         solapa                          gyrA (98% id)
      VEREDICTO: marcador de linaje. No reportar como causal.

  #7  TTGCAAACCGGTTGATTACAGATTACAGATT      coef +1.04
      ┌ identificabilidad
      │  clase de equivalencia            3 k-mers   IDENTIFICABLE
      ├ atribución
      │  linajes en los que aparece      11 de 14
      │  asociación dentro de linaje     p = 0.003   SOBREVIVE
      │  → presente en 11 linajes y asociado DENTRO de cada uno
      └ anotación
         solapa                          blaTEM-1 (100% id)
      VEREDICTO: candidato creíble.

  Resumen: 3 de 20 sobreviven. 17 son marcadores de linaje o bloques
  no identificables.
```

### Las tres pruebas, y qué mide cada una

| Prueba | Cómo se calcula | Qué descarta |
|---|---|---|
| **Identificabilidad** | Tamaño de la clase de equivalencia (`equivalence.py`) + extensión genómica si hay anotación | "Esta es *la* variante" cuando son 847 indistinguibles |
| **Atribución a linaje** | ¿En cuántos linajes aparece? Y asociación **dentro** de cada linaje (Mantel-Haenszel estratificado) | Marcadores de linaje disfrazados de marcadores de fenotipo |
| **Anotación** | `annotate.locate_kmer` contra una referencia | Features sin interpretación biológica posible |

La segunda es la decisiva. Un k-mer asociado al fenotipo **dentro de cada
linaje por separado** es evidencia real; uno asociado solo entre linajes es
estructura poblacional. Esa estratificación es la corrección estándar y
FastDNA tiene los linajes gratis vía `cv.lineage_groups()`.

### Por qué nadie más puede hacerlo

Necesita las cuatro piezas a la vez: conteos de k-mer, distancias entre
muestras, clases de equivalencia y anotación. `pyseer` tiene la asociación
pero le traes las features. `equivalence`/unitigs los tiene DBGWAS pero no
hace ML. Los toolkits de interpretabilidad (SHAP, LIME) no saben qué es un
linaje.

**Tú tienes las cuatro en un proceso.**

---

## G-9 · La cohorte no cabe: el vectorizador concatena todo en memoria

**Severidad: alta a escala de cohorte real.**

`_count_cohort` (`sklearn.py:202-213`) hace `pa.concat_arrays` sobre las
tablas de k-mers de **todas** las muestras antes de aprender el vocabulario.

Para 10 000 genomas bacterianos, cada uno con ~5 M de k-mers distintos:
5×10¹⁰ entradas × 12 bytes = **600 GB**. No cabe.

El techo práctico actual está en torno a unos cientos de muestras. Las
cohortes con las que se publica hoy son de miles a decenas de miles (el
propio `cv.py` cita un análisis sobre 24 000+ genomas).

**El arreglo** encaja con G-6: `CohortCounts` mantiene los conteos en disco
en Parquet ordenado y el ranking de vocabulario se hace por **merge
en streaming** sobre las tablas ordenadas — es la misma máquina de
`counter::k_way_merge_sorted_counts` que ya existe en Rust, aplicada un
nivel más arriba. Memoria pico O(muestras), no O(muestras × k-mers).

---

## G-10 · No hay historia de despliegue: entrenas un modelo, ¿y luego?

Hoy, puntuar un genoma nuevo con un modelo entrenado exige tener las rutas
FASTQ y volver a contar a través de `transform()`. No existe:

- un artefacto que empaquete `vocabulary_` + el estimador + `k` + la
  representación,
- una ruta rápida para puntuar **una** muestra,
- ninguna comprobación de que la muestra nueva sea comparable a las de
  entrenamiento (profundidad, contaminación, especie).

```python
model = fastdna.GenomicModel.fit(counts, y, estimator=LogisticRegression())
model.save("amr_ecoli_v1.fdna-model")

# En el laboratorio, seis meses despues:
model = fastdna.GenomicModel.load("amr_ecoli_v1.fdna-model")
result = model.predict_fastq("nuevo_aislado.fastq.gz")

# Prediccion: RESISTENTE (p = 0.91)
# Avisos:
#   - profundidad 8x, por debajo del minimo de entrenamiento (20x)
#   - contencion contra la cohorte de entrenamiento 0.31: esta muestra
#     puede pertenecer a un linaje no representado
```

Esa comprobación de aplicabilidad usa `sketch`/`containment`, que ya
tienes. Un modelo que **se niega a predecir confiadamente sobre una muestra
fuera de su dominio** es exactamente lo que un uso clínico requiere y lo
que ninguna librería de k-mers ofrece.

---

## G-11 · Adopción: falta el dataset y el notebook

Nadie adopta una librería por su README. La adopta porque reprodujo un
resultado con ella en veinte minutos.

Falta:

1. **Un dataset público empaquetado.** Existen recursos estandarizados de
   benchmark para AMR ([Nature Scientific Data 2022](https://www.nature.com/articles/s41597-022-01463-7)).
   Elegir uno, `fastdna.datasets.load_amr_ecoli()` que lo descargue y
   cachee.
2. **Un notebook de extremo a extremo** que vaya de FASTQ a `audit()` a
   `explain()` con números reales.
3. **Una tabla de reproducción** en el README: nuestro AUC con CV aleatorio,
   nuestro AUC bloqueado por linaje, y el número publicado por el paper
   original — mostrando la brecha.

Ese tercer punto es la jugada de marketing más fuerte que tienes: **coger un
resultado publicado y enseñar cuánto se cae bajo evaluación honesta.** No
como ataque, sino como demostración de para qué sirve la herramienta.

---

## Prioridades revisadas

| # | Qué | Ganancia | Esfuerzo |
|---|---|---|---|
| **G-6** | `CohortCounts`: contar una vez, reusar en todos los folds | **5–11×** en todo el camino ML | 1 semana |
| **G-7** | `representation="presence"` por defecto + aviso de profundidad | Quita un confusor técnico | 2 días |
| **G-8** | `fastdna.explain()` | **Ataca el problema abierto del campo** | 3 semanas |
| G-9 | Vocabulario en streaming | Cohortes de 10 000+ | 2 semanas |
| G-11 | Dataset + notebook + tabla de reproducción | Adopción | 1 semana |
| G-10 | `GenomicModel` con comprobación de dominio | Despliegue clínico | 2 semanas |

**G-6 y G-7 van antes del primer release** (G-7 es rompedor; G-6 hace que
todo lo demás sea utilizable).

---

## La posición completa

Con `audit()` + `explain()`, la frase deja de ser sobre velocidad:

> **FastDNA responde las dos preguntas que ninguna otra librería genómica
> responde: cuánto de tu precisión es real, y en cuáles de tus features
> puedes creer.**

Contar k-mers rápido es cómo lo consigues. El paper que define los problemas
abiertos del campo nombra exactamente esos dos, y hoy nadie los tiene
resueltos en una sola herramienta.

---

# Segunda ronda — otras direcciones (2026-08-26)

Investigación en metagenómica, eucariotas y modelos fundacionales. Dos
aperturas nuevas, una de ellas grande.

---

## G-12 · Ser el validador de los modelos fundacionales genómicos

**Esta es la apertura más grande que he encontrado, y encaja con la
filosofía estrecha sin forzarla.**

### El hecho

[*Fundamental limitations of genomic language models for realistic sequence
generation* (bioRxiv, 2026)](https://www.biorxiv.org/content/10.64898/2026.01.17.700093v2)
probó Evo 2 sobre genomas procariotas, eucariotas y virales y encontró
fallos sistemáticos. Cito el hallazgo:

> *"las secuencias sintéticas capturaban las estadísticas locales de
> secuencia, pero **fallaban consistentemente en preservar la organización
> genómica de largo alcance, la composición de repeticiones y de k-mers**, y
> la arquitectura de sitios de unión de factores de transcripción."*

**Los modelos fundacionales fallan en composición de k-mers.** Y la
composición de k-mers es exactamente lo que FastDNA calcula de forma exacta
y rápida.

Segundo hecho del mismo campo: el contexto máximo típico de un gLM ronda los
**4 kb**, *"significativamente menor que el genoma bacteriano medio"* (~5 Mb).
Evo 2 llega a 1 M de tokens, pero la mayoría no.

### La oportunidad

El campo de los gLM está generando secuencias —diseño de fagos, CRISPR-Cas,
genomas sintéticos— y **no tiene una herramienta rápida para responder "¿esto
tiene composición realista?"**. Hoy esa validación se hace a mano, con
scripts ad hoc, en cada paper.

```python
verdict = fastdna.validate_generated(
    generated="evo2_phages.fasta",
    reference="real_phages.fasta",
    k=[3, 6, 11, 31],
)
print(verdict)
```

```
FastDNA generative validation — 500 generadas vs 2 841 reales

  COMPOSICION DE K-MERS
    k=3    JS-divergencia   0.008   realista
    k=6    JS-divergencia   0.031   realista
    k=11   JS-divergencia   0.194   DESVIADA
    k=31   JS-divergencia   0.612   MUY DESVIADA
    → la composicion local es correcta y la de largo alcance no,
      exactamente el fallo que describe bioRxiv 2026.

  NOVEDAD Y MEMORIZACION
    contencion media vs el conjunto real       0.71
    generadas con contencion > 0.95              23   POSIBLE COPIA
    generadas con contencion < 0.10              41   sin anclaje

  ESPECTRO
    pico de cobertura                 ausente en 88 % de las generadas
    → las secuencias generadas no tienen estructura de repeticiones
```

### Por qué esto sí encaja con "estrecho"

`docs/philosophy-narrow-not-broad.md` rechaza módulos que *"no reutilizan
prácticamente nada del motor de k-mers existente"*. Esto es lo contrario:
es `count()` más `sketch()` más comparación de espectros. **Cero algoritmos
nuevos.** Es una pregunta nueva hecha a la maquinaria que ya existe — la
extensión más barata posible.

Y te pone delante de la parte mejor financiada y con más atención de
"genética + ML" ahora mismo, en un papel —el de árbitro— que ninguna
librería de k-mers ocupa.

### El riesgo, dicho

Es un campo que se mueve muy rápido y podría estabilizarse en otra métrica.
Mitigación: es una semana de trabajo sobre maquinaria existente, no un
compromiso arquitectónico. Si no prende, el coste hundido es bajo.

---

## G-13 · Metagenómica: hay competidor directo, y un hueco al lado

### El competidor

**MetaFX** ([*Bioinformatics* 42(2), 2026](https://academic.oup.com/bioinformatics/article/42/2/btag018/8431606))
— extracción de features de datos metagenómicos de genoma completo y
clasificación de grupos de muestras. Es competencia directa de
`KmerVectorizer` en el terreno metagenómico y **es de 2026**, o sea que el
hueco se está cerrando ahora mismo.

**Decisión: no perseguir metagenómica general.** MetaFX tiene paper,
publicación en *Bioinformatics* y foco. Entrar ahí es la pelea de FastK otra
vez, en otro terreno.

### El hueco que sí queda

De la misma búsqueda:

> *"Los enfoques tradicionales para ajustar efectos de lote asumen que
> distintos taxones microbianos son independientes, pero las técnicas de
> secuenciación de microbioma generan datos de conteo que representan
> **composiciones**. Los efectos de lote pueden llevar a excesivos
> descubrimientos falsos positivos y ocultar asociaciones reales, y
> **se ha hecho relativamente poco por mitigar los efectos de lote en datos
> de microbioma**."*

Efecto de lote es **otro confusor técnico**, de la misma clase que la
profundidad (G-7) y que la estructura poblacional (`cv.py`). Las
herramientas que existen (`MBECS`, `ConQuR`) son de R y parten de una tabla
de abundancias ya construida.

Eso no es un módulo nuevo de metagenómica: es **una tercera dimensión de la
misma tesis del proyecto**.

---

## La tesis unificada (y esto reordena el README)

Los cuatro problemas que FastDNA ataca no son cuatro features. Son **uno**:

| Confusor | Pieza | Estado |
|---|---|---|
| Estructura poblacional | `cv.py` → `audit()` | existe → tarea 14 |
| Profundidad de secuenciación | `representation="presence"` | tarea 34 |
| Efecto de lote | `batch=` en `audit()` | **hueco abierto en el campo** |
| Falsa causalidad | `equivalence.py` → `explain()` | maquinaria existe → tarea 35 |

> **FastDNA es la librería que encuentra los confusores de tu ML genómico.**

Eso es una frase que ninguna otra herramienta puede decir, y es más grande y
más defendible que "de FASTQ a un modelo en el que puedes confiar" — la
contiene.

Y da la extensión natural de `audit()`: aceptar covariables técnicas
conocidas y reportar cuánto de la señal se explica por cada una.

```python
report = fastdna.audit(
    "cohort/", phenotype=y,
    covariates={"batch": batch_ids, "year": years, "center": centers},
)
```

```
  ATRIBUCION DE SEÑAL
    linaje             62%   ALTA
    lote               18%   MODERADA
    profundidad         3%   baja
    año                 1%   baja
    sin explicar       16%   <- el techo de lo que podria ser biologia
```

Esa última línea —**el techo de lo que puede ser biología real**— es el
número que un revisor quiere y que hoy nadie calcula.

---

## Eucariotas y humano: no ir

Verificado y descartado deliberadamente, para que no se reabra:

- El ML genómico humano vive en SNPs y arrays, no en k-mers, y tiene
  incumbentes enormes (PLINK, REGENIE, el ecosistema de UK Biobank).
- k ≤ 32 y el tamaño del genoma hacen el conteo exacto mucho más caro.
- La estructura poblacional humana ya tiene solución estándar y madura
  (componentes principales, GRM), así que el diferenciador de FastDNA vale
  bastante menos ahí.

El nicho es **genómica microbiana clonal**, que es donde la estructura
poblacional rompe el ML y donde nadie lo ha resuelto en herramienta.

---

## Prioridades tras la segunda ronda

| # | Qué | Ganancia | Esfuerzo |
|---|---|---|---|
| **33** | G-6 `CohortCounts` | 5–11× en todo el camino ML | 1 semana |
| **34** | G-7 `representation="presence"` | Quita el confusor de profundidad | 2 días |
| 35 | G-8 `explain()` | El problema abierto del campo | 3 semanas |
| **39** | **G-12 `validate_generated()`** | **Territorio nuevo, coste bajo** | **1 semana** |
| **40** | **G-13 `covariates=` en `audit()`** | **Cierra la tesis unificada** | **1 semana** |
| 37 | G-11 dataset + notebook | Adopción | 1 semana |
| 36 | G-9 streaming (cohortes 10 000+) | Escala | 2 semanas |
| 38 | G-10 `GenomicModel` | Despliegue clínico | 2 semanas |

39 y 40 suben por encima de 36–38: ambas son una semana sobre maquinaria
existente y las dos amplían la tesis en vez de solo profundizarla.
