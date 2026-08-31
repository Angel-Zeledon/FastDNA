# Tarea 24 — Podar la superficie Python: aviso de congelamiento y reporte de hallazgos

**Tipo:** congelamiento documental (mecánico) + reporte de arquitectura para
el dueño del proyecto. No es una tarea de implementación: no se mueve, ni
se borra, ni se extrae código.

**Objetivo:** ejecutar la parte mecánica de la decisión ya cerrada en
`docs/audit/PLAN.md` §2 (congelar `mic`, `rules`, `anomaly`,
`active_learning`, `multiomics`) y reportar dos preguntas de arquitectura
que esa decisión no pudo contemplar al escribirse, porque los módulos que
las plantean no existían todavía: si algún módulo núcleo importa alguno de
los cinco módulos congelados, y si `validate_generated()` (tarea 39 / G-12,
añadida esta misma sesión) pertenece a núcleo o es una sexta adición que
diluye el alcance.

**Archivos tocados:** las cabeceras (docstring de módulo) de
`python/fastdna/mic.py`, `python/fastdna/rules.py`,
`python/fastdna/anomaly.py`, `python/fastdna/active_learning.py`,
`python/fastdna/multiomics.py`, y este archivo.

**Archivos explícitamente NO tocados:** el resto de esos cinco archivos
(código, tests), `CHANGELOG.md`, cualquier módulo núcleo, y
`python/fastdna/validate_generated.py` (se juzga en la §3, no se edita).

---

## 1. Aviso de congelamiento (hecho)

Confirmado en los cinco módulos. Cada uno recibió una sección corta —
`## Status: frozen` o `Status: frozen` con subrayado RST, según el estilo
de encabezados que ese archivo ya usaba (`mic.py`/`rules.py` usan
encabezados `##`; `anomaly.py` usa encabezados subrayados con guiones; y
`active_learning.py`/`multiomics.py`, que no usan ningún estilo de
encabezado, recibieron el mismo texto con una frase inicial en negrita)—
inmediatamente después de la descripción de apertura y antes de la primera
sección detallada de cada docstring. El texto apunta a `docs/audit/PLAN.md`
§2 por nombre para que quien lo lea encuentre el razonamiento completo, y
dice explícitamente que congelar no es borrar. No se tocó ninguna otra
línea de esos docstrings, ni código, ni tests, ni comportamiento.

## 2. El grafo de dependencias núcleo ↔ congelado

### Método

Se buscaron, con grep sobre líneas `import`/`from` reales (no menciones en
prosa ni ejemplos dentro de un docstring), en los 13 módulos que esta tarea
trata como "núcleo/keep" — `cv`, `sklearn`, `evaluation`, `calibration`,
`gwas`, `workflow`, `interpret`, `embed`, `genomic_model`, `audit`,
`explain`, `provenance`, `cohort_counts` — todos los imports de cualquiera
de los cinco módulos congelados (`mic`, `rules`, `anomaly`,
`active_learning`, `multiomics`), y también la dirección inversa.

### Resultado: DOS dependencias reales núcleo → congelado, no sólo una

| Módulo núcleo | Importa de (congelado) | Símbolo | Cómo se usa | Tipo de import |
|---|---|---|---|---|
| `python/fastdna/genomic_model.py` (línea 88) | `anomaly.py` | `CohortOutlierFlagger` | `GenomicModel.__init__` lo instancia y ajusta **sin condición** (líneas 227-232), en cada `GenomicModel` que se crea. Es el chequeo de aplicabilidad de dominio que le da sentido a la frase "modelo honesto" en el propio docstring del módulo. | Import duro de nivel de módulo: `from .anomaly import CohortOutlierFlagger`. |
| `python/fastdna/workflow.py` (línea 110) | `rules.py` | `SetCoveringClassifier` | `_new_classifier()` cae a `SetCoveringClassifier()` cuando `classifier=None` — es decir, es el clasificador **por defecto** de `AssociationWorkflow`, la pieza que el propio docstring de `workflow.py` describe como "one entry point for the canonical cohort question". También se referencia por nombre en el mensaje de error de la etapa de anotación (`annotate_rule`/`.rules_`), que asume ese default. | Import duro de nivel de módulo: `from .rules import SetCoveringClassifier`. |

Los otros once módulos núcleo investigados (`cv`, `sklearn`, `gwas`,
`interpret`, `embed`, `audit`, `explain`, `provenance`, `cohort_counts`, y
dos que sólo *mencionan* pero no importan — ver abajo: `evaluation`,
`calibration`) no importan ninguno de los cinco congelados.

**El hallazgo `workflow.py → rules.py` no estaba en el enunciado de esta
tarea, y es más urgente que el ya conocido `genomic_model → anomaly`, no
menos.** `genomic_model.py` se añadió esta sesión, después de que
`PLAN.md` §2 se escribiera, así que es razonable que el plan no lo viera
venir. `workflow.py`, en cambio, es uno de los módulos que ya existían
cuando se escribió la tabla de `PLAN.md` §2 — la misma tabla, en el mismo
documento, con la misma fecha (2026-08-26), que clasifica `workflow` como
núcleo ("Son el camino") y `rules` como congelar ("Alcance que diluye;
ninguno está en el camino"). Es decir: la contradicción no apareció
después de cerrar la decisión — ya estaba dentro del plan el día en que se
escribió, y nadie la verificó con grep en ese momento. De los cinco
módulos congelados, `rules` es además el único que tiene un importador
núcleo real; `mic`, `active_learning` y `multiomics` no tienen ninguno.

### Menciones sólo-documentación (no son imports; no bloquean nada, pero conviene anotarlas)

`evaluation.py` y `calibration.py` (núcleo), y `report.py`, `annotate.py`,
`equivalence.py` (se mantienen) mencionan `fastdna.rules.
SetCoveringClassifier` — o, en el caso de `equivalence.py`, la convención
de nombres de `multiomics._sample_id_from_filename` — extensamente en su
prosa: el caso "hard 0/1 uncalibrated" que `calibration_report` debe
detectar, ejemplos de uso en docstrings, duck-typing explícito
(`annotate.py` dice literalmente que evita importar `fastdna.rules` a
propósito, por la misma razón que `genomic_model.py` evita importar
`fastdna.sklearn` directamente). Ninguno de estos tiene una línea
`import`/`from` real apuntando a un módulo congelado. Si `rules` se
extrajera a `fastdna-contrib`, estos docstrings quedarían citando por
nombre un paquete que ya no es parte de `fastdna` — costo de
documentación (referencias que exigen una nota de "requiere
fastdna-contrib"), no un `ImportError` en tiempo de import.

### Dirección inversa: ¿algún módulo congelado importa un módulo núcleo?

**Ninguno.** Los cinco módulos congelados están limpios de imports de los
13 módulos núcleo/keep investigados — varios de ellos lo dicen incluso en
su propio docstring como decisión deliberada (`rules.py`: "this module
deliberately does not import [cv]"; `active_learning.py`: "this module
does not import or assume the existence of fastdna.taxonomy or
fastdna.sklearn"). `anomaly.py` y `multiomics.py` sí importan primitivas
del paquete base (`sketch`, `count`, `_column_as_array`), pero esas
primitivas están ancladas directamente en el núcleo Rust vía
`fastdna/__init__.py`, no en ninguno de los 13 módulos núcleo/keep — son
infraestructura compartida por prácticamente todos los módulos del
paquete, congelados o no. Es exactamente la dirección de dependencia
"hacia abajo, hacia las primitivas" que esta tarea da por buena y
esperada, y confirma que no hay ciclos entre las dos listas.

### Opciones para `genomic_model → anomaly` (lo pedido explícitamente por esta tarea)

1. **Mover el algoritmo central de `CohortOutlierFlagger` a un módulo
   núcleo nuevo y diminuto, con `anomaly.py` re-exportándolo.** Por
   ejemplo `fastdna/_cohort_distance.py`: sólo el sketching por cohorte,
   la distancia mediana Mash "leave-one-out", y el z-score robusto de
   Iglewicz & Hoaglin (1993) — el trío que `genomic_model` realmente
   necesita. `anomaly.py` importa de ahí y re-exporta `CohortOutlierFlagger`
   (y `flag_cohort`) sin cambiar su API pública, así que ningún código de
   terceros que ya importe `fastdna.anomaly` se rompe. `genomic_model.py`
   importa del módulo núcleo nuevo, no de `anomaly`. Cuando `anomaly` se
   extraiga a `fastdna-contrib`, `genomic_model` sigue funcionando sin
   necesitarlo instalado. Costo: hay que trazar la línea entre "lo que usa
   `genomic_model`" y "lo que se queda sólo en `anomaly`" (`flag_cohort()`
   como utilidad de QC de laboratorio, el framing de "tubo intercambiado",
   etc.) sin duplicar lógica en el proceso.

2. **`genomic_model` absorbe una copia privada de sólo lo que necesita.**
   Sin depender de `anomaly.py` en absoluto: una versión mínima e interna
   del sketch-and-compare + z-score robusto, no exportada. Rompe el
   acoplamiento por completo y de inmediato. Costo: duplica lógica ya
   escrita, probada y cuidadosamente documentada (incluyendo el
   razonamiento sobre mediana vs. media, leave-one-out, y por qué z-score
   robusto y no un one-class SVM, todo en el docstring de `anomaly.py`).
   Dos copias del mismo algoritmo divergen con el tiempo — exactamente el
   tipo de deuda que el propio `validate_generated.py` dice evitar
   explícitamente al reusar `fastdna.count()` en vez de escribir un
   segundo contador.

3. **Reclasificar `anomaly` como núcleo, ya que algo del núcleo depende de
   él.** No mueve código, sólo re-etiqueta la fila de `PLAN.md` §2. Costo:
   `anomaly.py` trae consigo superficie que no le sirve a `genomic_model`
   (`flag_cohort()` como utilidad de QC de laboratorio independiente, el
   framing completo de "¿cuál muestra de mi cohorte es la rara?") —
   reclasificar el módulo entero como núcleo por la dependencia de una
   sola clase reintroduce exactamente el alcance no relacionado con el
   camino núcleo que la poda quiere evitar.

**Recomendación: opción 1.** Mantiene el plan de congelamiento/extracción
intacto para la parte de `anomaly.py` que sí es alcance-que-diluye
(`flag_cohort()`, el framing de QC de laboratorio) mientras resuelve la
única pieza que un módulo núcleo necesita de verdad, sin duplicar un
algoritmo ya razonado con cuidado (riesgo de la opción 2) ni arrastrar de
vuelta a "núcleo" superficie no relacionada (riesgo de la opción 3). Es
además el mismo patrón de re-exportación/fachada delgada que el propio
código ya usa en otros puntos (duck-typing documentado explícitamente en
`genomic_model.py` e `interpret.py` frente a `fastdna.sklearn`), así que
no introduce una convención nueva.

### Y, ya que se encontró: opciones para `workflow → rules`

No lo pidió el enunciado de esta tarea, pero es el mismo tipo de problema
y, por lo argumentado arriba, más urgente. Se documenta con el mismo
formato para que el dueño del proyecto pueda decidir ambos a la vez.

1. **Cambiar el default de `AssociationWorkflow` a un estimador que sí
   viva en núcleo** (p. ej. `LogisticRegression`, el mismo default que
   `audit()` ya usa cuando `estimator=None`, tarea 14), y dejar
   `SetCoveringClassifier` como una opción explícita que el usuario
   importa y pasa él mismo (`classifier=SetCoveringClassifier()`). El
   import de `rules` dejaría de ser de nivel de módulo. Costo: es un
   cambio de comportamiento por defecto, no cosmético — el propio
   docstring de `workflow.py` dice que `SetCoveringClassifier` es "the
   classifier every other composed stage... is built to read from"
   (`.explain()`, `export_rules_fasta()`, `annotate_rule()`), así que
   cambiar el default también cambia qué hace `run()` sin argumentos.

2. **Import perezoso, sin cambiar el default.** Mover
   `from .rules import SetCoveringClassifier` de nivel de módulo a dentro
   de `_new_classifier()`, para que sólo se importe cuando de verdad se
   necesita (`classifier is None`). No cambia comportamiento ni API —sólo
   pospone el import—, así que `import fastdna.workflow` seguiría
   funcionando si `rules` se extrae; sólo fallaría, con un `ImportError`
   claro, el caso concreto de usar el default sin `fastdna-contrib`
   instalado. Costo: no resuelve el problema de fondo, sólo lo pospone de
   tiempo-de-import a tiempo-de-uso — el camino por defecto de la pieza
   estrella del núcleo seguiría necesitando un paquete que, por
   definición, no seria una dependencia dura de `fastdna`.

3. **Reclasificar `rules` como núcleo.** Mismo razonamiento que la opción
   3 de `anomaly`: si el propio plan ya eligió `SetCoveringClassifier`
   como el clasificador por defecto de `AssociationWorkflow` — descrita en
   su propio docstring como "one entry point for the canonical cohort
   question" —, es difícil sostener que `rules.py` "no está en el camino"
   tal como el plan define esa frase.

**Recomendación:** de las tres, la opción 3 es la que mejor encaja con la
evidencia reunida aquí — a diferencia de `anomaly` (donde sólo una clase
específica, no todo el módulo, es lo que necesita núcleo), en `rules` es
el módulo entero (`SetCoveringClassifier`, con `.rules_`, `.explain()`,
`export_rules_fasta()`) el que la pieza núcleo insignia usa como su
comportamiento por defecto documentado, así que separar "lo que usa
`workflow`" de "lo que no" (opción 1, al estilo de la opción 1 de
`anomaly`) dejaría muy poco de `rules.py` realmente libre para congelar.
Pero esta es la recomendación más débil de las dos que ofrece este
reporte: reclasificar `rules` significa admitir que sólo `mic`,
`anomaly`, `active_learning` y `multiomics` — no cinco módulos, sino
cuatro — quedan genuinamente libres de importadores núcleo, y esa es una
revisión de alcance que le corresponde decidir a quien es dueño del plan,
no a este reporte.

## 3. ¿Dónde encaja `validate_generated()`?

`python/fastdna/validate_generated.py` se añadió esta sesión (tarea 39 /
G-12), después de que la tabla de `PLAN.md` §2 se escribiera, así que
nunca pasó por la regla de esa tabla: *¿sirve al camino FASTQ → modelo
honesto?*

**Qué hace, en concreto** (leído de su propio docstring): dado un lote de
secuencias generadas por un modelo genómico fundacional (p. ej. Evo 2) y
un conjunto de referencia, produce tres chequeos de plausibilidad: (1)
divergencia de Jensen-Shannon entre las distribuciones de frecuencia de
k-mers a varios `k` (bajo = composición de bases; alto = estructura de
largo alcance, que es justo donde la literatura de 2026 citada en el
propio docstring dice que estos modelos fallan); (2) reutiliza el modelo
de mezcla de `fastdna.genomescope.profile_genome()` sobre un espectro de
profundidad sintético, como analogía estructural de la estructura de
repeticiones/cobertura (no son datos de secuenciación reales, y el
docstring lo dice explícitamente); (3) *containment* de cada secuencia
generada contra la referencia vía `fastdna.sketch()`, para señalar
memorización casi-literal (containment muy alto) o disimilitud total
(containment muy bajo). Las tres construidas enteramente sobre primitivas
que el paquete ya tiene — `fastdna.count()`, `fastdna.genomescope`,
`fastdna.sketch()` — "zero new counting or sketching algorithms", en
palabras del propio módulo.

**¿Sirve al camino FASTQ → modelo honesto, en el sentido literal?** No
exactamente. Su sujeto es secuencia generada/sintética por un modelo
*generativo* fundacional, no el FASTQ de una cohorte propia que se va a
usar para entrenar un clasificador de fenotipo. No hay cohorte, no hay
fenotipo, no hay `cv.LineageKFold`, no hay clasificador en ninguna parte
de este módulo. Tomada al pie de la letra, la pregunta de la tabla de
`PLAN.md` §2 no la responde con un "sí" limpio, y sería deshonesto no
decirlo así de claro.

**¿Es entonces la sexta adición que diluye el alcance, del mismo tipo que
los cinco módulos congelados?** No, y la diferencia es estructural, no de
etiqueta. Cada uno de los cinco módulos congelados añade una **capacidad
algorítmica nueva** que compite o se superpone con algo que ya existe en
el ecosistema ML: `rules` es una familia de clasificador nueva (Set
Covering Machine) que no reusa el motor de k-mers para nada más que la
matriz de entrada; `mic` es una cabeza de regresión nueva; `anomaly` es
un detector de outliers nuevo; `active_learning` es un paradigma de
workflow nuevo (priorización para revisión humana); `multiomics` es un
motor de *joins* nuevo. `validate_generated()`, en cambio, no introduce
ningún algoritmo nuevo — es composición y reporte puro sobre motores que
`fastdna` ya publica y ya mantiene (`count`, `sketch`,
`genomescope.profile_genome`), exactamente la disciplina que
`docs/philosophy-narrow-not-broad.md` pide y que los cinco congelados, en
mayor o menor medida, no siguen. Estructuralmente es el pariente más
cercano de `genomescope`/`assembly_qc`/`spectrum` — módulos que
`PLAN.md` §2 ya clasifica como "se mantienen: producen o explican
features" —, no de los cinco módulos congelados ni, en sentido estricto,
del núcleo de predicción de fenotipo.

Que el propio `PLAN.md` lo incluyera en la tabla de la fase 1 (tarea 39,
G-12, "El foso") es evidencia de que a quien escribió el plan le pareció
buena idea el mismo día — pero, igual que con `workflow → rules` en la
§2 de este reporte, estar en la lista no es lo mismo que haber pasado el
filtro de la tabla §2 explícitamente. Lo que de verdad sostiene la
recomendación de abajo es lo que el módulo hace (cero algoritmos nuevos),
no que ya estuviera planeado.

**Recomendación:** clasificar `validate_generated` junto con "se
mantienen" (el nivel de `genomescope`/`assembly_qc`/`spectrum`) — se
queda, no es núcleo en sentido estricto, y definitivamente no es
candidato a congelamiento/extracción. La única objeción honesta es que su
"cliente" (alguien diseñando ADN sintético con un modelo fundacional
generativo) es un perfil de usuario distinto del que centra el resto de
`PLAN.md` (AMR/GWAS/QC de laboratorio sobre FASTQ propio) — vale la pena
que el dueño del proyecto lo confirme explícitamente, aunque este reporte
no lo considera una sexta adición que diluye el alcance.

---

## Qué NO se hizo (a propósito)

- No se movió, copió, ni extrajo código de ninguno de los cinco módulos
  congelados.
- No se cambió ningún import, en `workflow.py`, `genomic_model.py`, ni en
  ningún otro archivo.
- No se editó `validate_generated.py`.
- No se tocó `CHANGELOG.md`, ningún test, ni ningún módulo núcleo.
- No se ejecutó ninguna operación de git.

Las dos preguntas de arquitectura de este reporte (§2 y §3) son para que
las decida el dueño del proyecto, con la misma lógica que cerró la
decisión original de `PLAN.md` §2 — este reporte reúne evidencia y ofrece
opciones razonadas, no reemplaza esa decisión.
