# 01 — Auditoría por ejes

Auditoría del 2026-08-26 contra el árbol descrito en `00-inventario.md`.
Toda cifra viene de una ejecución real. Los hallazgos se numeran `H-01`…`H-33`.

## Resumen de notas

| Eje | Nota | Una línea |
|---|:--:|---|
| DX de instalación | **1/5** | `pip install fastdna` falla; el wheel embarca bytecode ajeno |
| Honestidad de los tipos | **1/5** | Cero información de tipos publicada |
| Versionado y releases | **1/5** | Versión duplicada a mano, sin CHANGELOG, sin tags |
| Mensajes de error | **2/5** | Siete fallos distintos colapsan a `ValueError` |
| Documentación | **2/5** | El README cubre 5 de 13 funciones públicas |
| Seguridad y cadena de suministro | **2/5** | Sin auditoría de deps, actions sin fijar |
| Salud del proyecto | **2/5** | CI no ejecuta los tests; 247 líneas muertas |
| Depuración y observabilidad | **2/5** | Sin logging; escribe en el CWD sin pedirlo |
| Diseño de API y ergonomía | **3/5** | Núcleo sólido, fronteras inexistentes |
| Interoperabilidad | **3/5** | Arrow excelente, ecosistema genómico sin tocar |
| Corrección y edge cases | **4/5** | Muy bueno; un duplicado y un no-op silencioso |
| Performance y footprint | **4/5** | Motor rápido; artefactos descuidados |
| Tests | **4/5** | 93 %/92 % reales, pero el FFI no se mide |

**Lectura global.** El motor es de calidad alta y está medido con honestidad
poco común. Lo que falla es todo lo que rodea al motor: publicación, tipos,
versionado, CI. Un usuario no puede instalar esto hoy.

---

# Eje 1 · DX de instalación — 1/5

---
**H-01 · El wheel embarca 30 archivos `.pyc` de dos versiones de CPython y crece un 34 %**

- **Severidad:** alta
- **Dónde:** `pyproject.toml:44-48` (bloque `[tool.maturin]`, sin `exclude`); se manifiesta en `python/fastdna/__pycache__/`
- **Qué pasa hoy:** `[tool.maturin]` declara `python-source = "python"` y nada más:

```toml
[tool.maturin]
features = ["python"]
module-name = "fastdna._core"
python-source = "python"
```

  maturin copia el árbol `python/` completo dentro del wheel. Si el
  desarrollador ejecutó los tests o importó el paquete antes de construir,
  `python/fastdna/__pycache__/` existe y entra entero.

- **Por qué es un problema:** medido, no supuesto. Construyendo el wheel dos
  veces en el mismo contenedor, con y sin `__pycache__` presente:

  | | Bytes del wheel | Entradas `.pyc` | Bytes de `.pyc` |
  |---|---:|---:|---:|
  | Con `__pycache__` | **1 300 864** | **30** | 715 837 |
  | Tras `rm -rf __pycache__` | **972 655** | 0 | 0 |

  Entrada exacta → salida actual → salida esperada:

  ```
  entrada:  python -c "import fastdna"   (crea __pycache__)
            maturin build --release --features python
  actual:   wheel de 1 300 864 B que contiene, entre otros,
              68 923  fastdna/__pycache__/gwas.cpython-311.pyc
              35 696  fastdna/__pycache__/__init__.cpython-311.pyc
              34 090  fastdna/__pycache__/__init__.cpython-312.pyc
  esperado: wheel de 972 655 B, 0 entradas .pyc
  ```

  El wheel está etiquetado `cp38-abi3`: dice cubrir CPython 3.8–3.13+. El
  bytecode `cpython-311` es inservible en 3.9, 3.10, 3.12 y 3.13, y el
  `cpython-312` lo es en todas las demás. Se embarcan **las dos a la vez**,
  lo que además hace el wheel no reproducible: su contenido depende de qué
  intérpretes tocaron el directorio antes de construir. CI hace checkout
  limpio, así que **el fallo solo aparece en construcciones manuales** — es
  decir, en una subida de release hecha a mano.

- **Cómo se arregla, paso a paso:**
  1. En `pyproject.toml`, dentro de `[tool.maturin]`, añadir la clave
     `exclude` con los dos patrones de bytecode.
  2. No tocar `.gitignore`: ya ignora `__pycache__/` correctamente
     (`.gitignore:14-15`); el problema es de empaquetado, no de git.

- **Código ANTES / DESPUÉS:**

ANTES (`pyproject.toml:44-48`):
```toml
[tool.maturin]
features = ["python"]
module-name = "fastdna._core"
python-source = "python"
```

DESPUÉS (`pyproject.toml:44-58`):
```toml
[tool.maturin]
features = ["python"]
module-name = "fastdna._core"
python-source = "python"
# maturin copia `python-source` entero dentro del wheel y no consulta
# .gitignore para hacerlo. Sin estos patrones, un `python -c "import
# fastdna"` o una corrida de pytest previa a `maturin build` deja
# `python/fastdna/__pycache__/` en el arbol, y ese bytecode acaba dentro
# del artefacto publicado: medido en 30 entradas y 715 837 bytes, un 34 %
# de wheel de mas. Peor que el tamano es que es bytecode de una version
# concreta de CPython dentro de un wheel abi3 que declara cubrir 3.8-3.13+,
# y que el contenido del wheel pasa a depender de que interpretes tocaron
# el directorio antes de construir. CI hace checkout limpio, asi que esto
# solo se manifiesta en construcciones manuales, que es justo el caso de
# una subida de release hecha a mano.
exclude = ["**/__pycache__/**", "**/*.pyc"]
```

- **Efectos colaterales:** ninguno en tiempo de ejecución. No es breaking
  change: los `.pyc` nunca fueron API. Las construcciones de CI producen
  hoy wheels sin `.pyc` (checkout limpio) y seguirán igual; cambia solo el
  resultado de las construcciones locales.

- **Tests a añadir:** archivo nuevo `python/tests/test_wheel_contents.py`,
  caso `test_no_bytecode_in_built_wheel`.

```python
"""Comprueba que el artefacto publicable no lleva bytecode dentro.

No es un test de la API: es un test del empaquetado. Vive aqui, y no en
CI a mano, porque el fallo que cubre solo aparece en construcciones
locales -- CI hace checkout limpio y nunca tiene __pycache__ que meter --
y por tanto no lo detectaria ningun job existente.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import zipfile

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    shutil.which("maturin") is None,
    reason="maturin no esta instalado; este test comprueba el empaquetado, no la API",
)
def test_no_bytecode_in_built_wheel(tmp_path):
    # Fuerza la condicion que dispara el fallo: bytecode presente en el
    # arbol python-source antes de construir. Sin esto el test pasaria por
    # el motivo equivocado (no habia nada que excluir).
    subprocess.run(
        [sys.executable, "-c", "import fastdna"],
        cwd=REPO_ROOT / "python",
        check=False,
        capture_output=True,
    )

    out = tmp_path / "dist"
    subprocess.run(
        ["maturin", "build", "--features", "python", "--out", str(out)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )

    wheels = list(out.glob("*.whl"))
    assert wheels, "maturin no produjo ningun wheel"

    with zipfile.ZipFile(wheels[0]) as z:
        offenders = [
            i.filename
            for i in z.infolist()
            if i.filename.endswith(".pyc") or "__pycache__" in i.filename
        ]

    assert offenders == [], (
        f"el wheel embarca {len(offenders)} archivos de bytecode: {offenders[:5]}. "
        "Revisa `exclude` en [tool.maturin] de pyproject.toml."
    )
```

- **Cómo verificar:**
  ```bash
  python -c "import sys; sys.path.insert(0,'python'); import fastdna"
  maturin build --release --features python --out /tmp/dist
  python -c "
  import glob, zipfile
  z = zipfile.ZipFile(glob.glob('/tmp/dist/*.whl')[0])
  print(sum(1 for i in z.infolist() if i.filename.endswith('.pyc')))"
  ```
  Salida esperada: `0`. Antes del arreglo imprime `30`.

- **Esfuerzo:** 15 min (1 línea + el test).
---

---
**H-02 · La primera instrucción del README no funciona: `fastdna` no existe en PyPI**

- **Severidad:** crítica
- **Dónde:** `README.md:13` (bloque `pip install fastdna`); `README.md:16-26` (párrafo que describe wheels publicados); `recipe/README.md`
- **Qué pasa hoy:** `README.md:11-15`:

```markdown
## Installation

```bash
pip install fastdna
```
```

  seguido de: *"No Rust toolchain, no compiler, no build step. `fastdna`
  ships as a prebuilt wheel…"*.

- **Por qué es un problema:** verificado por HTTP el 2026-08-26:

  ```
  entrada:  GET https://pypi.org/pypi/fastdna/json
  actual:   HTTP 404 Not Found
  entrada:  GET https://crates.io/api/v1/crates/fastdna
  actual:   HTTP 404 Not Found
  ```

  Y el repositorio lo confirma desde dentro: `git tag -l` está vacío, no hay
  `CHANGELOG`, y `.github/workflows/wheels.yml:29-30` dice literalmente
  *"This workflow only builds and tests wheels. It intentionally has no
  publish step"*.

  Un lector que llegue al repo ejecuta la primera línea y obtiene:

  ```
  actual:   ERROR: Could not find a version that satisfies the requirement fastdna
            ERROR: No matching distribution found for fastdna
  esperado: o bien la instalación funciona, o bien el README dice que aún
            no se publica y explica cómo construir desde fuente
  ```

  Esto no es un detalle cosmético: es la primera impresión del proyecto y
  hoy es falsa. Además, con ambos nombres libres, cualquiera puede
  registrarlos y el README del proyecto estaría dirigiendo usuarios a
  paquete ajeno — ver H-28.

- **Cómo se arregla, paso a paso:**
  1. Reescribir `README.md:11-26` para que el estado real sea lo primero
     que se lee, con las instrucciones de construcción desde fuente (que sí
     funcionan) como camino principal.
  2. Mantener el bloque `pip install` visible pero marcado como pendiente,
     para que vuelva a activarse con un solo borrado de aviso cuando se
     publique (tarea `12-publicar-release`).

- **Código ANTES / DESPUÉS:**

ANTES (`README.md:11-26`):
```markdown
## Installation

```bash
pip install fastdna
```

No Rust toolchain, no compiler, no build step. `fastdna` ships as a prebuilt
wheel using [PyO3's `abi3` stable ABI](https://pyo3.rs), so one wheel per
platform covers CPython 3.8 through 3.13+. CI
([`.github/workflows/wheels.yml`](.github/workflows/wheels.yml)) builds and
tests wheels for five platforms: manylinux x86_64, manylinux aarch64
(cross-compiled, built but not test-executed in CI), macOS x86_64, macOS
arm64, and Windows x86_64. The wheels are tagged `cp38-abi3`
(`requires-python = ">=3.8"`), though the CI test matrix currently runs on
Python 3.9+, so 3.8 support is declared but not exercised by CI. A Bioconda
recipe is drafted but not yet submitted -- see [Roadmap](#roadmap).
```

DESPUÉS (`README.md:11-40`):
```markdown
## Installation

> **FastDNA has not been published yet.** There is no release on PyPI, no
> crate on crates.io, and no Bioconda package: `pip install fastdna` will
> fail with `No matching distribution found`. Both names are currently
> unregistered. Until the first release lands, build from source as below.
> Progress is tracked in [Roadmap](#roadmap).

### From source (works today)

```bash
git clone https://github.com/anzeledon/fastdna
cd fastdna
pip install maturin
maturin develop --release --features python
```

That produces the same extension module a wheel would install. A Rust
toolchain (1.75+) is required for this path; a published wheel will not
require one.

### From a wheel (once published)

```bash
pip install fastdna
```

`fastdna` will ship as a prebuilt wheel using
[PyO3's `abi3` stable ABI](https://pyo3.rs), so one wheel per platform
covers CPython 3.8 through 3.13+. CI
([`.github/workflows/wheels.yml`](.github/workflows/wheels.yml)) already
builds and tests wheels for five platforms: manylinux x86_64, manylinux
aarch64 (cross-compiled, built but not test-executed in CI), macOS x86_64,
macOS arm64, and Windows x86_64 -- but has no publish step. The wheels are
tagged `cp38-abi3` (`requires-python = ">=3.8"`), while the CI test matrix
runs on Python 3.11, so 3.8 support is declared but not exercised.
```

- **Efectos colaterales:** ninguno en código. El ancla `#installation` del
  índice (`README.md:56`) sigue siendo válida. Al publicar, la tarea
  `12-publicar-release` borra el bloque de aviso y sube `pip install` de
  nuevo al principio.

- **Tests a añadir:** ninguno automatizable sin red. Se cubre por revisión
  en la tarea `12`, cuyo criterio de aceptación exige que el aviso se borre
  en el mismo commit que publica.

- **Cómo verificar:**
  ```bash
  grep -n "has not been published yet" README.md
  ```
  Salida esperada: una línea coincidente en la sección Installation.

- **Esfuerzo:** 20 min.
---

---
**H-03 · El recipe de Bioconda no puede enviarse: `sha256` son 64 ceros**

- **Severidad:** media
- **Dónde:** `recipe/meta.yaml:52` (`sha256:`), `recipe/meta.yaml:47` (`url:`)
- **Qué pasa hoy:**

```yaml
source:
  url: https://github.com/anzeledon/fastdna/archive/refs/tags/v{{ version }}.tar.gz
  # PLACEHOLDER -- 64 zeros, the right shape for a sha256 but not a real
  # checksum of anything. There is no tagged release yet to compute one
  # from. Replace before submitting.
  sha256: "0000000000000000000000000000000000000000000000000000000000000000"
```

- **Por qué es un problema:** la URL apunta a `refs/tags/v0.1.0`, que no
  existe (`git tag -l` vacío).

  ```
  entrada:  conda build recipe/
  actual:   descarga fallida (404 en la URL del tag) y, si existiera,
            SHA256 mismatch contra los 64 ceros
  esperado: build correcto
  ```

  El propio recipe documenta el bloqueo con honestidad, así que **no es un
  hallazgo de que esté mal escrito, sino de que su desbloqueo depende de
  H-02**: sin release etiquetado no hay nada que empaquetar. Se registra
  aquí para que quede encadenado en el plan, no para reescribir el archivo.

- **Cómo se arregla, paso a paso:** es la tarea `13-bioconda`, y **su único
  prerequisito real es la tarea `12-publicar-release`**. Los pasos son los
  cuatro que el propio `recipe/meta.yaml:25-33` ya enumera. Decisión cerrada:
  **no se toca `recipe/meta.yaml` ahora**; se toca en la tarea 13, después
  del tag, cuando el sha256 puede calcularse de verdad. Escribir cualquier
  otra cosa antes sería inventar un dato.
  1. (En tarea 13, tras el tag) `curl -sL <url del tag> | sha256sum`.
  2. Sustituir los 64 ceros por ese digest.
  3. Añadir un FASTQ mínimo como `recipe/test_data/tiny.fastq` y una
     sección `test.commands` que cuente sobre él.
  4. Abrir PR en `bioconda/bioconda-recipes`.

- **Código ANTES / DESPUÉS:** no aplica todavía — ver decisión arriba. El
  único cambio de contenido queda especificado en la tarea 13.

- **Efectos colaterales:** ninguno hoy.

- **Tests a añadir:** ninguno; la verificación es la CI de bioconda
  (`bioconda-utils`), que es precisamente lo que la CI propia no puede hacer.

- **Cómo verificar:**
  ```bash
  grep -c "0000000000000000000000000000000000000000000000000000000000000000" recipe/meta.yaml
  ```
  Salida esperada hoy: `1` (bloqueado, correcto). Tras la tarea 13: `0`.

- **Esfuerzo:** 5 min ahora (registrar la dependencia); 1 h en la tarea 13.
---

# Eje 2 · Honestidad de los tipos — 1/5

---
**H-04 · El paquete no publica `py.typed`: todo consumidor con mypy o pyright ve `Any`**

- **Severidad:** alta
- **Dónde:** ausencia — no existe `python/fastdna/py.typed` (`find python -name py.typed` → vacío) ni ningún `.pyi` (0 archivos); `pyproject.toml:44-48` no declara nada al respecto
- **Qué pasa hoy:** el paquete no contiene marcador PEP 561. Cualquier
  proyecto que dependa de `fastdna` y ejecute un comprobador de tipos recibe
  el paquete como no tipado.

- **Por qué es un problema:** caso reproducible en un proyecto consumidor:

  ```python
  # user_project/analysis.py
  import fastdna
  result = fastdna.count("sample.fastq", k=31)
  reveal_type(result)
  ```

  ```
  entrada:  mypy user_project/analysis.py
  actual:   error: Skipping analyzing "fastdna": module is installed, but
            missing library stubs or py.typed marker  [import-untyped]
            note: Revealed type is "Any"
  esperado: note: Revealed type is "fastdna.KmerCounts"
  ```

  El efecto es que **toda la API queda fuera de la comprobación de tipos del
  usuario**: `result.tabel` (con error tipográfico) no se detecta,
  `fastdna.count(k="31")` tampoco. Para una librería que aspira a ser la
  referencia del nicho, y cuyos competidores directos en Python (`sourmash`)
  sí publican tipos, es una diferencia visible en el primer minuto de uso.

- **Cómo se arregla, paso a paso:**
  1. Crear el archivo vacío `python/fastdna/py.typed`.
  2. Verificar que maturin lo empaqueta (lo hace: copia `python-source`
     entero; H-01 añade `exclude` solo para bytecode).
  3. Anotar la API pública de `__init__.py` — eso es H-05, tarea aparte,
     porque el marcador sin anotaciones sería peor que nada: le diría a mypy
     "confía en mis tipos" cuando no hay ninguno.

  **Decisión cerrada:** el marcador se añade **en el mismo commit que las
  anotaciones de H-05**, no antes. Alternativa descartada: publicar
  `py.typed` ya y anotar después — eso hace que mypy infiera `Any` de
  funciones sin anotar *y* deje de avisar, que es estrictamente peor que el
  estado actual.

- **Código ANTES / DESPUÉS:**

ANTES: el archivo no existe.

DESPUÉS (`python/fastdna/py.typed`), contenido literal (archivo vacío):
```
```

Y en `pyproject.toml`, dentro de `[tool.maturin]`, junto al `exclude` de H-01:
```toml
[tool.maturin]
features = ["python"]
module-name = "fastdna._core"
python-source = "python"
exclude = ["**/__pycache__/**", "**/*.pyc"]
# `python/fastdna/py.typed` es el marcador PEP 561. Va incluido por estar
# dentro de `python-source`, pero se nombra aqui porque `exclude` de arriba
# es la unica pieza que podria dejarlo fuera si alguien ampliara sus
# patrones. Sin este archivo, mypy y pyright tratan `fastdna` como no
# tipado y toda la API publica se reduce a `Any` en el proyecto que la usa.
```

- **Efectos colaterales:** a partir de aquí, un cambio de firma es visible
  para los consumidores como error de tipos. Eso es el objetivo, pero
  implica que las firmas pasan a ser parte del contrato: no cambiarlas sin
  una entrada en el CHANGELOG (H-24).

- **Tests a añadir:** archivo nuevo `python/tests/test_typing_marker.py`,
  caso `test_py_typed_marker_is_installed`.

```python
"""El marcador PEP 561 tiene que llegar instalado, no solo existir en el
repositorio: es lo que hace que mypy/pyright del proyecto consumidor lean
las anotaciones en vez de tratar `fastdna` como `Any`.

Se comprueba sobre el paquete ya importado (`fastdna.__file__`), que es la
copia instalada, y no sobre una ruta relativa al repositorio -- de lo
contrario pasaria aunque maturin lo dejara fuera del wheel, que es
exactamente el fallo que este test tiene que atrapar.
"""
from __future__ import annotations

import pathlib

import fastdna


def test_py_typed_marker_is_installed():
    package_dir = pathlib.Path(fastdna.__file__).parent
    marker = package_dir / "py.typed"
    assert marker.is_file(), (
        f"falta {marker}: sin el marcador PEP 561 los consumidores ven "
        "toda la API publica como Any"
    )
```

- **Cómo verificar:**
  ```bash
  maturin develop --features python
  python -c "import fastdna, pathlib; print((pathlib.Path(fastdna.__file__).parent / 'py.typed').is_file())"
  ```
  Salida esperada: `True`.

- **Esfuerzo:** 10 min (el archivo); depende de H-05 para tener sentido.
---

---
**H-05 · Ninguna de las 13 funciones públicas de `__init__.py` declara tipo de retorno**

- **Severidad:** alta
- **Dónde:** `python/fastdna/__init__.py:262` (`count`), `:334` (`peek`), `:347` (`build_info`), `:453` (`sketch`), `:465` (`load_sketch`), `:531` (`frac_sketch`), `:547` (`load_frac_sketch`), `:552` (`compare`), `:564` (`compare_all`), `:614` (`estimate_cardinality`); y los métodos de `KmerCounts` (`:127` `table`, `:132` `qc`, `:136` `total_kmers`, `:140` `distinct_kmers`, `:144` `k`, `:148` `filter`, `:171` `sort_by`, `:179` `top`, `:186` `_head`, `:189` `to_pandas`, `:193` `to_polars`, `:199` `spectrum`, `:211` `suggest_min_count`)
- **Qué pasa hoy:** medido sobre el paquete completo:

  | Métrica | Valor |
  |---|---:|
  | `def` de nivel superior | 159 |
  | …con retorno anotado | 23 (14,5 %) |
  | Métodos (`    def `) | 103 |
  | …con retorno anotado | **2 (1,9 %)** |
  | `def` de nivel superior en `__init__.py` | 13 |
  | …con retorno anotado | **0** |

  Ejemplo real (`python/fastdna/__init__.py:334-345`):

```python
def peek(path, *, n_reads=10_000):
    """Samples the first `n_reads` records of `path` without counting the
    whole file, returning a `Preview` with read-length and GC statistics
    and a suggested `k`.
    ...
    """
    return _core.peek(str(path), n_reads)
```

- **Por qué es un problema:** el README documenta las firmas *con* tipos —
  `README.md:540` escribe literalmente
  `fastdna.count(...) -> KmerCounts` y `README.md:636`
  `fastdna.peek(path, *, n_reads=10_000) -> Preview`. **La documentación
  promete un contrato que el código no declara.** Caso reproducible en un
  editor con Pylance/pyright:

  ```
  entrada:  import fastdna
            fastdna.peek("s.fastq").suggest_k()
  actual:   sin autocompletado tras `peek(...)`; `suggest_k` no se ofrece,
            y `peek("s.fastq").nonexistent_method()` no da error
  esperado: autocompletado de Preview; error en el método inexistente
  ```

- **Cómo se arregla, paso a paso:**
  1. Añadir `from __future__ import annotations` al principio de
     `python/fastdna/__init__.py` (permite escribir `KmerCounts` como
     retorno de un método de la propia clase sin comillas, y mantiene
     compatibilidad con 3.8, que `pyproject.toml:8` declara soportar).
  2. Anotar las 13 funciones de nivel superior y los métodos de las tres
     clases.
  3. **Decisión cerrada sobre `path`:** se anota como
     `str | os.PathLike[str]`, no como `str`. El código hace `str(path)`
     internamente (`:345`, `:462`, …), así que acepta `pathlib.Path` de
     hecho; anotarlo como `str` haría fallar a mypy en el uso más común de
     la librería. Alternativa descartada: `Any`, que no dice nada.
  4. **Decisión cerrada sobre `qc` y `build_info`:** devuelven `dict` de
     forma libre desde Rust; se anotan `dict[str, object]`, no un
     `TypedDict`. Razón: las claves las produce `src/ffi.rs:269` y
     `:720` y cambiarían sin que el `TypedDict` se entere, dando falsa
     seguridad. `object` obliga al consumidor a estrechar, que es correcto.

- **Código ANTES / DESPUÉS:** (fragmento representativo; la tarea `04`
  lleva el bloque completo de las 13 funciones)

ANTES (`python/fastdna/__init__.py:1-18`):
```python
"""FastDNA -- a fast genomic k-mer counter.

This module re-exports the small FFI surface defined in `src/ffi.rs`
(compiled as the `fastdna._core` extension module) and adds nothing heavy:
anything that can be expressed in pure Python lives here instead of crossing
the Rust/Python boundary, per the packaging design (docs/superpowers/specs/
2026-08-22-fastdna-python-design.md, §9).
"""

import pyarrow as pa
import pyarrow.compute as pc

from . import _core
from ._progress import make_progress_adapter
from .spectrum import suggest_min_count as _suggest_min_count

__version__ = _core.__version__
```

DESPUÉS (`python/fastdna/__init__.py:1-30`):
```python
"""FastDNA -- a fast genomic k-mer counter.

This module re-exports the small FFI surface defined in `src/ffi.rs`
(compiled as the `fastdna._core` extension module) and adds nothing heavy:
anything that can be expressed in pure Python lives here instead of crossing
the Rust/Python boundary, per the packaging design (docs/superpowers/specs/
2026-08-22-fastdna-python-design.md, §9).
"""

# Required, not cosmetic: `KmerCounts.filter()` and friends return
# `KmerCounts`, which does not exist yet at the point its own methods are
# defined. PEP 563 postpones evaluation so the annotation can name the
# class being defined without quoting it, and it keeps these annotations
# legal on Python 3.8, which `pyproject.toml`'s `requires-python` declares
# support for.
from __future__ import annotations

import os
from typing import Callable, Iterable, Union

import pyarrow as pa
import pyarrow.compute as pc

from . import _core
from ._progress import make_progress_adapter
from .spectrum import suggest_min_count as _suggest_min_count

__version__: str = _core.__version__

# Every public entry point that names an input file accepts anything
# `str()` turns into a usable path -- the implementations all call
# `str(path)` before crossing into Rust -- so `pathlib.Path` works today
# and annotating these as plain `str` would make mypy reject the most
# common way the library is actually called.
PathLike = Union[str, "os.PathLike[str]"]
```

ANTES (`python/fastdna/__init__.py:334-345`):
```python
def peek(path, *, n_reads=10_000):
```

DESPUÉS:
```python
def peek(path: PathLike, *, n_reads: int = 10_000) -> "_core.Preview":
```

ANTES (`python/fastdna/__init__.py:148`, `:171`, `:179`):
```python
    def filter(self, min_count=None, max_count=None):
    def sort_by(self, column="frequency", *, descending=True):
    def top(self, n):
```

DESPUÉS:
```python
    def filter(
        self, min_count: int | None = None, max_count: int | None = None
    ) -> KmerCounts:
    def sort_by(self, column: str = "frequency", *, descending: bool = True) -> KmerCounts:
    def top(self, n: int) -> KmerCounts:
```

- **Efectos colaterales:** `from __future__ import annotations` debe ser la
  primera sentencia tras el docstring; ponerlo después de los `import`
  actuales es `SyntaxError`. No hay cambio de comportamiento en runtime
  (PEP 563 no evalúa las anotaciones). No es breaking change.

- **Tests a añadir:** archivo nuevo `python/tests/test_public_api_is_typed.py`,
  caso `test_every_public_callable_declares_a_return_type`.

```python
"""Impide que la API publica vuelva a quedarse sin anotar.

Un `py.typed` (H-04) le dice a mypy "confia en mis tipos". Este test es lo
que hace que esa promesa siga siendo cierta cuando alguien anada una
funcion publica nueva: sin el, el marcador degrada en silencio a medida
que la superficie crece.
"""
from __future__ import annotations

import inspect

import fastdna

PUBLIC_CALLABLES = [
    fastdna.count,
    fastdna.peek,
    fastdna.build_info,
    fastdna.sketch,
    fastdna.load_sketch,
    fastdna.frac_sketch,
    fastdna.load_frac_sketch,
    fastdna.compare,
    fastdna.compare_all,
    fastdna.estimate_cardinality,
]


def test_every_public_callable_declares_a_return_type():
    missing = [
        fn.__name__
        for fn in PUBLIC_CALLABLES
        if inspect.signature(fn).return_annotation is inspect.Signature.empty
    ]
    assert missing == [], f"sin anotacion de retorno: {missing}"


def test_every_public_callable_annotates_all_its_parameters():
    missing = []
    for fn in PUBLIC_CALLABLES:
        for name, param in inspect.signature(fn).parameters.items():
            if param.annotation is inspect.Parameter.empty:
                missing.append(f"{fn.__name__}({name})")
    assert missing == [], f"parametros sin anotar: {missing}"


def test_kmer_counts_chaining_methods_return_kmer_counts():
    # Las anotaciones son cadenas por PEP 563; se comparan como texto, que
    # es lo que un comprobador de tipos resolvera.
    for name in ("filter", "sort_by", "top"):
        method = getattr(fastdna.KmerCounts, name)
        annotation = inspect.signature(method).return_annotation
        assert annotation in ("KmerCounts", fastdna.KmerCounts), (
            f"KmerCounts.{name} declara {annotation!r}; el encadenamiento "
            "documentado exige KmerCounts"
        )
```

- **Cómo verificar:**
  ```bash
  python -m pytest python/tests/test_public_api_is_typed.py -q
  ```
  Salida esperada: `3 passed`.

- **Esfuerzo:** 2 h.
---

# Eje 3 · Diseño de API y ergonomía — 3/5

---
**H-06 · `KmerCounts.top()` acepta negativos, `None` y decimales, y devuelve resultados plausibles pero equivocados**

- **Severidad:** alta
- **Dónde:** `python/fastdna/__init__.py:179-184` (`top`), `python/fastdna/__init__.py:186-187` (`_head`)
- **Qué pasa hoy:**

```python
    def top(self, n):
        """The `n` most frequent k-mers in the current view, as a new
        `KmerCounts` -- sugar for `.sort_by("frequency").table.slice(0, n)`
        that stays chainable, e.g. `counts.filter(min_count=5).top(20)`.
        """
        return self.sort_by("frequency", descending=True)._head(n)

    def _head(self, n):
        return KmerCounts(self._raw, self.table.slice(0, n))
```

  `n` se pasa sin validar a `pyarrow.Table.slice(offset, length)`.

- **Por qué es un problema:** ejecutado sobre una tabla de 5 filas
  (sonda real, 2026-08-26):

  | Entrada | Salida actual | Salida esperada |
  |---|---|---|
  | `counts.top(-5)` | `len() == 0` | `ValueError: top(n) needs n >= 0, got -5` |
  | `counts.top(None)` | `len() == 5` (**todas**) | `TypeError` |
  | `counts.top(2.5)` | `len() == 2` | `TypeError` |
  | `counts.top(0)` | `len() == 0` | `len() == 0` (correcto) |
  | `counts.top(999)` | `len() == 5` (correcto) | `len() == 5` |

  Los tres primeros son fallos silenciosos y el peor es `top(None)`:
  devuelve **el conjunto completo**, lo contrario de "los n más frecuentes".
  El caso real que esto rompe:

  ```python
  n = config.get("top_kmers")        # ausente en el config -> None
  top = counts.top(n)                 # devuelve 53 millones de filas
  df = top.to_pandas()                # y aqui se cae la maquina
  ```

  `top(-5)` es igual de traicionero: un `n = len(x) - 10` que se vuelve
  negativo produce una tabla vacía que parece un resultado biológico
  ("ningún k-mer superó el filtro") en vez de un error de programa.

- **Cómo se arregla, paso a paso:**
  1. En `python/fastdna/__init__.py`, validar `n` al principio de `top`.
  2. **Decisión cerrada:** rechazar `bool` explícitamente. En Python
     `isinstance(True, int)` es `True`, así que `counts.top(True)` pasaría
     como `n = 1`. Se descarta aceptarlo: `top(True)` no significa nada y
     es casi seguro un argumento mal puesto.
  3. **Decisión cerrada:** `n` negativo lanza `ValueError`, no se satura a
     0. Saturar preservaría el fallo silencioso que este hallazgo describe.
  4. No tocar `_head`: es privado y sus dos llamadas (`top`, y nada más)
     quedan validadas arriba.

- **Código ANTES / DESPUÉS:**

ANTES (`python/fastdna/__init__.py:179-187`):
```python
    def top(self, n):
        """The `n` most frequent k-mers in the current view, as a new
        `KmerCounts` -- sugar for `.sort_by("frequency").table.slice(0, n)`
        that stays chainable, e.g. `counts.filter(min_count=5).top(20)`.
        """
        return self.sort_by("frequency", descending=True)._head(n)

    def _head(self, n):
        return KmerCounts(self._raw, self.table.slice(0, n))
```

DESPUÉS:
```python
    def top(self, n: int) -> KmerCounts:
        """The `n` most frequent k-mers in the current view, as a new
        `KmerCounts` -- sugar for `.sort_by("frequency").table.slice(0, n)`
        that stays chainable, e.g. `counts.filter(min_count=5).top(20)`.

        `n` must be a non-negative `int`. The validation below is not
        defensive boilerplate: `pyarrow.Table.slice()` accepts all three of
        the values it rejects and answers each of them plausibly rather
        than raising. `slice(0, None)` means "to the end", so `top(None)`
        used to return the *entire* table -- the exact opposite of what the
        name promises, and a config key that resolved to `None` would
        silently hand back 53 million rows. `slice(0, -5)` returns an empty
        table, so an `n` that went negative through arithmetic looked like
        the biological result "no k-mer passed the filter" instead of a
        bug. A float was truncated silently. `bool` is rejected ahead of
        `int` on purpose: `isinstance(True, int)` is `True` in Python, so
        `top(True)` would otherwise slip through as `n = 1`.
        """
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(
                f"top(n) needs a non-negative int, got {type(n).__name__}: {n!r}"
            )
        if n < 0:
            raise ValueError(f"top(n) needs n >= 0, got {n}")
        return self.sort_by("frequency", descending=True)._head(n)

    def _head(self, n: int) -> KmerCounts:
        return KmerCounts(self._raw, self.table.slice(0, n))
```

- **Efectos colaterales:** **breaking change** para cualquier código que
  hoy dependa de `top(None)` devolviendo todo o de `top(negativo)`
  devolviendo vacío. Ambos comportamientos son accidentes de
  `pyarrow.slice`, ninguno está documentado y el proyecto no tiene usuarios
  publicados (H-02), así que la migración es: usar `counts` directamente en
  lugar de `top(None)`. Entra en el CHANGELOG (H-24) como
  `Changed (breaking)`.

- **Tests a añadir:** en `python/tests/test_fluent_api.py`, caso
  `test_top_rejects_arguments_pyarrow_would_have_accepted`.

```python
def test_top_rejects_arguments_pyarrow_would_have_accepted(tmp_path):
    """`pyarrow.Table.slice` responde a None, negativos y floats en vez de
    fallar, y cada respuesta es plausible pero equivocada. Se fija aqui
    que `top()` no herede ese comportamiento.
    """
    import pytest

    path = tmp_path / "s.fastq"
    path.write_text("".join(f"@r{i}\nACGTACGTACGT\n+\n{'I'*12}\n" for i in range(20)))
    counts = fastdna.count(str(path), k=5)
    assert len(counts) > 0, "el fixture debe producir k-mers o el test no prueba nada"

    # El fallo mas grave: slice(0, None) significa "hasta el final".
    with pytest.raises(TypeError, match="non-negative int"):
        counts.top(None)

    # Un n negativo llegado por aritmetica parecia "ningun k-mer paso".
    with pytest.raises(ValueError, match="n >= 0"):
        counts.top(-5)

    with pytest.raises(TypeError, match="non-negative int"):
        counts.top(2.5)

    # bool es subclase de int en Python; top(True) no significa nada.
    with pytest.raises(TypeError, match="non-negative int"):
        counts.top(True)

    # Los casos legitimos siguen intactos.
    assert len(counts.top(0)) == 0
    assert len(counts.top(10 ** 9)) == len(counts)
```

- **Cómo verificar:**
  ```bash
  python -m pytest python/tests/test_fluent_api.py -q -k top_rejects
  ```
  Salida esperada: `1 passed`.

- **Esfuerzo:** 40 min.
---

---
**H-07 · El CLI no alcanza once funciones que la librería sí expone**

- **Severidad:** alta
- **Dónde:** `src/cli.rs:102-232` (struct `Cli`, 17 flags, `grep -c "Subcommand" src/cli.rs` → 0); `src/ffi.rs:409`, `:705`, `:719`, `:820`, `:828`, `:898`, `:906`, `:922`, y los tres `#[pyfunction]` de traducción y `build_database`
- **Qué pasa hoy:** el binario hace exactamente una cosa: contar. `Cli` no
  tiene ningún campo `#[command(subcommand)]`. Todo lo demás de la librería
  —`peek`, `sketch`, `load_sketch`, `frac_sketch`, `load_frac_sketch`,
  `estimate_cardinality`, `translate_sequences`, `translate_file`,
  `protein_kmers`, `build_database`, `build_info`— sólo existe cruzando el
  FFI desde Python.

- **Por qué es un problema:** el usuario de bioinformática vive en el shell
  y en Nextflow/Snakemake. Reproducible:

  ```
  entrada:  fastdna sketch sample.fastq -o sample.sig
  actual:   error: unexpected argument 'sketch' found
  esperado: escribe el sketch
  ```

  `docs/feature-gap-analysis.md` ya lo registró como ítem **Q4** el
  2026-08-24 y sigue sin hacerse. Los competidores lo tienen todos:
  `kmc_tools` con `transform`/`simple`/`filter`, FastK con `Histex`,
  `Tabex`, `Profex`, `Logex`, sourmash con `sourmash sketch dna`,
  `sourmash compare`, `sourmash gather`.

  Consecuencia concreta: `workflow_templates/nextflow` y
  `workflow_templates/snakemake` sólo pueden invocar el conteo; cualquier
  otro paso obliga a escribir un `python -c "..."` incrustado en el
  pipeline.

- **Cómo se arregla, paso a paso:** es un cambio grande; la tarea `20`
  lleva el detalle. Esqueleto de la decisión, cerrada aquí:
  1. **Decisión cerrada:** se añade `Commands` como
     `Option<Commands>` y **el conteo sigue siendo el comportamiento por
     defecto cuando no se nombra subcomando**. Alternativa descartada:
     hacer `count` obligatorio — rompería todos los `workflow_templates` y
     todos los ejemplos del README a cambio de nada.
  2. Primeros tres subcomandos: `sketch`, `dist`, `card`. Cubren el 80 %
     de lo que un pipeline necesita y reutilizan FFI ya probado.
  3. `peek` y `translate` quedan para una segunda tanda (tarea `21`).

- **Código ANTES / DESPUÉS:**

ANTES (`src/cli.rs:102-107`):
```rust
#[derive(Parser, Debug)]
#[command(name = "fastdna", version = "0.1.0", author = "FastDNA Team")]
pub struct Cli {
    /// Input FASTQ/FASTA file(s), optionally gzipped, or "-" for stdin.
```

DESPUÉS (`src/cli.rs:102-140`):
```rust
/// The subcommands that reach library features the flat flag interface
/// cannot express.
///
/// `Option<Commands>` rather than a required field, and counting stays the
/// behaviour when none is named: `fastdna --input x --output y` is what
/// every example in the README, both files under `workflow_templates/`,
/// and every integration test in `tests/cli_args.rs` already run. Making
/// `count` mandatory would break all of them to buy nothing.
#[derive(clap::Subcommand, Debug)]
pub enum Commands {
    /// Write a MinHash sketch of the input, for later comparison.
    Sketch {
        /// Input FASTQ/FASTA file, optionally gzipped, or "-" for stdin.
        #[arg(short, long, value_name = "FILE")]
        input: PathBuf,
        /// Where to write the sketch.
        #[arg(short, long, value_name = "FILE")]
        output: PathBuf,
        /// Length of k-mers (1 <= k <= 32).
        #[arg(short, long, default_value_t = 21)]
        kmer_size: usize,
        /// Number of hashes to keep.
        #[arg(long, default_value_t = 1000)]
        sketch_size: usize,
    },
    /// Compare two previously written sketches.
    Dist {
        /// The two sketch files to compare.
        #[arg(value_name = "SKETCH", num_args = 2)]
        sketches: Vec<PathBuf>,
        /// Which measure to report.
        #[arg(long, value_enum, default_value = "jaccard")]
        metric: CliMetric,
    },
    /// Estimate the number of distinct k-mers without counting them.
    Card {
        /// Input FASTQ/FASTA file, optionally gzipped, or "-" for stdin.
        #[arg(short, long, value_name = "FILE")]
        input: PathBuf,
        /// Length of k-mers (1 <= k <= 32).
        #[arg(short, long, default_value_t = 31)]
        kmer_size: usize,
        /// HyperLogLog precision; the sketch uses 2^precision bytes.
        #[arg(long, default_value_t = 14)]
        precision: u32,
    },
}

/// Which similarity measure `fastdna dist` reports.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CliMetric {
    #[default]
    Jaccard,
    Containment,
    MashDistance,
}

#[derive(Parser, Debug)]
#[command(name = "fastdna", version = env!("CARGO_PKG_VERSION"), author = "FastDNA Team")]
pub struct Cli {
    /// The operation to run. Omitted means "count", which is what every
    /// existing invocation does.
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Input FASTQ/FASTA file(s), optionally gzipped, or "-" for stdin.
```

- **Efectos colaterales:** `Cli::validate` (`src/cli.rs:236-311`) sólo debe
  ejecutar sus comprobaciones de `--input`/`--paired-dir` cuando
  `command.is_none()`; con un subcomando presente, `input` vacío es legal.
  `src/main.rs` necesita un `match` sobre `command` antes de la ruta de
  conteo. `tests/cli_args.rs` (23 tests) sigue pasando sin cambios porque
  ninguno nombra subcomando. No es breaking change.

- **Tests a añadir:** en `tests/cli_args.rs`, caso
  `counting_still_works_with_no_subcommand_named`.

```rust
/// La garantia que hace no-breaking la introduccion de subcomandos: una
/// linea de comandos sin subcomando sigue siendo un conteo. Si esto se
/// rompe, se rompen a la vez los dos `workflow_templates/`, todos los
/// ejemplos del README y los otros 23 tests de este archivo.
#[test]
fn counting_still_works_with_no_subcommand_named() {
    let cli = Cli::try_parse_from([
        "fastdna",
        "--input",
        "sample.fastq",
        "--output",
        "counts.parquet",
    ])
    .expect("la forma sin subcomando debe seguir parseando");

    assert!(cli.command.is_none(), "sin subcomando nombrado, `command` es None");
    assert_eq!(cli.input, vec![PathBuf::from("sample.fastq")]);
    assert!(cli.validate().is_ok());
}

#[test]
fn sketch_subcommand_parses_its_own_flags() {
    let cli = Cli::try_parse_from([
        "fastdna", "sketch", "--input", "s.fastq", "--output", "s.sig", "--kmer-size", "21",
    ])
    .expect("el subcomando sketch debe parsear");

    match cli.command {
        Some(Commands::Sketch { input, output, kmer_size, sketch_size }) => {
            assert_eq!(input, PathBuf::from("s.fastq"));
            assert_eq!(output, PathBuf::from("s.sig"));
            assert_eq!(kmer_size, 21);
            assert_eq!(sketch_size, 1000, "el default documentado");
        }
        other => panic!("se esperaba Commands::Sketch, se obtuvo {other:?}"),
    }
}

/// Con un subcomando presente, `--input` vacio en el nivel raiz es legal:
/// el subcomando trae el suyo. Sin este ajuste, `validate` rechazaria
/// toda invocacion de subcomando con "no input given".
#[test]
fn validate_does_not_demand_root_input_when_a_subcommand_is_named() {
    let cli = Cli::try_parse_from(["fastdna", "card", "--input", "s.fastq"])
        .expect("el subcomando card debe parsear");
    assert!(cli.validate().is_ok(), "validate no debe exigir --input de raiz aqui");
}
```

- **Cómo verificar:**
  ```bash
  cargo test --test cli_args
  cargo run --release -- sketch --input test.fastq --output /tmp/t.sig
  ```
  Salida esperada: `26 passed` y un archivo `/tmp/t.sig` escrito.

- **Esfuerzo:** 6 h (los tres subcomandos, el `match` de `main.rs` y los tests).
---

---
**H-08 · `lib.rs` publica los 24 módulos: no existe frontera entre API y detalle interno**

- **Severidad:** media
- **Dónde:** `src/lib.rs:3-28` (24 declaraciones `pub mod`)
- **Qué pasa hoy:**

```rust
pub mod adaptive_bins;
pub mod atomic;
pub mod binned;
pub mod cli;
pub mod cms;
pub mod cohort;
pub mod counter;
pub mod disk_spill;
pub mod error;
pub mod export;
pub mod fastq;
pub mod hll;
#[cfg(feature = "python")]
pub mod ffi;
pub mod kmer;
pub mod mem_estimate;
pub mod metagenomics;
pub mod minimizer;
pub mod pipeline;
pub mod preview;
pub mod progress;
pub mod qc;
pub mod sketch;
pub mod superkmer;
pub mod translate;
pub mod wasm;
```

  Son **296 ítems `pub`** alcanzables desde fuera.

- **Por qué es un problema:** módulos que son claramente implementación
  —`adaptive_bins` (el mapa de bins KMC2), `superkmer` (el empaquetado de
  2 bits), `disk_spill` (los archivos temporales), `atomic` (el renombrado
  de escritura), `cli` (la struct de clap)— forman parte del contrato
  público. Caso concreto:

  ```
  entrada:  un consumidor escribe
              use fastdna_core::adaptive_bins::DynamicBinMap;
  actual:   compila; ese consumidor queda acoplado a una estructura interna
            que `docs/design-minimizer-counting.md` describe como sujeta a
            cambio, y cualquier ajuste del mapa de bins pasa a ser breaking
  esperado: no compila; `DynamicBinMap` es detalle de `binned`
  ```

  Hoy no hay usuarios (H-02), así que el coste de arreglarlo es cero. En
  cuanto se publique, cada uno de esos 296 ítems es un compromiso.

- **Cómo se arregla, paso a paso:**
  1. **Decisión cerrada** sobre qué es API: se mantienen públicos los
     módulos que el binario, el FFI o un consumidor razonable necesitan —
     `error`, `kmer`, `counter`, `fastq`, `pipeline`, `sketch`, `export`,
     `qc`, `preview`, `progress`, `translate`, `metagenomics`, `hll`,
     `cohort`, `mem_estimate`, `cli`. Pasan a `pub(crate)`:
     `adaptive_bins`, `atomic`, `binned`, `disk_spill`, `minimizer`,
     `superkmer`. Alternativa descartada: marcarlos `#[doc(hidden)]` —
     esconde de la documentación pero sigue siendo API que compila.
  2. `cms` se borra entero (H-29), no se degrada.
  3. `wasm` y `ffi` quedan públicos: son puntos de entrada por definición.
  4. Ajustar los usos internos que dejen de compilar.

- **Código ANTES / DESPUÉS:**

ANTES (`src/lib.rs:1-31`) — ver bloque de arriba.

DESPUÉS (`src/lib.rs:1-46`):
```rust
// src/lib.rs

// The public surface. These are the modules a consumer of `fastdna_core`
// is expected to name, and the ones the `[[bin]]`, the `python` feature's
// `ffi` and the `wasm` feature all reach through.
pub mod cli;
pub mod cohort;
pub mod counter;
pub mod error;
pub mod export;
pub mod fastq;
pub mod hll;
pub mod kmer;
pub mod mem_estimate;
pub mod metagenomics;
pub mod pipeline;
pub mod preview;
pub mod progress;
pub mod qc;
pub mod sketch;
pub mod translate;

// Implementation. Reachable from anywhere inside the crate, from nowhere
// outside it.
//
// Every one of these is a strategy or a storage detail whose shape is
// documented as subject to change: `adaptive_bins` is the KMC2-style bin
// map (`docs/design-minimizer-counting.md` step 6), `superkmer` is the
// 2-bit packing, `binned` is the opt-in strategy that
// `pipeline::resolve_strategy` deliberately never selects automatically,
// `disk_spill` owns temporary files, `atomic` owns write-then-rename,
// `minimizer` owns the signature function. Leaving them `pub` made all of
// them part of the crate's compatibility contract; each is exercised by
// its own tests and by `pipeline`, not by outside callers.
pub(crate) mod adaptive_bins;
pub(crate) mod atomic;
pub(crate) mod binned;
pub(crate) mod disk_spill;
pub(crate) mod minimizer;
pub(crate) mod superkmer;

// Entry points for the optional targets, public by definition.
#[cfg(feature = "python")]
pub mod ffi;
pub mod wasm;

pub use error::{FastDnaError, Result};
pub use progress::{Progress, ProgressFn};
```

- **Efectos colaterales:** `examples/binned_occupancy_report.rs` usa
  `fastdna_core::binned` y `fastdna_core::minimizer` desde fuera del crate;
  deja de compilar. **Decisión cerrada:** ese ejemplo se convierte en un
  test `#[ignore]`d dentro de `src/binned.rs`, porque su propósito
  (informe de ocupación por bin) es diagnóstico interno, no una demostración
  de API para usuarios. Es breaking change para consumidores del crate; no
  hay ninguno (H-02).

- **Tests a añadir:** archivo nuevo `tests/public_surface.rs`, caso
  `implementation_modules_are_not_reachable_from_outside`.

```rust
//! La frontera entre API publica y detalle interno, comprobada por el
//! compilador en vez de por revision.
//!
//! Un test de integracion vive fuera del crate, asi que solo ve lo que un
//! consumidor veria. Que este archivo compile es la prueba de que los
//! modulos publicos siguen siendolo; que los `pub(crate)` no se puedan
//! nombrar aqui lo garantiza el propio `pub(crate)`, y anadir una linea
//! que los nombre haria fallar la compilacion -- que es exactamente el
//! aviso que se quiere.

/// Cada `use` de aqui es una promesa de compatibilidad explicita. Anadir
/// uno significa "esto es API"; quitarlo, "esto deja de serlo".
#[test]
fn the_documented_public_modules_are_reachable() {
    use fastdna_core::counter::KmerCounter;
    use fastdna_core::error::FastDnaError;
    use fastdna_core::kmer::{canonical_kmer_u64, extract_canonical_kmers};
    use fastdna_core::pipeline::CountStrategy;
    use fastdna_core::sketch::GenomeSketch;

    let counter = KmerCounter::new(4);
    assert_eq!(counter.k(), 4);

    assert_eq!(extract_canonical_kmers(b"ACGT", 4).len(), 1);
    assert_eq!(canonical_kmer_u64(0b00_01_10_11, 4), 0b00_01_10_11);

    let sketch = GenomeSketch::new(10, 21);
    assert_eq!(sketch.k(), 21);

    assert_eq!(CountStrategy::InMemory.as_str(), "in-memory");

    let err = FastDnaError::InvalidK { k: 99 };
    assert!(err.to_string().contains("99"));
}
```

- **Cómo verificar:**
  ```bash
  cargo test --test public_surface
  cargo build --all-targets
  ```
  Salida esperada: `1 passed`, y la compilación completa sin errores.

- **Esfuerzo:** 2 h.
---

---
**H-09 · Ocho de veintiséis módulos Python no definen `__all__`, incluido `__init__.py`**

- **Severidad:** media
- **Dónde:** `python/fastdna/__init__.py` (sin `__all__`), y `_progress.py`, `assembly_qc.py`, `embed.py`, `interop.py`, `interpret.py`, `rules.py`, `sklearn.py`, `spectrum.py`
- **Qué pasa hoy:** 18 de 26 módulos declaran `__all__` con cuidado —
  `annotate.py` lista siete nombres, `taxonomy.py` cinco— y ocho no
  declaran nada. El más importante de esos ocho es `__init__.py`.

- **Por qué es un problema:** sin `__all__`, todo nombre importado al
  módulo pasa a ser parte de su superficie aparente:

  ```
  entrada:  python -c "import fastdna; print(fastdna.pa, fastdna.pc, fastdna.os)"
  actual:   imprime los modulos pyarrow, pyarrow.compute y os
            -- `fastdna.pa` es hoy un atributo publico del paquete
  esperado: AttributeError: module 'fastdna' has no attribute 'pa'
  ```

  Consecuencias reales: `from fastdna import *` arrastra `pa`, `pc` y
  `_core`; las herramientas de documentación automática listan `pa` como
  parte de la API; y —lo que más pesa una vez exista `py.typed` (H-04)—
  pyright trata los reexports sin `__all__` como privados y **avisa al
  consumidor de que `fastdna.count` no es un reexport explícito**.

- **Cómo se arregla, paso a paso:**
  1. Añadir `__all__` a `python/fastdna/__init__.py` con las 13 funciones y
     3 clases públicas, más `__version__`.
  2. **Decisión cerrada:** los submódulos (`annotate`, `gwas`, …) **no** se
     listan en el `__all__` del paquete. Se importan explícitamente
     (`from fastdna import gwas`), que es como los documenta el README y
     como los usan los tests. Listarlos forzaría a importarlos todos al
     cargar `fastdna`, y varios arrastran scipy/sklearn — que son extras
     opcionales, no dependencias de runtime.
  3. Añadir `__all__` a los siete submódulos restantes con sus nombres ya
     públicos de facto (los que sus tests importan).

- **Código ANTES / DESPUÉS:**

ANTES (`python/fastdna/__init__.py:17`):
```python
__version__ = _core.__version__
```

DESPUÉS (`python/fastdna/__init__.py:28-56`):
```python
__version__: str = _core.__version__

# Sin esto, todo lo importado arriba -- `pa`, `pc`, `os`, `_core` -- es un
# atributo publico del paquete: `fastdna.pa` resuelve hoy al modulo
# pyarrow entero. Tambien es lo que hace que pyright trate estos nombres
# como reexports deliberados en vez de avisar al consumidor de que
# `fastdna.count` no lo es, algo que solo se vuelve visible una vez el
# paquete publica `py.typed`.
#
# Los submodulos (`gwas`, `annotate`, `taxonomy`, ...) NO se listan a
# proposito: se importan por su nombre (`from fastdna import gwas`), que
# es como los documenta el README. Listarlos aqui obligaria a cargarlos al
# importar `fastdna`, y varios arrastran scipy o scikit-learn, que son
# extras opcionales y no dependencias de runtime.
__all__ = [
    "__version__",
    # Conteo
    "count",
    "KmerCounts",
    # Inspeccion previa
    "peek",
    "build_info",
    # Sketching
    "sketch",
    "load_sketch",
    "Sketch",
    "frac_sketch",
    "load_frac_sketch",
    "FracSketch",
    # Comparacion
    "compare",
    "compare_all",
    # Cardinalidad
    "estimate_cardinality",
]
```

- **Efectos colaterales:** `from fastdna import *` deja de traer `pa`/`pc`.
  Ningún test del repo usa esa forma (`grep -rn "from fastdna import \*"
  python/tests/` → vacío). No es breaking change para código que importa
  por nombre.

- **Tests a añadir:** en `python/tests/test_optional_dependencies.py`, caso
  `test_star_import_exposes_only_the_public_api`.

```python
def test_star_import_exposes_only_the_public_api():
    """`import *` no debe arrastrar las dependencias del modulo.

    Antes de que `__init__.py` declarara `__all__`, `fastdna.pa` resolvia
    al modulo pyarrow entero y `from fastdna import *` lo traia al espacio
    de nombres del usuario.
    """
    namespace: dict[str, object] = {}
    exec("from fastdna import *", namespace)  # noqa: S102

    leaked = [n for n in ("pa", "pc", "os", "_core") if n in namespace]
    assert leaked == [], f"`import *` filtro nombres internos: {leaked}"

    for expected in ("count", "sketch", "compare_all", "KmerCounts"):
        assert expected in namespace, f"falta {expected} en la API publica"


def test_public_api_matches_dunder_all():
    """Todo nombre de `__all__` tiene que existir de verdad."""
    import fastdna

    missing = [n for n in fastdna.__all__ if not hasattr(fastdna, n)]
    assert missing == [], f"__all__ nombra atributos inexistentes: {missing}"
```

- **Cómo verificar:**
  ```bash
  python -m pytest python/tests/test_optional_dependencies.py -q
  python -c "import fastdna; print(hasattr(fastdna, 'pa'))"
  ```
  Salida esperada: tests en verde y `False`.

- **Esfuerzo:** 1 h (paquete + siete submódulos).
---

# Eje 4 · Mensajes de error — 2/5

---
**H-10 · Siete fallos distintos llegan a Python como el mismo `ValueError`**

- **Severidad:** alta
- **Dónde:** `src/ffi.rs:82-96` (el brazo del `match` que los agrupa)
- **Qué pasa hoy:**

```rust
            FastDnaError::MalformedFastq { .. }
            | FastDnaError::InvalidK { .. }
            | FastDnaError::MismatchedK { .. }
            | FastDnaError::MismatchedScale { .. }
            | FastDnaError::InvalidConfig { .. }
            | FastDnaError::NoSamplesFound { .. }
            | FastDnaError::Load { .. } => PyValueError::new_err(err.to_string()),
```

  No existe ninguna clase de excepción propia: el módulo `_core` no
  registra ninguna (`grep -n "create_exception\|add_class.*Error" src/ffi.rs`
  → vacío).

- **Por qué es un problema:** el usuario de Python no puede distinguir por
  tipo entre "tu archivo está corrupto" y "pasaste un `k` inválido". Caso
  real de un pipeline que procesa un lote:

  ```python
  for path in samples:
      try:
          results[path] = fastdna.count(path, k=k)
      except ValueError:
          # Quiero saltarme las muestras corruptas, pero abortar si mi
          # configuracion esta mal. Hoy no puedo distinguirlas.
          skipped.append(path)
  ```

  ```
  entrada:  k = 99 (error de configuracion) sobre 500 muestras sanas
  actual:   las 500 se marcan como "corruptas" y se saltan; el lote
            termina "correctamente" con cero resultados
  esperado: la primera muestra aborta el lote con un error de configuracion
  ```

  La única salida hoy es hacer coincidencia de subcadenas sobre el mensaje,
  que es exactamente lo que una jerarquía de excepciones existe para evitar.

- **Cómo se arregla, paso a paso:**
  1. Crear las clases de excepción en `src/ffi.rs` con
     `pyo3::create_exception!`, colgando de una base común.
  2. **Decisión cerrada sobre la base:** `FastDnaError` (Python) hereda de
     `Exception`, y las hojas heredan además del builtin que hoy se lanza
     (`MalformedFastqError(FastDnaError, ValueError)`). Así **todo el código
     existente que captura `ValueError` sigue funcionando** y el nuevo puede
     ser específico. Alternativa descartada: heredar sólo de `Exception` —
     rompería a cualquiera que capture `ValueError` hoy, sin ganar nada.
  3. Registrarlas en el módulo y reexportarlas desde
     `python/fastdna/__init__.py`.

- **Código ANTES / DESPUÉS:**

ANTES (`src/ffi.rs:66-104`), el `impl From<FastDnaError> for PyErr` completo
tal como está hoy.

DESPUÉS — se añade, antes del `impl`:
```rust
// The exception hierarchy Python callers catch on.
//
// Every leaf inherits from *two* bases: `FastDnaError`, so a caller can
// catch everything this library raises with one clause, and the builtin
// this error already mapped to before these classes existed, so every
// `except ValueError:` / `except OSError:` written against the old
// behaviour keeps working unchanged. That dual inheritance is the whole
// reason this is not a breaking change.
//
// Without these, seven genuinely different failures -- a corrupt FASTQ
// record, an out-of-range k, two sketches built with different k, two
// FracSketches built with different scale, a bad config value, an empty
// cohort directory and an unreadable sketch file -- all arrived as a bare
// `ValueError`, and the only way to tell them apart was substring
// matching on the message.
pyo3::create_exception!(_core, FastDnaError_, pyo3::exceptions::PyException);
pyo3::create_exception!(_core, MalformedFastqError, FastDnaError_, "A FASTQ/FASTA record could not be parsed.");
pyo3::create_exception!(_core, InvalidKError, FastDnaError_, "`k` is outside the 1..=32 range 2-bit packing allows.");
pyo3::create_exception!(_core, MismatchedKError, FastDnaError_, "Two sketches built with different `k` cannot be compared.");
pyo3::create_exception!(_core, MismatchedScaleError, FastDnaError_, "Two FracSketches built with different `scale` cannot be compared.");
pyo3::create_exception!(_core, InvalidConfigError, FastDnaError_, "A caller-supplied configuration value is not usable.");
pyo3::create_exception!(_core, NoSamplesFoundError, FastDnaError_, "A cohort directory held no recognizable FASTQ files.");
pyo3::create_exception!(_core, LoadError, FastDnaError_, "A saved artifact could not be read back.");
pyo3::create_exception!(_core, ExportError, FastDnaError_, "Serialization or writer failure while exporting.");
```

y el `match` pasa a:
```rust
impl From<FastDnaError> for PyErr {
    fn from(err: FastDnaError) -> PyErr {
        let message = err.to_string();
        match &err {
            FastDnaError::Io { source, .. } => {
                if source.kind() == std::io::ErrorKind::NotFound {
                    PyFileNotFoundError::new_err(message)
                } else {
                    PyOSError::new_err(message)
                }
            }
            FastDnaError::MalformedFastq { .. } => MalformedFastqError::new_err(message),
            FastDnaError::InvalidK { .. } => InvalidKError::new_err(message),
            FastDnaError::MismatchedK { .. } => MismatchedKError::new_err(message),
            FastDnaError::MismatchedScale { .. } => MismatchedScaleError::new_err(message),
            FastDnaError::InvalidConfig { .. } => InvalidConfigError::new_err(message),
            FastDnaError::NoSamplesFound { .. } => NoSamplesFoundError::new_err(message),
            FastDnaError::Load { .. } => LoadError::new_err(message),
            FastDnaError::MatrixTooLarge { .. } | FastDnaError::VocabTooLarge { .. } => {
                PyMemoryError::new_err(message)
            }
            FastDnaError::Export { .. } => ExportError::new_err(message),
            FastDnaError::Cancelled => PyKeyboardInterrupt::new_err(message),
            FastDnaError::Internal { .. } => PyRuntimeError::new_err(message),
        }
    }
}
```

En el `#[pymodule]` (`src/ffi.rs:1550` está el `m.add("__version__", ...)`),
se añade junto a él:
```rust
    // Registradas en el modulo para que `except fastdna.MalformedFastqError`
    // resuelva. El nombre publico es `FastDnaError`; el simbolo Rust lleva
    // guion bajo final para no chocar con `crate::error::FastDnaError`.
    m.add("FastDnaError", py.get_type_bound::<FastDnaError_>())?;
    m.add("MalformedFastqError", py.get_type_bound::<MalformedFastqError>())?;
    m.add("InvalidKError", py.get_type_bound::<InvalidKError>())?;
    m.add("MismatchedKError", py.get_type_bound::<MismatchedKError>())?;
    m.add("MismatchedScaleError", py.get_type_bound::<MismatchedScaleError>())?;
    m.add("InvalidConfigError", py.get_type_bound::<InvalidConfigError>())?;
    m.add("NoSamplesFoundError", py.get_type_bound::<NoSamplesFoundError>())?;
    m.add("LoadError", py.get_type_bound::<LoadError>())?;
    m.add("ExportError", py.get_type_bound::<ExportError>())?;
```

> **Nota de implementación para el ejecutor:** `pyo3::create_exception!` con
> herencia doble no está soportada por la macro en pyo3 0.22. La forma
> soportada es crear la base con la macro y las hojas con
> `PyErr::new_type_bound` en el `#[pymodule]`, pasando una tupla de bases.
> La tarea `07` lleva el código exacto y compilable de las dos variantes;
> si la macro falla al compilar, se usa la segunda sin más decisiones.

- **Efectos colaterales:** no es breaking change gracias a la herencia
  doble. `python/fastdna/__init__.py` reexporta los nombres y los añade a
  `__all__` (H-09). Los tests existentes que hacen
  `pytest.raises(ValueError)` siguen pasando.

- **Tests a añadir:** archivo nuevo `python/tests/test_exceptions.py`,
  caso `test_distinct_failures_have_distinct_exception_types`.

```python
"""La jerarquia de excepciones, comprobada por el sintoma que la motiva.

Antes de esto, un `k` invalido y un FASTQ corrupto eran ambos `ValueError`
y un pipeline por lotes no podia saltarse lo segundo sin saltarse tambien
lo primero.
"""
from __future__ import annotations

import pytest

import fastdna


def test_invalid_k_and_malformed_input_are_different_types(tmp_path):
    good = tmp_path / "good.fastq"
    good.write_text("@r0\nACGTACGTACGT\n+\nIIIIIIIIIIII\n")

    bad = tmp_path / "bad.fastq"
    bad.write_text("not a fastq at all\n")

    with pytest.raises(fastdna.InvalidKError):
        fastdna.count(str(good), k=99)

    with pytest.raises(fastdna.MalformedFastqError):
        fastdna.count(str(bad), k=5)

    # El sintoma exacto: distinguirlas por tipo.
    assert not issubclass(fastdna.InvalidKError, fastdna.MalformedFastqError)
    assert not issubclass(fastdna.MalformedFastqError, fastdna.InvalidKError)


def test_every_library_error_shares_one_catchable_base(tmp_path):
    bad = tmp_path / "bad.fastq"
    bad.write_text("not a fastq at all\n")
    with pytest.raises(fastdna.FastDnaError):
        fastdna.count(str(bad), k=5)


def test_old_valueerror_handlers_keep_working(tmp_path):
    """La herencia doble es lo que hace este cambio no-breaking."""
    good = tmp_path / "good.fastq"
    good.write_text("@r0\nACGTACGTACGT\n+\nIIIIIIIIIIII\n")
    with pytest.raises(ValueError):
        fastdna.count(str(good), k=99)


def test_missing_file_is_still_filenotfounderror():
    with pytest.raises(FileNotFoundError):
        fastdna.count("/no/such/file.fastq", k=21)
```

- **Cómo verificar:**
  ```bash
  maturin develop --features python
  python -m pytest python/tests/test_exceptions.py -q
  ```
  Salida esperada: `4 passed`.

- **Esfuerzo:** 3 h.
---

---
**H-11 · `FastDnaError::Export` y `::Load` destruyen el error original, que es justo lo que el propio módulo critica**

- **Severidad:** media
- **Dónde:** `src/error.rs:32` (`Export { path, reason: String }`), `src/error.rs:39` (`Load { path, reason: String }`), `src/error.rs:105-111` (`impl Error::source`), `src/export.rs:27-32` (`export_err`)
- **Qué pasa hoy:** `source()` sólo devuelve algo para `Io`:

```rust
impl std::error::Error for FastDnaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FastDnaError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}
```

  y `export_err` convierte el error de Arrow/Parquet a texto:

```rust
fn export_err<E: std::fmt::Display>(path: &Path, err: E) -> FastDnaError {
    FastDnaError::Export {
        path: path.to_path_buf(),
        reason: err.to_string(),
    }
}
```

- **Por qué es un problema:** el propio archivo argumenta lo contrario para
  el caso `Io`. `src/export.rs:21-26` dice, sobre no usar `Io` para errores
  de Arrow: *"laundering them through `FastDnaError::Io` with a synthetic
  `std::io::Error` **erases their real type and misleads callers**"*. `Export`
  hace exactamente eso con el error de Parquet: lo aplana a `String`.

  ```
  entrada:  un consumidor Rust escribe
              match err.source() {
                  Some(e) if e.downcast_ref::<parquet::errors::ParquetError>().is_some()
                      => reintentar con otra compresion,
                  _ => propagar,
              }
  actual:   `source()` devuelve None para Export; el downcast nunca ocurre
            y no hay forma de programar contra la causa
  esperado: Some(&ParquetError)
  ```

  El mismo razonamiento vale para `Load`, que envuelve fallos de
  `serde_json`.

- **Cómo se arregla, paso a paso:**
  1. Añadir un campo `source: Option<Box<dyn Error + Send + Sync>>` a
     `Export` y `Load`.
  2. **Decisión cerrada:** `Option`, no obligatorio. Hay sitios que
     construyen `Export`/`Load` a partir de una condición propia sin error
     subyacente (por ejemplo `validate_sketch_invariants`); forzar un
     `source` obligaría a inventar uno.
  3. Devolverlo desde `source()`.
  4. Mantener `reason` — es lo que va al mensaje y lo que ve el usuario de
     Python; sólo se añade la cadena tipada por debajo.

- **Código ANTES / DESPUÉS:**

ANTES (`src/error.rs:30-40`):
```rust
    /// Serialization or writer failure while exporting results.
    Export { path: PathBuf, reason: String },
    /// A file that was supposed to be a previously-saved artifact (e.g. a
    /// `GenomeSketch` written by `save`) could not be read back -- corrupt
    /// JSON, a foreign file, or data that fails the loader's own
    /// consistency checks. Deliberately distinct from `Export`: an error
    /// while reading must never claim to be an error while writing, which
    /// is what reusing `Export` for both would tell a caller.
    Load { path: PathBuf, reason: String },
```

DESPUÉS:
```rust
    /// Serialization or writer failure while exporting results.
    ///
    /// `source` carries the original `ParquetError`/`ArrowError` when there
    /// was one. This module already argues, at `export.rs`'s `export_err`,
    /// that laundering a typed error through a synthetic one "erases their
    /// real type and misleads callers" -- flattening the cause into
    /// `reason: String` and returning `None` from `source()` did the same
    /// thing one level further down. `Option` rather than mandatory
    /// because some construction sites (a loader's own consistency check,
    /// for one) have no underlying error to carry and should not have to
    /// invent one.
    Export {
        path: PathBuf,
        reason: String,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    /// A file that was supposed to be a previously-saved artifact (e.g. a
    /// `GenomeSketch` written by `save`) could not be read back -- corrupt
    /// JSON, a foreign file, or data that fails the loader's own
    /// consistency checks. Deliberately distinct from `Export`: an error
    /// while reading must never claim to be an error while writing, which
    /// is what reusing `Export` for both would tell a caller.
    ///
    /// Same `source` contract as `Export` above.
    Load {
        path: PathBuf,
        reason: String,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
```

ANTES (`src/error.rs:105-111`):
```rust
impl std::error::Error for FastDnaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FastDnaError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}
```

DESPUÉS:
```rust
impl std::error::Error for FastDnaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FastDnaError::Io { source, .. } => Some(source),
            FastDnaError::Export { source, .. } | FastDnaError::Load { source, .. } => {
                source.as_ref().map(|e| &**e as &(dyn std::error::Error + 'static))
            }
            _ => None,
        }
    }
}
```

ANTES (`src/export.rs:27-32`):
```rust
fn export_err<E: std::fmt::Display>(path: &Path, err: E) -> FastDnaError {
    FastDnaError::Export {
        path: path.to_path_buf(),
        reason: err.to_string(),
    }
}
```

DESPUÉS:
```rust
fn export_err<E>(path: &Path, err: E) -> FastDnaError
where
    E: std::error::Error + Send + Sync + 'static,
{
    FastDnaError::Export {
        path: path.to_path_buf(),
        reason: err.to_string(),
        source: Some(Box::new(err)),
    }
}
```

- **Efectos colaterales:** **breaking change** en la construcción por
  literal de esas dos variantes. Sitios a actualizar (verificados con
  `grep -rn "FastDnaError::Export {\|FastDnaError::Load {" src/`):
  `src/export.rs:27`, `src/sketch.rs` (rutas `save`/`load`),
  `src/metagenomics.rs` (`save`/`load` de la base de datos),
  `src/qc.rs` (`export_json`). En cada uno se añade `source: None` si no hay
  causa tipada, o `Some(Box::new(e))` si la hay. El cambio de firma de
  `export_err` de `Display` a `Error` obliga a que las llamadas pasen un
  error real; todas lo hacen ya.

- **Tests a añadir:** en `src/error.rs`, dentro de `mod tests`, caso
  `export_and_load_expose_their_underlying_cause`.

```rust
    /// `Io` ya exponia su causa; `Export` y `Load` la aplanaban a texto y
    /// devolvian `None`, que es el mismo borrado de tipo que el comentario
    /// de `export.rs::export_err` reprocha a envolver Arrow en `Io`.
    #[test]
    fn export_and_load_expose_their_underlying_cause() {
        use std::error::Error;

        let cause = std::io::Error::new(std::io::ErrorKind::Other, "el escritor de parquet fallo");
        let err = FastDnaError::Export {
            path: PathBuf::from("counts.parquet"),
            reason: cause.to_string(),
            source: Some(Box::new(cause)),
        };

        let source = err.source().expect("Export debe exponer su causa");
        assert!(source.to_string().contains("el escritor de parquet fallo"));
        assert!(
            source.downcast_ref::<std::io::Error>().is_some(),
            "la causa debe seguir siendo del tipo original, no una cadena"
        );
        assert!(err.to_string().contains("counts.parquet"));
    }

    /// El campo es opcional a proposito: hay sitios que construyen estas
    /// variantes desde una comprobacion propia sin error subyacente.
    #[test]
    fn export_without_a_cause_reports_no_source_rather_than_a_synthetic_one() {
        use std::error::Error;

        let err = FastDnaError::Load {
            path: PathBuf::from("sample.sig"),
            reason: "sketch_size no coincide con el numero de hashes".to_string(),
            source: None,
        };
        assert!(err.source().is_none());
        assert!(err.to_string().contains("sample.sig"));
    }
```

- **Cómo verificar:**
  ```bash
  cargo test --lib error::
  ```
  Salida esperada: `6 passed` (los 4 existentes más 2).

- **Esfuerzo:** 2 h.
---

---
**H-12 · Errores de librería citan flags de CLI que el usuario de Python no tiene**

- **Severidad:** media
- **Dónde:** `src/error.rs:72-75` (`MatrixTooLarge`), `src/error.rs:76-80` (`VocabTooLarge`)
- **Qué pasa hoy:**

```rust
            FastDnaError::MatrixTooLarge { estimated_bytes, limit } => write!(
                f,
                "dense matrix would need {estimated_bytes} bytes, over the {limit} byte limit; \
                 lower --top-features or use --format sparse"
            ),
            FastDnaError::VocabTooLarge { estimated_bytes, limit } => write!(
                f,
                "vocabulary table would need {estimated_bytes} bytes, over the {limit} byte limit; \
                 raise --min-count or use --approx-vocab"
            ),
```

- **Por qué es un problema:** esos cuatro flags **no existen en el CLI**
  (`grep -n "top-features\|approx-vocab\|format sparse" src/cli.rs` → vacío;
  el CLI tiene 17 flags y ninguno es esos). Y el error llega a Python vía
  `src/ffi.rs:97-99` como `PyMemoryError`:

  ```
  entrada:  fastdna.multiomics.kmer_feature_table(paths, top_features=100000)
  actual:   MemoryError: dense matrix would need 8000000000 bytes, over the
            4000000000 byte limit; lower --top-features or use --format sparse
  esperado: ...; lower `top_features` or ask for a sparse result
  ```

  El usuario busca `--top-features` en `fastdna --help`, no lo encuentra, y
  el mensaje que debía guiarle le hace perder el tiempo. El parámetro real
  en Python se llama `top_features` (guion bajo), y `--format sparse` no
  existe en ninguna parte.

- **Cómo se arregla, paso a paso:**
  1. Reescribir los dos mensajes en términos del **concepto**, no de la
     sintaxis de una interfaz concreta.
  2. **Decisión cerrada:** no se parametriza el mensaje por interfaz (no se
     pasa un "soy CLI/soy Python"). Sería complejidad para un texto de
     ayuda. Se nombra el parámetro en la forma neutra `top_features`, que
     es como se llama en la API Python y como se llamaría un futuro flag.

- **Código ANTES / DESPUÉS:**

ANTES: ver bloque de arriba (`src/error.rs:72-80`).

DESPUÉS:
```rust
            // Los dos mensajes nombran el *parametro*, no la sintaxis de
            // una interfaz. Antes decian "lower --top-features or use
            // --format sparse" y "raise --min-count or use --approx-vocab":
            // ninguno de esos cuatro flags existe en `cli.rs`, y estos
            // errores llegan a Python como `MemoryError`, donde el
            // parametro se llama `top_features` y no hay flags en absoluto.
            FastDnaError::MatrixTooLarge { estimated_bytes, limit } => write!(
                f,
                "dense matrix would need {estimated_bytes} bytes, over the {limit} byte limit; \
                 lower `top_features` to keep fewer k-mer columns, or raise the byte limit"
            ),
            FastDnaError::VocabTooLarge { estimated_bytes, limit } => write!(
                f,
                "vocabulary table would need {estimated_bytes} bytes, over the {limit} byte limit; \
                 raise `min_count` to drop rare k-mers, or raise the byte limit"
            ),
```

- **Efectos colaterales:** `src/error.rs:145-150`
  (`matrix_too_large_reports_both_numbers`) sólo comprueba que aparezcan las
  dos cifras, así que sigue pasando. Ningún test busca los nombres de flag.

- **Tests a añadir:** en `src/error.rs`, `mod tests`, caso
  `size_limit_messages_do_not_name_flags_that_do_not_exist`.

```rust
    /// Estos mensajes llegan tanto al CLI como a Python (como
    /// `MemoryError`). Nombraban cuatro flags -- `--top-features`,
    /// `--format sparse`, `--min-count` como remedio de vocabulario y
    /// `--approx-vocab` -- de los que solo `--min-count` existe, y ninguno
    /// significa nada para quien llama desde Python.
    #[test]
    fn size_limit_messages_do_not_name_flags_that_do_not_exist() {
        let messages = [
            FastDnaError::MatrixTooLarge { estimated_bytes: 8_000_000_000, limit: 4_000_000_000 }
                .to_string(),
            FastDnaError::VocabTooLarge { estimated_bytes: 8_000_000_000, limit: 4_000_000_000 }
                .to_string(),
        ];

        for msg in &messages {
            for ghost in ["--top-features", "--format sparse", "--approx-vocab"] {
                assert!(
                    !msg.contains(ghost),
                    "el mensaje nombra {ghost}, que no existe en cli.rs: {msg}"
                );
            }
        }

        assert!(messages[0].contains("top_features"), "{}", messages[0]);
        assert!(messages[1].contains("min_count"), "{}", messages[1]);
    }
```

- **Cómo verificar:**
  ```bash
  cargo test --lib error::size_limit_messages
  ```
  Salida esperada: `1 passed`.

- **Esfuerzo:** 30 min.
---

---
**H-13 · `sort_by()` con una columna inexistente no dice cuáles son válidas**

- **Severidad:** baja
- **Dónde:** `python/fastdna/__init__.py:171-177`
- **Qué pasa hoy:**

```python
    def sort_by(self, column="frequency", *, descending=True):
        order = "descending" if descending else "ascending"
        return KmerCounts(self._raw, self.table.sort_by([(column, order)]))
```

- **Por qué es un problema:** medido sobre tablas de 5, 100 y 100 000
  filas — el mensaje **no** crece con el tamaño (pyarrow trunca su repr:
  250, 550 y 580 caracteres respectivamente), así que no es el desastre que
  parece a primera vista; es un problema de claridad, no de volumen.

  ```
  entrada:  counts.sort_by("freq")     # el nombre real es "frequency"
  actual:   ArrowInvalid: Invalid sort key column: No match for
            FieldRef.Name(freq) in kmer_u64: uint64
            kmer_sequence: string
            frequency: uint32
            ----
            kmer_u64:
              [ [ 0, 1, 2, 3, 4 ] ]
            ... (un volcado de las primeras filas de cada columna)
  esperado: ValueError: unknown column 'freq'; sort_by accepts one of
            ['kmer_u64', 'kmer_sequence', 'frequency']
  ```

  El mensaje actual es de pyarrow, menciona `FieldRef` (un concepto interno
  de Arrow que el usuario de FastDNA no conoce) y entierra los nombres
  válidos en un volcado de datos.

- **Cómo se arregla, paso a paso:**
  1. Comprobar `column` contra `self.table.column_names` antes de llamar a
     `sort_by`.
  2. **Decisión cerrada:** `ValueError`, no `KeyError`. El valor es de tipo
     correcto (`str`) pero fuera del dominio permitido, que es lo que
     `ValueError` significa; además `filter` ya usa `ValueError` de facto vía
     pyarrow para argumentos malos.

- **Código ANTES / DESPUÉS:**

ANTES (`python/fastdna/__init__.py:171-177`):
```python
    def sort_by(self, column="frequency", *, descending=True):
        """Returns a new `KmerCounts` with the current view sorted by
        `column` (any of `table.column_names`; `frequency` by default,
        matching what "the most/least common k-mers" means in practice).
        """
        order = "descending" if descending else "ascending"
        return KmerCounts(self._raw, self.table.sort_by([(column, order)]))
```

DESPUÉS:
```python
    def sort_by(self, column: str = "frequency", *, descending: bool = True) -> KmerCounts:
        """Returns a new `KmerCounts` with the current view sorted by
        `column` (any of `table.column_names`; `frequency` by default,
        matching what "the most/least common k-mers" means in practice).

        The name is checked here rather than left to pyarrow, whose own
        message for an unknown key talks about `FieldRef` -- an Arrow
        concept a FastDNA caller has no reason to know -- and buries the
        valid column names inside a dump of the table's first rows.
        """
        available = self.table.column_names
        if column not in available:
            raise ValueError(
                f"unknown column {column!r}; sort_by accepts one of {available}"
            )
        order = "descending" if descending else "ascending"
        return KmerCounts(self._raw, self.table.sort_by([(column, order)]))
```

- **Efectos colaterales:** el tipo de excepción cambia de `ArrowInvalid` a
  `ValueError`. `ArrowInvalid` **es** subclase de `ValueError` en pyarrow, así
  que cualquier `except ValueError` existente sigue capturándolo; sólo se
  rompería un `except pa.ArrowInvalid` explícito, que no aparece en el repo
  (`grep -rn "ArrowInvalid" python/` → vacío).

- **Tests a añadir:** en `python/tests/test_fluent_api.py`, caso
  `test_sort_by_unknown_column_lists_the_valid_ones`.

```python
def test_sort_by_unknown_column_lists_the_valid_ones(tmp_path):
    import pytest

    path = tmp_path / "s.fastq"
    path.write_text("@r0\nACGTACGTACGT\n+\nIIIIIIIIIIII\n")
    counts = fastdna.count(str(path), k=5)

    with pytest.raises(ValueError) as excinfo:
        counts.sort_by("freq")          # el nombre real es "frequency"

    message = str(excinfo.value)
    assert "freq" in message
    assert "frequency" in message, "el mensaje debe nombrar las columnas validas"
    assert "FieldRef" not in message, "no debe filtrarse el vocabulario interno de Arrow"
```

- **Cómo verificar:**
  ```bash
  python -m pytest python/tests/test_fluent_api.py -q -k sort_by_unknown
  ```
  Salida esperada: `1 passed`.

- **Esfuerzo:** 20 min.
---

# Eje 5 · Corrección y edge cases — 4/5

Nota alta merecida: la suite es de 496 tests en Rust y 674 en Python, todos
en verde, con `tests/dual_strategy.rs` exigiendo acuerdo bit a bit entre
estrategias. Los dos hallazgos son reales pero acotados.

---
**H-14 · `metagenomics.rs` ejecuta una copia lenta del extractor de k-mers, con un TODO que ya caducó**

- **Severidad:** media
- **Dónde:** `src/metagenomics.rs:669-674` (el TODO), `src/metagenomics.rs:675-699` (la copia), llamada en `src/metagenomics.rs:826` (construcción de base de datos) y `src/metagenomics.rs:1259` (clasificación); el test que las fija en `src/metagenomics.rs:1838-1861`
- **Qué pasa hoy:**

```rust
/// TODO: delete this and call `kmer::extract_canonical_kmers_into` once
/// that lands in `src/kmer.rs` -- it is being added concurrently in the
/// main tree, and duplicating it here rather than adding a second copy to
/// `kmer.rs` is what keeps that merge trivial.
/// `the_local_kmer_extractor_matches_the_shared_one` pins the two together
/// so this copy cannot drift in the meantime.
fn extract_canonical_kmers_into(seq: &[u8], k: usize, out: &mut Vec<u64>) {
    out.clear();
    if seq.len() < k || k == 0 || k > 32 {
        return;
    }

    let mask = if k == 32 { u64::MAX } else { (1u64 << (2 * k)) - 1 };
    let mut current_kmer: u64 = 0;
    let mut valid_len = 0;

    for &base in seq {
        if let Some(bits) = crate::kmer::base_to_bits(base) {
            current_kmer = ((current_kmer << 2) | bits) & mask;
            valid_len += 1;
            if valid_len >= k {
                out.push(crate::kmer::canonical_kmer_u64(current_kmer, k));
            }
        } else {
            current_kmer = 0;
            valid_len = 0;
        }
    }
}
```

  **La función que el TODO espera ya existe**: `src/kmer.rs:171`
  (`pub fn extract_canonical_kmers_into(seq: &[u8], k: usize, out: &mut Vec<u64>)`),
  con firma idéntica, desde el commit `106ae78`.

- **Por qué es un problema:** no es sólo código duplicado. La copia local
  llama a `canonical_kmer_u64` **una vez por k-mer** (línea 690), y esa
  función recalcula el complemento reverso entero: `!kmer`, dos pasos de
  máscara-y-desplazamiento, `swap_bytes` y un desplazamiento final — 13
  operaciones ALU por k-mer, según la aritmética que `src/kmer.rs:140-152`
  documenta. La versión compartida hace rodar el complemento reverso junto
  al directo: **4 operaciones por base**, ~9 operaciones menos por k-mer.

  ```
  entrada:  fastdna.metagenomics.build_database(refs, k=31) sobre una
            referencia de 100 Mbase
  actual:   ~1e8 k-mers x 9 operaciones ALU de mas = ~9e8 operaciones
            evitables, en el camino que el propio comentario describe como
            "the builder and the classifier call it once per record"
  esperado: el mismo resultado con la ruta ya optimizada del crate
  ```

  Los resultados son idénticos —lo garantiza
  `the_local_kmer_extractor_matches_the_shared_one`— así que esto es
  puramente trabajo desperdiciado, más 25 líneas de deuda que el propio
  autor pidió borrar.

- **Cómo se arregla, paso a paso:**
  1. Borrar `src/metagenomics.rs:669-699` (el comentario TODO y la función).
  2. Añadir `use crate::kmer::extract_canonical_kmers_into;` a los imports
     del módulo. Las dos llamadas (`:826`, `:1259`) quedan intactas: el
     nombre y la firma coinciden exactamente.
  3. Borrar el test `the_local_kmer_extractor_matches_the_shared_one`
     (`src/metagenomics.rs:1834-1861`): fija una duplicación que deja de
     existir, y compararía la función compartida consigo misma.

- **Código ANTES / DESPUÉS:**

ANTES (`src/metagenomics.rs:660-699`): el bloque completo pegado arriba,
precedido de su comentario de documentación.

DESPUÉS: las 40 líneas desaparecen. En su lugar, en los imports del módulo
(cabecera de `src/metagenomics.rs`), se añade:

```rust
// La copia local de este extractor se borro: `kmer.rs` ya expone la version
// compartida, y la copia llamaba a `canonical_kmer_u64` una vez por k-mer,
// recalculando el complemento reverso entero (13 operaciones ALU) donde la
// compartida lo hace rodar junto al directo (4 operaciones por base). El
// TODO que pedia este borrado estaba en la propia copia.
use crate::kmer::extract_canonical_kmers_into;
```

ANTES (`src/metagenomics.rs:1834-1861`): el test
`the_local_kmer_extractor_matches_the_shared_one` completo.

DESPUÉS: borrado. En su lugar no se añade nada — la corrección del
extractor la cubren los 25 tests de `src/kmer.rs`, incluido
`rolled_reverse_complement_matches_recomputed_at_every_position`, que
compara contra una implementación de referencia en todas las posiciones
para todo `k` en `1..=32`.

- **Efectos colaterales:** ninguno observable. La salida es idéntica por
  construcción (el test borrado lo demostraba). Los tests de
  `src/metagenomics.rs` que dependen de conteos concretos
  (`the_toy_reference_blocks_share_no_kmers` y los demás del módulo) siguen
  pasando sin cambios. No es breaking change: la función era privada
  (`fn`, no `pub fn`).

- **Tests a añadir:** en `src/metagenomics.rs`, `mod tests`, caso
  `the_module_uses_the_shared_extractor`.

```rust
    /// Sustituye a `the_local_kmer_extractor_matches_the_shared_one`, que
    /// fijaba una duplicacion ya borrada. Lo que queda por comprobar no es
    /// que dos implementaciones coincidan -- solo hay una -- sino que este
    /// modulo sigue produciendo el mismo conjunto de k-mers que el resto
    /// del crate para las entradas que le llegan de verdad, incluidas las
    /// que contienen bases ambiguas.
    #[test]
    fn the_module_uses_the_shared_extractor() {
        let cases: [&[u8]; 6] = [
            b"",
            b"ACGTACGTACGTACGT",
            b"ACGTNACGTNACGTACGTAC",
            b"NNNNNNNNNNNN",
            b"acgtacgtacgtacgt",
            b"TTTTTTTTTTTTTTTTTTTT",
        ];
        let mut buffer = Vec::new();
        for seq in cases {
            for k in [1usize, 4, 11, 31, 32] {
                extract_canonical_kmers_into(seq, k, &mut buffer);
                assert_eq!(
                    buffer,
                    crate::kmer::extract_canonical_kmers(seq, k),
                    "k={k} sobre {:?}",
                    String::from_utf8_lossy(seq)
                );
            }
        }
    }
```

- **Cómo verificar:**
  ```bash
  cargo test --lib metagenomics::
  grep -c "fn extract_canonical_kmers_into" src/metagenomics.rs
  ```
  Salida esperada: los tests de metagenomics en verde, y `0` en el grep
  (hoy imprime `1`).

- **Esfuerzo:** 30 min.
---

---
**H-15 · `filter(min_count=-3)` es un no-op silencioso**

- **Severidad:** baja
- **Dónde:** `python/fastdna/__init__.py:148-170`
- **Qué pasa hoy:** `filter` pasa los umbrales a `pyarrow.compute` sin
  validarlos:

```python
        freq = self.table.column("frequency")
        mask = None
        if min_count is not None:
            mask = pc.greater_equal(freq, min_count)
```

- **Por qué es un problema:** medido sobre una tabla de 5 filas:

  | Entrada | Salida actual | Salida esperada |
  |---|---|---|
  | `filter(min_count=-3)` | `len() == 5` (todas) | `ValueError: min_count must be >= 0` |
  | `filter(min_count=10, max_count=1)` | `len() == 0` | `len() == 0` (correcto: banda vacía) |
  | `filter(min_count="x")` | `ArrowNotImplementedError` | `TypeError` |

  Las frecuencias son `uint32`, así que ningún umbral negativo puede
  excluir nada: `filter(min_count=-3)` **siempre** devuelve todo. Un
  `min_count = suggested - 5` que se vuelva negativo produce un filtrado
  que aparenta funcionar y no filtra nada. Es el mismo patrón que H-06 pero
  menos grave, porque el resultado (todo) es menos destructivo que el de
  `top(None)` y porque el rango invertido sí se comporta bien.

  Nótese que el CLI **sí** valida esto: `src/cli.rs:129` usa
  `clap::value_parser!(u32).range(1..)` para `--min-count`, y
  `Cli::validate` (`src/cli.rs:252-259`) rechaza `--max-count < --min-count`.
  La ruta Python no tiene ninguna de las dos comprobaciones. Es una
  incoherencia entre las dos interfaces del mismo producto.

- **Cómo se arregla, paso a paso:**
  1. Validar tipo y signo de ambos umbrales al principio de `filter`.
  2. **Decisión cerrada:** el rango invertido (`min_count > max_count`)
     **no** se rechaza en Python, a diferencia del CLI. Razón: en el CLI
     esa combinación sólo puede ser un error de tecleo, mientras que en
     Python es un resultado legítimo de aritmética sobre datos (`filter(lo,
     hi)` con `lo`/`hi` calculados), y devolver una vista vacía es la
     respuesta correcta a "ningún k-mer está en esta banda". Se documenta.
  3. Rechazar `bool` por la misma razón que en H-06.

- **Código ANTES / DESPUÉS:**

ANTES (`python/fastdna/__init__.py:148-170`):
```python
    def filter(self, min_count=None, max_count=None):
        """Returns a new `KmerCounts` restricted to k-mers whose frequency
        falls in `[min_count, max_count]` (either bound optional, both
        inclusive) -- applied on top of whatever view this one already
        holds, so `.filter(min_count=5).filter(max_count=100)` composes
        rather than the second call replacing the first.

        This is a *view* over already-counted data, not a re-count: it
        cannot recover k-mers `count()`'s own `min_count`/`max_count`
        already dropped during counting. Use it to explore a single
        `count()` result at several thresholds without re-reading the
        FASTQ file for each one.
        """
        if min_count is None and max_count is None:
            return KmerCounts(self._raw, self.table)

        freq = self.table.column("frequency")
        mask = None
        if min_count is not None:
            mask = pc.greater_equal(freq, min_count)
        if max_count is not None:
            upper = pc.less_equal(freq, max_count)
            mask = upper if mask is None else pc.and_(mask, upper)
        return KmerCounts(self._raw, self.table.filter(mask))
```

DESPUÉS:
```python
    def filter(
        self, min_count: int | None = None, max_count: int | None = None
    ) -> KmerCounts:
        """Returns a new `KmerCounts` restricted to k-mers whose frequency
        falls in `[min_count, max_count]` (either bound optional, both
        inclusive) -- applied on top of whatever view this one already
        holds, so `.filter(min_count=5).filter(max_count=100)` composes
        rather than the second call replacing the first.

        This is a *view* over already-counted data, not a re-count: it
        cannot recover k-mers `count()`'s own `min_count`/`max_count`
        already dropped during counting. Use it to explore a single
        `count()` result at several thresholds without re-reading the
        FASTQ file for each one.

        A negative bound is rejected rather than applied. Frequencies are
        `uint32`, so no negative threshold can exclude anything and
        `filter(min_count=-3)` silently returned the whole table -- a
        `min_count` that went negative through arithmetic looked like a
        filter that ran and kept everything. The CLI already rejects this
        (`--min-count` is `value_parser!(u32).range(1..)`); this brings the
        Python path in line.

        An *inverted* band (`min_count > max_count`) is deliberately NOT
        rejected here, unlike in the CLI. On a command line that
        combination can only be a typo; in Python both bounds are commonly
        computed from the data, and an empty view is the correct answer to
        "no k-mer falls in this band".
        """
        for name, value in (("min_count", min_count), ("max_count", max_count)):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an int or None, got {type(value).__name__}: {value!r}"
                )
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        if min_count is None and max_count is None:
            return KmerCounts(self._raw, self.table)

        freq = self.table.column("frequency")
        mask = None
        if min_count is not None:
            mask = pc.greater_equal(freq, min_count)
        if max_count is not None:
            upper = pc.less_equal(freq, max_count)
            mask = upper if mask is None else pc.and_(mask, upper)
        return KmerCounts(self._raw, self.table.filter(mask))
```

- **Efectos colaterales:** breaking change teórico para código que pasara
  un negativo; ese código estaba roto ya (no filtraba). `ArrowNotImplementedError`
  para una cadena pasa a ser `TypeError`, que es más correcto y no es
  subclase del anterior — pero nadie captura `ArrowNotImplementedError`
  (`grep -rn "ArrowNotImplemented" python/` → vacío).

- **Tests a añadir:** en `python/tests/test_fluent_api.py`, caso
  `test_filter_rejects_negative_bounds_but_allows_an_empty_band`.

```python
def test_filter_rejects_negative_bounds_but_allows_an_empty_band(tmp_path):
    """Un umbral negativo no puede excluir nada sobre frecuencias uint32,
    asi que `filter(min_count=-3)` devolvia la tabla entera. Una banda
    invertida, en cambio, es un resultado legitimo y sigue permitida.
    """
    import pytest

    path = tmp_path / "s.fastq"
    path.write_text("".join(f"@r{i}\nACGTACGTACGT\n+\n{'I'*12}\n" for i in range(10)))
    counts = fastdna.count(str(path), k=5)
    assert len(counts) > 0

    with pytest.raises(ValueError, match="min_count must be >= 0"):
        counts.filter(min_count=-3)
    with pytest.raises(ValueError, match="max_count must be >= 0"):
        counts.filter(max_count=-1)
    with pytest.raises(TypeError, match="min_count must be an int"):
        counts.filter(min_count="5")
    with pytest.raises(TypeError, match="min_count must be an int"):
        counts.filter(min_count=True)

    # Banda invertida: vista vacia, no excepcion -- decision documentada.
    assert len(counts.filter(min_count=10 ** 6, max_count=1)) == 0

    # Y los casos normales intactos.
    assert len(counts.filter(min_count=0)) == len(counts)
```

- **Cómo verificar:**
  ```bash
  python -m pytest python/tests/test_fluent_api.py -q -k filter_rejects
  ```
  Salida esperada: `1 passed`.

- **Esfuerzo:** 30 min.
---

# Eje 6 · Performance y footprint — 4/5

El motor está medido con un rigor poco habitual (`docs/BENCHMARKS.md`,
`docs/design-minimizer-counting.md`, los comentarios de `counter.rs` sobre
el radix revertido). Lo que está descuidado son los artefactos.

---
**H-16 · El binario de release se publica sin `strip`: 1,15 MB de tabla de símbolos**

- **Severidad:** baja
- **Dónde:** `Cargo.toml:62-89` (bloque `[profile.release]`, sin `strip`)
- **Qué pasa hoy:** el perfil declara `panic`, `lto` y `codegen-units`, con
  comentarios extensos justificando cada uno, y nada sobre símbolos:

```toml
[profile.release]
panic = "unwind"
lto = "fat"
codegen-units = 1
```

- **Por qué es un problema:** medido:

  ```
  entrada:  cargo build --release && ls -l target/release/fastdna
  actual:   7 872 664 bytes
  entrada:  strip target/release/fastdna && ls -l
  actual:   6 719 296 bytes
  ```

  Son **1 153 368 bytes (14,6 %)** de información de depuración en el
  binario que se distribuiría por Bioconda. No afecta al wheel — la cdylib
  ya sale en 1 024 520 bytes — así que el impacto es sólo sobre el CLI.

- **Cómo se arregla, paso a paso:**
  1. Añadir `strip = "symbols"` a `[profile.release]`.
  2. **Decisión cerrada:** `strip = "symbols"`, no `"debuginfo"`. `"debuginfo"`
     deja la tabla de símbolos y ahorra menos; como el crate no depende de
     backtraces simbolizados para nada (los panics de worker se capturan con
     `catch_unwind` y se convierten en `FastDnaError::Internal` con su propio
     `detail`, ver `src/pipeline.rs`), no hay razón para conservarla.
  3. **Decisión cerrada:** NO se toca `lto`/`codegen-units`. `Cargo.toml:72-86`
     documenta que están sin medir a propósito y por qué; ese razonamiento
     sigue vigente y no es asunto de este hallazgo.

- **Código ANTES / DESPUÉS:**

ANTES (`Cargo.toml:87-89`):
```toml
lto = "fat"
codegen-units = 1
```

DESPUÉS:
```toml
lto = "fat"
codegen-units = 1

# Medido, no asumido: el binario pasa de 7 872 664 a 6 719 296 bytes,
# 1 153 368 bytes (14,6 %) de tabla de simbolos e informacion de
# depuracion que se distribuirian por Bioconda sin que nada los use.
# Nada de este crate depende de backtraces simbolizados: un panic de
# worker lo captura `catch_unwind` en `pipeline.rs` y se convierte en
# `FastDnaError::Internal` con su propio campo `detail`, que es lo que el
# usuario ve. Se elige "symbols" y no "debuginfo" porque el segundo
# conserva la tabla de simbolos y ahorra menos, sin comprar nada aqui.
strip = "symbols"
```

- **Efectos colaterales:** un `perf`/`gdb` sobre el binario de release deja
  de mostrar nombres de función. Para perfilar, `cargo build --profile
  release` con `strip = "none"` en la línea de comandos, o el perfil `dev`.
  No afecta a la cdylib de Python ni a los tests (perfil `test`).

- **Tests a añadir:** ninguno automatizable de forma barata: comprobar el
  tamaño del binario en un test lo haría frágil frente a versiones de
  rustc. Se verifica a mano en la tarea y queda anotado en el CHANGELOG.

- **Cómo verificar:**
  ```bash
  cargo build --release
  ls -l target/release/fastdna
  ```
  Salida esperada: alrededor de 6 719 296 bytes, no 7 872 664.

- **Esfuerzo:** 10 min.
---

---
**H-17 · El SBOM CycloneDX ocupa el 20 % del wheel y nada lo pidió**

- **Severidad:** baja
- **Dónde:** artefacto generado — `fastdna-0.1.0.dist-info/sboms/fastdna.cyclonedx.json`, 201 731 bytes de un wheel de 1 007 127; no hay configuración en `pyproject.toml` que lo active ni que lo desactive
- **Qué pasa hoy:** maturin ≥1.9 genera un SBOM CycloneDX por defecto y lo
  mete en `dist-info/sboms/`. Medido: **201 731 bytes descomprimidos**, la
  segunda entrada más grande del wheel después de la propia extensión.

- **Por qué es un problema:** matizado — **un SBOM es algo bueno**, y en un
  paquete de bioinformática que aspira a entrar en pipelines regulados,
  probablemente se quiera conservar. El hallazgo no es "bórralo", es que
  **hoy nadie ha decidido nada al respecto**: aparece porque la versión de
  maturin lo pone, no porque el proyecto lo haya elegido, no se menciona en
  ningún documento, y su tamaño (20 % del wheel) es invisible.

  ```
  entrada:  unzip -l fastdna-0.1.0-cp38-abi3-*.whl | grep sbom
  actual:   201731  fastdna-0.1.0.dist-info/sboms/fastdna.cyclonedx.json
  esperado: lo mismo, pero como decision registrada, no como accidente de
            version de la herramienta
  ```

- **Cómo se arregla, paso a paso:** **Decisión cerrada: se conserva y se
  documenta.** Un SBOM de 109 crates transitivas es exactamente lo que un
  usuario institucional pide, y 200 KB comprimidos son ~50 KB reales.
  1. Añadir un comentario en `pyproject.toml` registrando la decisión, para
     que nadie lo borre por sorpresa al ver el tamaño.
  2. Nombrarlo en el README como argumento a favor, no dejarlo escondido.
  3. Alternativa descartada: desactivarlo con `--no-sbom`. Ahorra 200 KB de
     un artefacto que ya pesa 1 MB, a cambio de perder la trazabilidad de
     dependencias que ninguna otra parte del proyecto ofrece.

- **Código ANTES / DESPUÉS:**

ANTES (`pyproject.toml:44-48`): sin mención al SBOM.

DESPUÉS (añadido al bloque `[tool.maturin]`, junto a lo de H-01 y H-04):
```toml
# maturin genera un SBOM CycloneDX en `dist-info/sboms/` por defecto desde
# la 1.9, y aqui se conserva a proposito: son 201 731 bytes descomprimidos
# -- la segunda entrada mas grande del wheel tras la propia extension --
# que describen las 109 crates del grafo de dependencias de runtime. Es la
# unica trazabilidad de dependencias que este paquete ofrece, y es lo que
# un usuario institucional pide. Se anota aqui para que el tamano no
# sorprenda a nadie y nadie lo desactive con `--no-sbom` creyendo que es
# basura de la herramienta.
```

- **Efectos colaterales:** ninguno; no cambia nada, sólo registra.

- **Tests a añadir:** en `python/tests/test_wheel_contents.py` (el archivo
  que crea H-01), caso `test_wheel_carries_an_sbom`.

```python
@pytest.mark.skipif(
    shutil.which("maturin") is None,
    reason="maturin no esta instalado; este test comprueba el empaquetado",
)
def test_wheel_carries_an_sbom(tmp_path):
    """El SBOM es una decision registrada (ver [tool.maturin] en
    pyproject.toml), no un accidente de la version de maturin. Si una
    actualizacion de la herramienta deja de generarlo, esto lo detecta en
    vez de que desaparezca en silencio.
    """
    out = tmp_path / "dist"
    subprocess.run(
        ["maturin", "build", "--features", "python", "--out", str(out)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    wheels = list(out.glob("*.whl"))
    assert wheels, "maturin no produjo ningun wheel"

    with zipfile.ZipFile(wheels[0]) as z:
        sboms = [i for i in z.infolist() if "sboms/" in i.filename]

    assert sboms, "el wheel deberia llevar un SBOM CycloneDX; ver pyproject.toml"
    assert sboms[0].file_size > 1000, "el SBOM esta vacio o truncado"
```

- **Cómo verificar:**
  ```bash
  python -m pytest python/tests/test_wheel_contents.py -q
  ```
  Salida esperada: `2 passed` (este más el de H-01).

- **Esfuerzo:** 15 min.
---

# Eje 7 · Tests — 4/5

Cobertura real medida: **Rust 93,00 % de regiones / 92,96 % de líneas**,
**Python 92,4 %** con los extras `[test]` completos. Son cifras altas y
honestas. Los dos huecos son estructurales, no de descuido.

---
**H-18 · `src/ffi.rs` — 1 570 líneas, la frontera Rust↔Python entera — no entra en ninguna medición de cobertura**

- **Severidad:** alta
- **Dónde:** `src/ffi.rs` (todo el archivo); `src/lib.rs:14-15` (`#[cfg(feature = "python")] pub mod ffi;`); `.github/workflows/wheels.yml:20-27` (el comentario que prohíbe `--features python` fuera de maturin)
- **Qué pasa hoy:** `cargo llvm-cov --all-targets` compila sin features, así
  que `ffi.rs` no se compila y **no aparece en la tabla de cobertura**:

```
Filename          Regions  Missed  Cover   ...
adaptive_bins.rs      363       8  97.80%
atomic.rs              98      11  88.78%
...
translate.rs          626      30  95.21%
-------------------------------------------
TOTAL              16 362    1 146  93.00%
```

  `ffi.rs` y `wasm.rs` sencillamente no están en la lista. El 93 % es el
  93 % de lo que se compila, no del crate.

- **Por qué es un problema:** `ffi.rs` es donde vive:
  la conversión `FastDnaError → PyErr` (H-10), el bucle de sondeo de
  `count()` con su cancelación, la construcción del `RecordBatch` de Arrow,
  el `catch_unwind` que impide que un panic cruce a Python, y el manejo del
  GIL. Es el código **más peligroso del repositorio** —un fallo aquí es
  corrupción de memoria o un cuelgue, no un conteo equivocado— y es el
  único del que no se sabe qué fracción está ejercitada.

  ```
  entrada:  cargo llvm-cov --all-targets --summary-only
  actual:   ffi.rs no figura; el TOTAL declarado (93,00 %) excluye 1 570
            lineas, un 7,5 % del arbol de src/
  esperado: una cifra que incluya ffi.rs, o una nota explicita de que no
  ```

  Sí está probado indirectamente: los 674 tests de `python/tests` lo
  atraviesan entero. Lo que falta no son tests, es **la medición**: nadie
  sabe si esos tests tocan el brazo `Cancelled`, o la rama de `MemoryError`.

- **Cómo se arregla, paso a paso:**
  1. Añadir un job a `.github/workflows/wheels.yml` (o al workflow de CI que
     crea H-26) que mida cobertura de `ffi.rs` a través de la suite Python.
  2. **Decisión cerrada sobre cómo:** `cargo llvm-cov` con la variable
     `LLVM_PROFILE_FILE` puesta, construyendo la extensión con
     `maturin develop` bajo instrumentación y ejecutando `pytest`. **No** se
     intenta `cargo test --features python`: el comentario de
     `wheels.yml:20-27` explica que eso falla al enlazar en Linux/macOS por
     `extension-module`, y ese análisis es correcto.
  3. Mientras eso no exista, **declarar el hueco en el propio informe de
     cobertura** en vez de dejar que el 93 % se lea como si cubriera todo.
     Este segundo paso es el que entra en la tarea inmediata; el job de CI
     es la tarea `18`.

- **Código ANTES / DESPUÉS:**

ANTES: no existe ningún script de cobertura; se invoca a mano.

DESPUÉS — archivo nuevo `scripts/coverage.sh`:
```bash
#!/usr/bin/env bash
# Cobertura de las dos mitades del crate, medidas por separado porque no
# se pueden medir juntas.
#
# `cargo llvm-cov --all-targets` compila sin features, asi que NO ve
# `src/ffi.rs` (1 570 lineas) ni `src/wasm.rs`. El total que imprime es el
# total de lo que compila, no del arbol, y leerlo como si cubriera todo
# oculta justo el archivo mas peligroso del repositorio: la frontera con
# Python, donde un fallo es corrupcion de memoria y no un conteo malo.
#
# `cargo test --features python` no es una salida: `pyo3/extension-module`
# suprime el enlace con libpython en Linux y macOS, y cualquier target que
# enlace la rlib y arrastre `ffi` falla con simbolos `Py_*` sin resolver.
# Es el mismo analisis que documenta `.github/workflows/wheels.yml`. La via
# que si funciona es instrumentar la extension que maturin construye y
# ejercitarla desde pytest, que es lo que hace la segunda mitad.
set -euo pipefail

echo "=== 1/2  Nucleo Rust (sin features) ==="
cargo llvm-cov --all-targets --summary-only

echo
echo "=== 2/2  Frontera FFI, a traves de la suite de Python ==="
source <(cargo llvm-cov show-env --export-prefix)
cargo llvm-cov clean --workspace
maturin develop --features python
python -m pytest python/tests -q
cargo llvm-cov report --summary-only --ignore-filename-regex '^(?!.*ffi\.rs)'

echo
echo "NOTA: src/wasm.rs (84 lineas) no tiene ningun test y no aparece en"
echo "      ninguna de las dos mitades. Ver H-19."
```

- **Efectos colaterales:** ninguno en el código de producción. El script es
  nuevo; `scripts/` ya existe (contiene `scripts/bench`).

- **Tests a añadir:** no aplica — esto *es* infraestructura de test. El
  criterio de aceptación es que el script corra y emita las dos tablas.

- **Cómo verificar:**
  ```bash
  bash scripts/coverage.sh 2>&1 | grep -c "ffi.rs"
  ```
  Salida esperada: al menos `1` (hoy: `0`, porque `ffi.rs` no aparece nunca).

- **Esfuerzo:** 3 h (el script y hacer que la instrumentación funcione bajo
  maturin).
---

---
**H-19 · `src/wasm.rs` no tiene un solo test, y es API pública**

- **Severidad:** media
- **Dónde:** `src/wasm.rs` (84 líneas, `grep -c "mod tests"` → `0`); `src/lib.rs:28` (`pub mod wasm;`)
- **Qué pasa hoy:** el módulo compila —lo verifiqué en ambos targets:

  ```
  cargo check --features wasm                                    -> exit 0
  cargo check --features wasm --target wasm32-unknown-unknown    -> exit 0
  ```

  y no tiene ninguna cobertura ni ningún test. Ningún archivo de `tests/`
  lo menciona (`grep -rn "wasm" tests/` → vacío).

- **Por qué es un problema:** es el único módulo del crate en esa
  situación. `cms.rs` está muerto pero **sí tiene 13 tests** (96,34 % de
  cobertura); `wasm.rs` está vivo, es alcanzable, se anuncia como feature y
  no lo prueba nada.

  ```
  entrada:  un cambio en `kmer::extract_canonical_kmers` que rompa la
            firma que `wasm.rs` usa
  actual:   `cargo test` pasa; `cargo check` (sin features) pasa; nadie se
            entera hasta que alguien compila con --features wasm a mano
  esperado: un test que falle
  ```

  El coste de que se rompa es bajo (nadie lo usa hoy), pero el coste de
  arreglarlo también, y hoy la feature está anunciada en `Cargo.toml:49`
  como si estuviera soportada.

- **Cómo se arregla, paso a paso:** **Decisión cerrada: se prueba, no se
  borra.** A diferencia de `cms.rs` (H-29), `wasm.rs` es un punto de entrada
  con un caso de uso claro (contar k-mers en el navegador, sin servidor) que
  encaja con el territorio declarado en `docs/philosophy-narrow-not-broad.md`.
  1. Añadir un `#[cfg(test)] mod tests` a `src/wasm.rs` que ejercite la
     lógica **sin** `wasm_bindgen`, extrayendo la parte pura a una función
     privada testeable en el host.
  2. Añadir el `cargo check --features wasm --target wasm32-unknown-unknown`
     al job de CI de la tarea `18`.

- **Código ANTES / DESPUÉS:**

ANTES: `src/wasm.rs` termina sin módulo de tests.

DESPUÉS — se añade al final de `src/wasm.rs`:
```rust
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Este modulo era el unico del crate sin un solo test, y a la vez el
    /// unico que solo se compila bajo una feature que CI nunca activaba en
    /// modo test. La consecuencia practica es que un cambio de firma en
    /// `kmer` lo rompia sin que nada fallara hasta que alguien compilara a
    /// mano con `--features wasm`.
    ///
    /// Los tests de aqui se ejecutan en el host, no en wasm: lo que
    /// comprueban es la logica de conteo, que no depende de
    /// `wasm_bindgen`. Que la capa `#[wasm_bindgen]` compile de verdad lo
    /// cubre `cargo check --features wasm --target wasm32-unknown-unknown`
    /// en CI, que es la comprobacion que corresponde a esa capa.
    #[test]
    fn counting_a_known_sequence_agrees_with_the_core() {
        let seq = b"ACGTACGTACGTACGT";
        let k = 4;

        let expected = crate::kmer::extract_canonical_kmers(seq, k);
        assert!(!expected.is_empty(), "el fixture debe producir k-mers");

        let mut counter = crate::counter::KmerCounter::new(k);
        counter.insert_batch(&expected);
        assert_eq!(counter.distinct_kmers(), {
            let mut unique = expected.clone();
            unique.sort_unstable();
            unique.dedup();
            unique.len()
        });
    }

    #[test]
    fn an_ambiguous_base_resets_the_window_here_too() {
        // Misma regla que en el nucleo: una N no produce k-mers corruptos.
        let with_n = crate::kmer::extract_canonical_kmers(b"ACGTNACGT", 4);
        assert_eq!(with_n.len(), 2, "cuatro bases a cada lado de la N");
    }

    #[test]
    fn an_out_of_range_k_yields_nothing_rather_than_panicking() {
        for k in [0usize, 33, 64] {
            assert!(
                crate::kmer::extract_canonical_kmers(b"ACGTACGTACGT", k).is_empty(),
                "k={k} debe dar vacio, no entrar en panico"
            );
        }
    }
}
```

- **Efectos colaterales:** ninguno. Los tests no requieren la feature `wasm`
  activa porque no tocan `wasm_bindgen`.

- **Tests a añadir:** son los de arriba.

- **Cómo verificar:**
  ```bash
  cargo test --lib wasm::
  cargo check --features wasm --target wasm32-unknown-unknown
  ```
  Salida esperada: `3 passed` y la comprobación cruzada en verde.

- **Esfuerzo:** 1 h.
---

# Eje 8 · Documentación — 2/5

---
**H-20 · El README documenta 5 de las 13 funciones públicas y no hay ningún sitio de documentación**

- **Severidad:** alta
- **Dónde:** `README.md:538-740` (la sección "Python API reference" completa); ausencia de `mkdocs.yml`, `docs/conf.py` o cualquier configuración de sitio
- **Qué pasa hoy:** la referencia de API del README tiene exactamente cinco
  entradas (`grep -n "^### \`fastdna\." README.md`):

```
540:### `fastdna.count(path, *, k=31, min_count=1, ...) -> KmerCounts`
636:### `fastdna.peek(path, *, n_reads=10_000) -> Preview`
666:### `fastdna.build_info() -> dict`
679:### `fastdna.sketch(path, *, k=21, sketch_size=1000) -> Sketch`
721:### `fastdna.estimate_cardinality(path, *, k=31, precision=14) -> float`
```

  **Sin entrada:** `compare`, `compare_all`, `frac_sketch`, `load_sketch`,
  `load_frac_sketch`, y las tres clases `KmerCounts`, `Sketch`, `FracSketch`.
  Tampoco hay nada sobre los 25 submódulos (`gwas`, `taxonomy`, `annotate`,
  `cv`, `sklearn`, `workflow`, …), que son 11 967 líneas de código Python
  con `__all__` cuidados y docstrings largos que **no se publican en ninguna
  parte**.

- **Por qué es un problema:** caso reproducible:

  ```
  entrada:  un usuario quiere comparar 200 muestras y busca como
  actual:   `compare_all` no aparece en el README; para descubrirla hay que
            leer python/fastdna/__init__.py:564 en GitHub
  esperado: una pagina de referencia con las 13 funciones y los submodulos
  ```

  El desequilibrio es llamativo: hay 104 KB de `docs/ARCHITECTURE.html` y
  77 KB de `docs/design-minimizer-counting.md` —documentos internos, para
  quien mantiene— y ninguna página para quien usa. `sourmash`, el
  competidor más directo en el ecosistema Python, publica en readthedocs.

- **Cómo se arregla, paso a paso:**
  1. **Decisión cerrada: MkDocs con `mkdocstrings`, no Sphinx.** Razón: los
     docstrings del proyecto son prosa markdown, no reStructuredText;
     `mkdocstrings` los renderiza tal cual y Sphinx obligaría a reescribir
     11 967 líneas de docstrings o a añadir `myst-parser` y configurarlo.
     Alternativa descartada: Sphinx + napoleon, más pesado para el mismo
     resultado.
  2. Crear `mkdocs.yml` con navegación por secciones.
  3. Crear `docs/site/` con una página por área que use `::: fastdna.<mod>`
     para autogenerar desde los docstrings ya escritos — no se redacta
     documentación nueva, se publica la que existe.
  4. Publicar con GitHub Pages desde el workflow de CI.
  5. **Decisión cerrada:** el README **no** se convierte en un índice
     mínimo. Se mantiene largo: es la primera impresión en GitHub y su
     contenido técnico ("How the Rust core actually works") es un
     diferenciador real. Sólo se le añade un enlace al sitio.

- **Código ANTES / DESPUÉS:**

ANTES: no existe `mkdocs.yml`.

DESPUÉS — archivo nuevo `mkdocs.yml`:
```yaml
# El sitio publica los docstrings que ya existen; no se redacta
# documentacion nueva aqui. `python/fastdna/` tiene 11 967 lineas con
# docstrings largos y `__all__` cuidados que hasta ahora solo eran
# legibles abriendo los archivos en GitHub: el README documentaba 5 de
# las 13 funciones publicas y ninguno de los 25 submodulos.
#
# MkDocs y no Sphinx porque los docstrings de este proyecto son prosa en
# markdown. mkdocstrings los renderiza tal cual; Sphinx exigiria
# reescribirlos en reStructuredText o anadir y configurar myst-parser
# para llegar al mismo sitio.
site_name: FastDNA
site_description: A fast genomic k-mer counter with a zero-copy Arrow bridge to Python
repo_url: https://github.com/anzeledon/fastdna
edit_uri: ""

theme:
  name: material
  features:
    - navigation.sections
    - content.code.copy

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [python]
          options:
            docstring_style: google
            show_source: false
            show_root_heading: true
            members_order: source

nav:
  - Home: index.md
  - Counting: api/counting.md
  - Sketching and comparison: api/sketching.md
  - Cohorts and machine learning: api/ml.md
  - Genomics utilities: api/genomics.md
```

Archivo nuevo `docs/site/api/counting.md`:
```markdown
# Counting

Everything that turns FASTQ/FASTA input into k-mer counts.

::: fastdna.count

::: fastdna.KmerCounts

::: fastdna.peek

::: fastdna.estimate_cardinality

::: fastdna.build_info
```

Archivo nuevo `docs/site/api/sketching.md`:
```markdown
# Sketching and comparison

MinHash and FracMinHash sketches, and the comparisons built on them.
`compare_all` is the entry point for cohort-scale work: it builds one
sketch per sample and compares every pair, which is O(N) FASTQ reads
rather than the O(N^2) that comparing full k-mer sets pairwise would cost.

::: fastdna.sketch

::: fastdna.Sketch

::: fastdna.load_sketch

::: fastdna.frac_sketch

::: fastdna.FracSketch

::: fastdna.load_frac_sketch

::: fastdna.compare

::: fastdna.compare_all
```

Archivo nuevo `docs/site/api/ml.md`:
```markdown
# Cohorts and machine learning

The layer with no equivalent in KMC3, FastK or Jellyfish: k-mer features
that go straight into scikit-learn, with cross-validation folds derived
from the samples' own genetic distances rather than assumed independent.

::: fastdna.sklearn

::: fastdna.cv

::: fastdna.gwas

::: fastdna.workflow

::: fastdna.multiomics

::: fastdna.evaluation

::: fastdna.calibration

::: fastdna.interpret

::: fastdna.active_learning

::: fastdna.anomaly

::: fastdna.mic

::: fastdna.rules

::: fastdna.embed
```

Archivo nuevo `docs/site/api/genomics.md`:
```markdown
# Genomics utilities

::: fastdna.taxonomy

::: fastdna.metagenomics

::: fastdna.genomescope

::: fastdna.assembly_qc

::: fastdna.annotate

::: fastdna.translate

::: fastdna.spectrum

::: fastdna.equivalence

::: fastdna.interop

::: fastdna.plotting

::: fastdna.report
```

Y en `README.md`, justo debajo del título (`README.md:1-9`), se añade:
```markdown
📖 **[Full API documentation](https://anzeledon.github.io/fastdna/)** — the
README covers the engine and the CLI; every public function and submodule
is documented on the site.
```

- **Efectos colaterales:** ninguno en código. Añade dependencias de
  documentación, que van en un extra `docs` de `pyproject.toml`, no en
  runtime.

- **Tests a añadir:** archivo nuevo `python/tests/test_docs_cover_public_api.py`,
  caso `test_every_public_function_appears_in_the_docs_site`.

```python
"""Impide que el sitio se quede atras cuando se anada API publica.

El README llego a documentar 5 de 13 funciones publicas sin que nada lo
detectara. Esto convierte esa deriva en un fallo de test.
"""
from __future__ import annotations

import pathlib

import fastdna

DOCS = pathlib.Path(__file__).resolve().parents[2] / "docs" / "site" / "api"


def test_every_public_function_appears_in_the_docs_site():
    assert DOCS.is_dir(), f"falta el directorio del sitio: {DOCS}"
    rendered = "\n".join(p.read_text(encoding="utf-8") for p in DOCS.glob("*.md"))

    missing = [
        name
        for name in fastdna.__all__
        if name != "__version__" and f"::: fastdna.{name}" not in rendered
    ]
    assert missing == [], (
        f"estos nombres de fastdna.__all__ no tienen entrada en docs/site/api/: {missing}"
    )
```

- **Cómo verificar:**
  ```bash
  pip install mkdocs-material mkdocstrings[python]
  mkdocs build --strict
  python -m pytest python/tests/test_docs_cover_public_api.py -q
  ```
  Salida esperada: build sin avisos y `1 passed`.

- **Esfuerzo:** 5 h.
---

---
**H-21 · `cargo doc` emite un aviso: documentación pública enlaza a un ítem privado**

- **Severidad:** baja
- **Dónde:** `src/metagenomics.rs` — el doc comment de `Taxonomy` que
  referencia `Taxonomy::from_taxa`
- **Qué pasa hoy:** medido:

  ```
  entrada:  cargo doc --no-deps
  actual:   warning: public documentation for `Taxonomy` links to private
            item `Taxonomy::from_taxa`
            warning: `fastdna` (lib doc) generated 1 warning
  esperado: 0 warnings
  ```

- **Por qué es un problema:** en la documentación generada, ese enlace se
  renderiza como texto plano en vez de como enlace, así que el lector ve
  una referencia a algo que no puede encontrar. Es el único aviso de
  rustdoc en todo el crate, lo que lo hace barato de dejar en cero — y una
  vez en cero, se puede exigir con `-D warnings` en CI y no volver a
  acumular.

- **Cómo se arregla, paso a paso:**
  1. Localizar la línea:
     `grep -n "from_taxa" src/metagenomics.rs`.
  2. **Decisión cerrada:** se degrada el enlace a código en línea
     (`` `from_taxa` `` con comillas simples en vez de `[` `]`), **no** se
     hace pública `from_taxa`. Razón: el constructor es privado a propósito
     —la construcción pública pasa por `build_database`— y hacerlo público
     para arreglar un aviso de documentación ampliaría la API por el motivo
     equivocado.
  3. Añadir `RUSTDOCFLAGS="-D warnings"` al job de CI (tarea `18`).

- **Código ANTES / DESPUÉS:**

ANTES (en el doc comment de `Taxonomy`, `src/metagenomics.rs`):
```rust
/// ... construida por [`Taxonomy::from_taxa`], que valida que cada
/// identificador padre exista antes de aceptar el arbol.
```

DESPUÉS:
```rust
/// ... construida internamente por `from_taxa` (privada: la via publica de
/// construccion es [`build_database`], que la invoca), la cual valida que
/// cada identificador padre exista antes de aceptar el arbol.
```

> Nota para el ejecutor: la línea exacta se localiza con
> `cargo doc --no-deps 2>&1 | grep -A3 "links to private item"`, que imprime
> el número de línea. El texto de arriba muestra la forma del cambio —
> reemplazar el enlace intra-doc por código en línea y apuntar al camino
> público— no una cita literal que pueda estar desfasada.

- **Efectos colaterales:** ninguno. Es un comentario.

- **Tests a añadir:** ninguno unitario; la verificación es el propio
  `cargo doc`, que entra en CI como paso con `-D warnings`.

- **Cómo verificar:**
  ```bash
  RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
  ```
  Salida esperada: termina en éxito. Hoy falla con el aviso convertido en
  error.

- **Esfuerzo:** 15 min.
---

---
**H-22 · El comentario que gobierna la CI dice "la suite de 50 tests"; hoy son 496**

- **Severidad:** baja
- **Dónde:** `.github/workflows/wheels.yml:26-27`
- **Qué pasa hoy:**

```yaml
# A separate "run the Rust tests" job, if one is ever added to this file,
# must use bare `cargo test` (no features) -- the existing 50-test suite
# -- never `--all-features` or `--features python`.
```

- **Por qué es un problema:** la suite real, medida hoy, es de **358 tests
  de lib más 138 de integración = 496**. El número del comentario está
  desfasado casi diez veces. No es cosmético: ese comentario es la
  instrucción normativa que leerá quien implemente el job de tests (tarea
  `18`), y una cifra tan lejana de la realidad hace dudar de si el resto
  del párrafo —que **sí es correcto y sí importa**, el análisis sobre
  `extension-module` y los símbolos `Py_*`— sigue vigente.

  ```
  entrada:  alguien va a anadir el job de tests y lee este comentario
  actual:   "50-test suite" -- no coincide con nada; ¿esta obsoleto todo?
  esperado: una cifra correcta, o ninguna cifra
  ```

- **Cómo se arregla, paso a paso:** **Decisión cerrada: se quita la cifra en
  vez de actualizarla.** Un número exacto en un comentario vuelve a quedarse
  obsoleto en el siguiente commit que añada un test; lo que el comentario
  necesita transmitir es *qué* comando usar, no cuántos tests hay.
  1. Reemplazar "the existing 50-test suite" por una descripción sin cifra.
  2. Mantener intacto todo el análisis sobre `extension-module`.

- **Código ANTES / DESPUÉS:**

ANTES (`.github/workflows/wheels.yml:20-27`):
```yaml
# This file never runs `cargo build`/`cargo test` with `--features python` or
# `--all-features`, and never will, on any OS. `pyo3/extension-module` tells
# the crate not to link `libpython` -- correct for the cdylib maturin builds,
# but on Linux and macOS (unlike Windows, where pythonXY.lib gets linked
# regardless) that genuinely suppresses the libpython link, so any target
# that links the rlib and pulls in the `ffi` module -- a plain `cargo
# build --features python`, run outside maturin -- fails there with undefined
# `Py_*` symbols. `maturin build --release --features python` below is the one
# safe way to build this feature: maturin builds `--lib` as a cdylib and never
# turns the feature on for the `[[bin]]`. A separate "run the Rust tests" job,
# if one is ever added to this file, must use bare `cargo test` (no features)
# -- the existing 50-test suite -- never `--all-features` or `--features python`.
```

DESPUÉS:
```yaml
# This file never runs `cargo build`/`cargo test` with `--features python` or
# `--all-features`, and never will, on any OS. `pyo3/extension-module` tells
# the crate not to link `libpython` -- correct for the cdylib maturin builds,
# but on Linux and macOS (unlike Windows, where pythonXY.lib gets linked
# regardless) that genuinely suppresses the libpython link, so any target
# that links the rlib and pulls in the `ffi` module -- a plain `cargo
# build --features python`, run outside maturin -- fails there with undefined
# `Py_*` symbols. `maturin build --release --features python` below is the one
# safe way to build this feature: maturin builds `--lib` as a cdylib and never
# turns the feature on for the `[[bin]]`. The "run the Rust tests" job in
# `ci.yml` must use bare `cargo test` (no features), never `--all-features`
# or `--features python`, for exactly that reason.
#
# No test count is quoted here on purpose: this comment used to say "the
# existing 50-test suite" and the suite is now 496 tests, which made the
# whole paragraph -- including the `extension-module` analysis above, which
# is still correct and still load-bearing -- look stale.
```

- **Efectos colaterales:** ninguno; es un comentario. La referencia a
  `ci.yml` presupone la tarea `18`, que crea ese archivo; ambas van en la
  misma fase.

- **Tests a añadir:** ninguno.

- **Cómo verificar:**
  ```bash
  grep -c "50-test suite" .github/workflows/wheels.yml
  ```
  Salida esperada: `0` (hoy: `1`).

- **Esfuerzo:** 10 min.
---

# Eje 9 · Versionado y releases — 1/5

---
**H-23 · La versión del CLI está escrita a mano: `fastdna --version` mentirá en el próximo bump**

- **Severidad:** alta
- **Dónde:** `src/cli.rs:103`
- **Qué pasa hoy:**

```rust
#[derive(Parser, Debug)]
#[command(name = "fastdna", version = "0.1.0", author = "FastDNA Team")]
pub struct Cli {
```

  Mientras que el FFI lo hace bien, en dos sitios:

```rust
src/ffi.rs:722:    dict.set_item("version", env!("CARGO_PKG_VERSION"))?;
src/ffi.rs:1550:   m.add("__version__", env!("CARGO_PKG_VERSION"))?;
```

  Y `pyproject.toml:14-18` documenta explícitamente que la versión vive
  sólo en `Cargo.toml` y se declara `dynamic` justamente para no duplicarla:
  *"the version stays sourced from Cargo.toml's `[package] version` alone
  -- the same value `ffi.rs` reports to Python via
  `env!("CARGO_PKG_VERSION")` -- rather than duplicating it."*

  **`cli.rs:103` es exactamente la duplicación que ese comentario dice
  haber evitado.**

- **Por qué es un problema:** reproducible en un solo paso:

  ```
  entrada:  sed -i 's/^version = "0.1.0"/version = "0.2.0"/' Cargo.toml
            cargo build --release
            ./target/release/fastdna --version
  actual:   fastdna 0.1.0
            python -c "import fastdna; print(fastdna.__version__)"  -> 0.2.0
  esperado: fastdna 0.2.0  en ambos
  ```

  Un usuario que reporte un bug dirá "estoy en 0.1.0" y el mantenedor
  buscará en el código equivocado. Peor: un pipeline que registre
  `fastdna --version` para reproducibilidad estará guardando un dato falso.
  Y sucede **en el primer release**, que es exactamente la tarea `12`.

- **Cómo se arregla, paso a paso:**
  1. Sustituir el literal por `env!("CARGO_PKG_VERSION")`.
  2. **Decisión cerrada sobre `author`:** `"FastDNA Team"` se sustituye por
     `env!("CARGO_PKG_AUTHORS")`… salvo que `Cargo.toml` no declara
     `authors`. Así que: **se borra el campo `author` entero.** clap lo
     omite si no está, y "FastDNA Team" es una entidad que no existe —
     `git log` muestra un único autor. Alternativa descartada: poner el
     nombre real a mano, que reintroduce el mismo problema de duplicación.
  3. Añadir un test que ate las dos versiones.

- **Código ANTES / DESPUÉS:**

ANTES (`src/cli.rs:102-104`):
```rust
#[derive(Parser, Debug)]
#[command(name = "fastdna", version = "0.1.0", author = "FastDNA Team")]
pub struct Cli {
```

DESPUÉS:
```rust
// `version` sale de Cargo.toml, no de un literal. `pyproject.toml` ya
// documenta que la version vive en un solo sitio -- "the version stays
// sourced from Cargo.toml's [package] version alone ... rather than
// duplicating it" -- y esta linea era precisamente la duplicacion que ese
// parrafo daba por evitada: `ffi.rs` usa env!("CARGO_PKG_VERSION") en sus
// dos sitios y el CLI llevaba "0.1.0" escrito a mano, asi que el primer
// bump habria hecho que `fastdna --version` y `fastdna.__version__`
// discreparan.
//
// `author` se elimina en vez de corregirse: Cargo.toml no declara
// `authors`, asi que no hay nada de donde leerlo, y "FastDNA Team" no es
// una entidad que exista. clap omite el campo cuando no se da.
#[derive(Parser, Debug)]
#[command(name = "fastdna", version = env!("CARGO_PKG_VERSION"))]
pub struct Cli {
```

- **Efectos colaterales:** la salida de `fastdna --help` pierde la línea de
  autor. `tests/cli_args.rs` no la comprueba (`grep -n "author\|FastDNA Team"
  tests/` → vacío). No es breaking change.

- **Tests a añadir:** en `tests/cli_args.rs`, caso
  `cli_version_matches_the_crate_version`.

```rust
/// Ata la version que imprime el CLI a la de Cargo.toml.
///
/// `src/cli.rs` llevaba `version = "0.1.0"` escrito a mano mientras
/// `src/ffi.rs` usaba `env!("CARGO_PKG_VERSION")`, de modo que el primer
/// bump de version habria hecho que `fastdna --version` reportara una
/// cosa y `fastdna.__version__` otra. Este test falla si alguien vuelve a
/// poner un literal.
#[test]
fn cli_version_matches_the_crate_version() {
    let rendered = Cli::command().render_version();
    let expected = env!("CARGO_PKG_VERSION");

    assert!(
        rendered.contains(expected),
        "`fastdna --version` imprime {rendered:?}, que no contiene la version \
         del crate ({expected}). Revisa el atributo `version` de `#[command(...)]` \
         en src/cli.rs."
    );
}
```

  (Requiere `use clap::CommandFactory;` en la cabecera del archivo de test.)

- **Cómo verificar:**
  ```bash
  cargo test --test cli_args cli_version_matches
  ```
  Salida esperada: `1 passed`. Para comprobar el síntoma original:
  cambiar la versión en `Cargo.toml`, recompilar y confirmar que
  `fastdna --version` la sigue.

- **Esfuerzo:** 20 min.
---

---
**H-24 · No hay CHANGELOG, ni tags, ni política de versionado declarada**

- **Severidad:** alta
- **Dónde:** ausencia — `ls CHANGELOG*` vacío, `git tag -l` vacío, y ningún documento de `docs/` declara política de compatibilidad
- **Qué pasa hoy:** el proyecto está en `0.1.0` desde el snapshot inicial
  (`2ca63c3`) hasta hoy (`7daa41f`), 48 commits después, sin ningún tag ni
  registro de cambios. `git log --oneline` muestra trabajo sustancial
  (nuevas estrategias de conteo, módulos ML completos, entrada FASTA,
  soporte paired-end) que un usuario no puede consultar.

- **Por qué es un problema:** el plan de esta auditoría introduce varios
  cambios rompedores conscientes (H-06 `top(None)`, H-08 `pub(crate)`,
  H-11 campos de `FastDnaError`, H-15 umbrales negativos). Sin CHANGELOG no
  hay dónde anunciarlos.

  ```
  entrada:  un usuario actualiza de la version X a la Y
  actual:   no existe ningun documento que diga que cambio; ni siquiera
            existen X e Y, porque no hay tags
  esperado: CHANGELOG.md con las secciones Added/Changed/Fixed/Removed
  ```

  Y hay un problema más agudo que se vuelve permanente en cuanto se
  publique: **no está declarado qué es API pública.** Con 296 ítems `pub`
  (H-08) y sin política, cada uno de ellos es implícitamente estable.

- **Cómo se arregla, paso a paso:**
  1. Crear `CHANGELOG.md` en formato Keep a Changelog, con una sección
     `[Unreleased]` que recoja lo que ya está hecho y lo que este plan añade.
  2. **Decisión cerrada sobre la política:** SemVer, con la salvedad de que
     mientras la versión sea `0.x`, un bump de menor (`0.1 → 0.2`) puede
     romper. Es lo que SemVer especifica para `0.x` y lo que el ecosistema
     Rust asume.
  3. **Decisión cerrada sobre el alcance de la API:** lo público es lo que
     `python/fastdna/__init__.py.__all__` exporta (H-09) más los módulos
     `pub mod` que quedan tras H-08. Todo lo demás es interno, se declare
     `pub` o no.

- **Código ANTES / DESPUÉS:**

ANTES: el archivo no existe.

DESPUÉS — archivo nuevo `CHANGELOG.md`:
```markdown
# Changelog

Formato: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versionado: [SemVer](https://semver.org/), con la salvedad que SemVer mismo
define para `0.x` — mientras la versión mayor sea 0, un incremento de la
menor (`0.1 → 0.2`) puede introducir cambios rompedores.

## Qué cubre este contrato

La superficie con compatibilidad garantizada es:

- **Python:** los nombres de `fastdna.__all__`, y los submódulos
  documentados en el sitio (`docs/site/api/`). Los nombres con guion bajo
  inicial y cualquier cosa no listada en `__all__` son internos.
- **Rust:** los módulos declarados `pub mod` en `src/lib.rs`. Los
  `pub(crate) mod` son implementación y cambian sin aviso.
- **CLI:** los flags y subcomandos documentados en `fastdna --help`.

Los formatos de archivo (Parquet de conteos, JSON de sketch, base de datos
de metagenómica) llevan su propia versión interna y se tratan como parte
del contrato.

## [Unreleased]

### Added
- Excepciones propias en Python (`fastdna.FastDnaError` y sus siete
  subclases), de forma que un fallo de configuración y un archivo corrupto
  puedan distinguirse por tipo. Heredan también del builtin que lanzaban
  antes, así que todo `except ValueError:` existente sigue funcionando.
- Marcador PEP 561 (`py.typed`) y anotaciones de tipo en la API pública.
- Subcomandos `sketch`, `dist` y `card` en el CLI.
- Sitio de documentación en `https://anzeledon.github.io/fastdna/`.

### Changed
- **(rompedor)** `KmerCounts.top(n)` exige un `int` no negativo. Antes
  `top(None)` devolvía la tabla entera y `top(-5)` una vacía, ambos en
  silencio. Migración: usar el propio objeto en lugar de `top(None)`.
- **(rompedor)** `KmerCounts.filter()` rechaza umbrales negativos, que
  antes eran un no-op silencioso.
- **(rompedor, sólo Rust)** `adaptive_bins`, `atomic`, `binned`,
  `disk_spill`, `minimizer` y `superkmer` pasan a `pub(crate)`.
- **(rompedor, sólo Rust)** `FastDnaError::Export` y `::Load` ganan un
  campo `source` que preserva el error original.
- `fastdna --version` lee la versión de `Cargo.toml` en lugar de un
  literal.

### Fixed
- El wheel ya no embarca archivos `.pyc` (30 entradas, 715 837 bytes).
- Los mensajes de `MatrixTooLarge`/`VocabTooLarge` ya no citan flags de
  CLI inexistentes.
- `sort_by()` con una columna desconocida nombra las válidas.

### Removed
- `src/cms.rs` (Count-Min Sketch): 247 líneas sin ningún uso.
- La copia duplicada del extractor de k-mers en `src/metagenomics.rs`.
```

- **Efectos colaterales:** ninguno en código.

- **Tests a añadir:** ninguno automatizable con valor. La disciplina se
  sostiene con una casilla en la plantilla de PR (H-30).

- **Cómo verificar:**
  ```bash
  test -f CHANGELOG.md && head -5 CHANGELOG.md
  ```
  Salida esperada: el archivo existe y empieza por `# Changelog`.

- **Esfuerzo:** 1 h.
---

---
**H-25 · `FastDnaError` no es `#[non_exhaustive]`: añadir una variante rompe a todo consumidor**

- **Severidad:** media
- **Dónde:** `src/error.rs:6-47` (la declaración del `enum`)
- **Qué pasa hoy:**

```rust
/// Every fallible operation in the FastDNA core returns this type.
#[derive(Debug)]
pub enum FastDnaError {
    Io { path: PathBuf, source: std::io::Error },
    ...
    Cancelled,
}
```

- **Por qué es un problema:** un consumidor puede escribir un `match`
  exhaustivo sin brazo comodín, y entonces cualquier variante nueva es un
  cambio rompedor.

  ```
  entrada:  un consumidor escribe
              match err {
                  FastDnaError::Io { .. } => ...,
                  FastDnaError::InvalidK { .. } => ...,
                  // ... las 13 variantes, sin `_ =>`
              }
            y el crate anade una variante 14
  actual:   error[E0004]: non-exhaustive patterns -- el consumidor no
            compila tras un cambio que deberia ser menor
  esperado: compila; el brazo comodin que `#[non_exhaustive]` obliga a
            escribir absorbe la variante nueva
  ```

  Esto no es hipotético en este crate: la historia muestra variantes
  añadidas repetidamente —`Load` y `NoSamplesFound` llevan comentarios que
  dicen literalmente "not in the spec's table (which predates the cohort
  engine)"— y el plan de esta auditoría no añade ninguna, pero el roadmap
  (S1/S2 de `feature-gap-analysis.md`) sí lo hará.

- **Cómo se arregla, paso a paso:**
  1. Añadir `#[non_exhaustive]` sobre el `enum`.
  2. **Decisión cerrada:** sólo sobre el `enum`, no sobre cada variante.
     Marcar las variantes obligaría al propio crate a construirlas con
     `..Default::default()`, y aquí las variantes tienen campos
     obligatorios sin default sensato. El nivel de `enum` es el que
     resuelve el problema descrito.
  3. Añadir el brazo `_ =>` que ahora exige el `match` de
     `src/ffi.rs:71-103`. **Decisión cerrada:** ese brazo mapea a
     `PyRuntimeError`, igual que `Internal`, porque una variante que este
     `match` no conoce es por definición un fallo no clasificado.

- **Código ANTES / DESPUÉS:**

ANTES (`src/error.rs:5-8`):
```rust
/// Every fallible operation in the FastDNA core returns this type.
#[derive(Debug)]
pub enum FastDnaError {
    /// An underlying I/O failure, carrying the path that caused it.
```

DESPUÉS:
```rust
/// Every fallible operation in the FastDNA core returns this type.
///
/// `#[non_exhaustive]`: adding a variant must stay a minor change. Without
/// it, a consumer writing an exhaustive `match` with no wildcard arm stops
/// compiling every time this enum grows -- and it has grown repeatedly
/// (`Load` and `NoSamplesFound` both carry comments noting they postdate
/// the original spec's error table), with more coming as the k-mer
/// database and set-operation work lands. The attribute forces outside
/// callers to write a `_ =>` arm today so that tomorrow's variant is
/// absorbed by it.
///
/// Applied to the enum and not to individual variants on purpose: marking
/// variants would force this crate's own construction sites into
/// functional-update syntax, and these variants have required fields with
/// no sensible default.
#[derive(Debug)]
#[non_exhaustive]
pub enum FastDnaError {
    /// An underlying I/O failure, carrying the path that caused it.
```

ANTES (`src/ffi.rs:100-103`), el final del `match`:
```rust
            FastDnaError::Cancelled => PyKeyboardInterrupt::new_err(err.to_string()),
            FastDnaError::Internal { .. } => PyRuntimeError::new_err(err.to_string()),
        }
    }
}
```

DESPUÉS:
```rust
            FastDnaError::Cancelled => PyKeyboardInterrupt::new_err(message),
            FastDnaError::Internal { .. } => PyRuntimeError::new_err(message),
            // `FastDnaError` es `#[non_exhaustive]`, asi que este brazo es
            // obligatorio aunque hoy sea inalcanzable. Una variante que
            // esta tabla no conoce es, por definicion, un fallo sin
            // clasificar: se trata como `Internal`. Anadir una variante y
            // olvidarse de mapearla degrada a RuntimeError con su mensaje
            // real, en vez de impedir la compilacion del binding.
            _ => PyRuntimeError::new_err(message),
        }
    }
}
```

- **Efectos colaterales:** dentro del crate, `#[non_exhaustive]` no aplica
  (un `enum` así se puede seguir emparejando exhaustivamente desde su
  propio crate), así que sólo `ffi.rs` —que está en el mismo crate— **no**
  necesitaría el brazo... salvo que sí conviene por lo que explica el
  comentario. `tests/error_handling.rs` está fuera del crate; verificar que
  ningún `match` suyo sea exhaustivo sin comodín (hoy usa `matches!`, que no
  lo es). Es breaking change para consumidores externos; no hay (H-02).

- **Tests a añadir:** en `tests/error_handling.rs`, caso
  `error_enum_can_be_matched_with_a_wildcard_from_outside_the_crate`.

```rust
/// Un test de integracion vive fuera del crate, que es donde
/// `#[non_exhaustive]` tiene efecto. Que este `match` compile con su brazo
/// comodin es la prueba de que un consumidor puede escribir codigo que
/// sobreviva a la siguiente variante.
#[test]
fn error_enum_can_be_matched_with_a_wildcard_from_outside_the_crate() {
    use fastdna_core::FastDnaError;

    fn classify(err: &FastDnaError) -> &'static str {
        match err {
            FastDnaError::InvalidK { .. } => "configuracion",
            FastDnaError::Cancelled => "cancelado",
            _ => "otro",
        }
    }

    assert_eq!(classify(&FastDnaError::InvalidK { k: 99 }), "configuracion");
    assert_eq!(classify(&FastDnaError::Cancelled), "cancelado");
    assert_eq!(
        classify(&FastDnaError::Internal { detail: "x".to_string() }),
        "otro"
    );
}
```

- **Cómo verificar:**
  ```bash
  cargo test --test error_handling
  ```
  Salida esperada: `18 passed` (los 17 existentes más éste).

- **Esfuerzo:** 30 min.
---
