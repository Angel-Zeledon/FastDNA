# Tarea 01 — Excluir el bytecode del wheel

**Objetivo:** que `maturin build` deje de meter archivos `.pyc` dentro del
wheel publicable.

**Hallazgos que resuelve:** H-01.
**Prerequisitos:** tarea 00.
**Archivos exactos:** `pyproject.toml`, `python/tests/test_wheel_contents.py` (nuevo).
**Esfuerzo:** 15 min.

## El problema, medido

Construyendo el wheel dos veces en el mismo contenedor:

| | Bytes del wheel | Entradas `.pyc` | Bytes de `.pyc` |
|---|---:|---:|---:|
| Con `__pycache__` presente | **1 300 864** | **30** | 715 837 |
| Tras `rm -rf __pycache__` | **972 655** | 0 | 0 |

maturin copia el árbol `python-source` entero y no consulta `.gitignore`.
Si alguien ejecutó los tests o importó el paquete antes de construir, el
bytecode entra. Se embarcan a la vez `.pyc` de cpython-311 y cpython-312
dentro de un wheel `abi3` que dice cubrir 3.8–3.13+. CI hace checkout
limpio, así que **el fallo solo aparece en construcciones manuales** — que
es justo el caso de una subida de release hecha a mano.

## Pasos

1. Abrir `pyproject.toml` y localizar el bloque `[tool.maturin]`.
2. Añadir la clave `exclude` como se muestra abajo.
3. Crear `python/tests/test_wheel_contents.py` con el contenido de abajo.

## Código ANTES

`pyproject.toml`, bloque `[tool.maturin]`:

```toml
[tool.maturin]
features = ["python"]
module-name = "fastdna._core"
python-source = "python"
```

## Código DESPUÉS

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
# concreta de CPython dentro de un wheel abi3 que declara cubrir
# 3.8-3.13+, y que el contenido del wheel pasa a depender de que
# interpretes tocaron el directorio antes de construir. CI hace checkout
# limpio, asi que esto solo se manifiesta en construcciones manuales, que
# es justo el caso de una subida de release hecha a mano.
exclude = ["**/__pycache__/**", "**/*.pyc"]
```

## Test completo

Archivo nuevo `python/tests/test_wheel_contents.py`:

```python
\"\"\"Comprueba que el artefacto publicable no lleva bytecode dentro.

No es un test de la API: es un test del empaquetado. Vive aqui, y no en
CI a mano, porque el fallo que cubre solo aparece en construcciones
locales -- CI hace checkout limpio y nunca tiene __pycache__ que meter --
y por tanto no lo detectaria ningun job existente.
\"\"\"
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

## Criterios de aceptación

- El wheel construido tras haber importado el paquete no contiene ninguna
  entrada `.pyc`.
- El tamaño del wheel baja de ~1 300 000 a ~972 000 bytes.

## Comando exacto de verificación

```bash
python -c "import sys; sys.path.insert(0,'python'); import fastdna"
maturin build --release --features python --out /tmp/dist
python -c "
import glob, zipfile
z = zipfile.ZipFile(glob.glob('/tmp/dist/*.whl')[0])
print(sum(1 for i in z.infolist() if i.filename.endswith('.pyc')))"
```

Salida esperada: `0`. Antes del arreglo imprime `30`.

## Qué NO tocar

No toques `.gitignore`: ya ignora `__pycache__/` correctamente. El problema
es de empaquetado, no de git.

## Riesgos

Ninguno en tiempo de ejecución. Los `.pyc` nunca fueron API.
