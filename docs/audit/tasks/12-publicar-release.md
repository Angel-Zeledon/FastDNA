# Tarea 12 — Publicar la primera release

**Objetivo:** que `pip install fastdna` funcione.

**Hallazgos que resuelve:** H-02.
**Prerequisitos:** tareas 00–11 completas.
**Archivos exactos:** `Cargo.toml`, `README.md`, `CHANGELOG.md`,
`.github/workflows/wheels.yml`.
**Esfuerzo:** 2 h.

## Por qué esta tarea es el hito de la fase 0

Hoy, verificado por HTTP el 2026-08-26:

```
GET https://pypi.org/pypi/fastdna/json          -> HTTP 404
GET https://crates.io/api/v1/crates/fastdna     -> HTTP 404
```

`git tag -l` está vacío. `.github/workflows/wheels.yml` dice literalmente
*"This workflow only builds and tests wheels. It intentionally has no
publish step."*

Hay ~32 000 líneas de código con 93 % de cobertura y cero usuarios
posibles. **Ambos nombres están libres**, lo que también significa que
cualquiera puede registrarlos y el README estaría dirigiendo usuarios a un
paquete ajeno.

## Pasos

1. Subir la versión en `Cargo.toml` a `0.2.0` (hay cambios rompedores en el
   CHANGELOG, y bajo SemVer `0.x` un bump de menor puede romper).

2. Cerrar la sección `[Unreleased]` de `CHANGELOG.md` como `[0.2.0]` con la
   fecha de hoy.

3. Quitar el aviso de "no publicado" del README que puso la tarea 02, y
   devolver `pip install fastdna` al principio de la sección Installation.

4. Añadir el job de publicación a `.github/workflows/wheels.yml`:

```yaml
  publish:
    name: Publicar en PyPI
    needs: build
    runs-on: ubuntu-latest
    # Solo en tags. Un push a una rama no debe publicar nunca.
    if: startsWith(github.ref, 'refs/tags/v')
    environment: release
    permissions:
      # Trusted publishing de PyPI: sin token de API guardado en secretos.
      id-token: write
    steps:
      - uses: actions/download-artifact@v7
        with:
          pattern: wheel-*
          merge-multiple: true
          path: dist
      - uses: pypa/gh-action-pypi-publish@release/v1
```

5. Configurar trusted publishing en PyPI para el repositorio
   (`pypi.org/manage/account/publishing/`). No guardar tokens en secretos.

6. Etiquetar y empujar:
   ```bash
   git tag -a v0.2.0 -m "FastDNA 0.2.0"
   git push origin v0.2.0
   ```

7. Publicar el crate:
   ```bash
   cargo publish --dry-run
   cargo publish
   ```

## Criterios de aceptación

- `pip install fastdna` en una máquina limpia instala y
  `python -c "import fastdna; print(fastdna.__version__)"` imprime `0.2.0`.
- `fastdna --version` imprime `fastdna 0.2.0` (esto lo garantiza la tarea 03).
- `https://pypi.org/project/fastdna/` existe.
- `README.md` ya no contiene la cadena `has not been published yet`.

## Comando exacto de verificación

```bash
python -m venv /tmp/clean && /tmp/clean/bin/pip install fastdna
/tmp/clean/bin/python -c "import fastdna; print(fastdna.__version__)"
```

Salida esperada: `0.2.0`.

## Qué NO tocar

No publiques sin que las tareas 01–11 estén hechas. Una API publicada es
casi permanente: los cambios rompedores de `top()`, `filter()`,
`pub(crate)` y `FastDnaError` tienen que entrar **antes** del primer
release, no después.

## Riesgos

**`cargo publish` es irreversible.** Una versión publicada en crates.io no
se puede borrar, solo retirar (`yank`). Ejecuta `--dry-run` primero y
revisa la lista de archivos empaquetados.
