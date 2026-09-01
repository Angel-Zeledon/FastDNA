# Tarea 03 — La versión del CLI sale de Cargo.toml

**Objetivo:** que `fastdna --version` no mienta en el próximo bump.

**Hallazgos que resuelve:** H-23.
**Prerequisitos:** tarea 00.
**Archivos exactos:** `src/cli.rs`, `tests/cli_args.rs`.
**Esfuerzo:** 20 min.

## El problema

`src/cli.rs:103` lleva la versión escrita a mano:

```rust
#[command(name = "fastdna", version = "0.1.0", author = "FastDNA Team")]
```

Mientras que el FFI lo hace bien en dos sitios:

```
src/ffi.rs:722:    dict.set_item("version", env!("CARGO_PKG_VERSION"))?;
src/ffi.rs:1550:   m.add("__version__", env!("CARGO_PKG_VERSION"))?;
```

Y `pyproject.toml` documenta explícitamente que la versión vive sólo en
`Cargo.toml` *"rather than duplicating it"*. La línea 103 de `cli.rs` **es**
esa duplicación.

Reproducible en un paso: cambia `version` en `Cargo.toml` a `0.2.0`,
recompila, y `fastdna --version` sigue diciendo `0.1.0` mientras
`fastdna.__version__` dice `0.2.0`.

## Pasos

1. En `src/cli.rs`, sustituir el atributo `#[command(...)]`.
2. En `tests/cli_args.rs`, añadir `use clap::CommandFactory;` a la cabecera
   si no está, y el test de abajo.

## Código ANTES

`src/cli.rs:102-104`:

```rust
#[derive(Parser, Debug)]
#[command(name = "fastdna", version = "0.1.0", author = "FastDNA Team")]
pub struct Cli {
```

## Código DESPUÉS

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

## Test completo

Añadir al final de `tests/cli_args.rs`:

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
    use clap::CommandFactory;

    let rendered = Cli::command().render_version();
    let expected = env!("CARGO_PKG_VERSION");

    assert!(
        rendered.contains(expected),
        "`fastdna --version` imprime {rendered:?}, que no contiene la version \
         del crate ({expected}). Revisa el atributo `version` de \
         `#[command(...)]` en src/cli.rs."
    );
}
```

## Criterios de aceptación

- `cargo test --test cli_args` en verde.
- Cambiar la versión de `Cargo.toml` y recompilar hace que
  `fastdna --version` la siga.

## Comando exacto de verificación

```bash
cargo test --test cli_args cli_version_matches
```

Salida esperada: `test cli_version_matches_the_crate_version ... ok`.

## Qué NO tocar

No añadas `authors` a `Cargo.toml` para "arreglar" el campo `author`. Se
elimina a propósito.

## Riesgos

`fastdna --help` pierde la línea de autor. Ningún test la comprueba
(verificado: `grep -n "FastDNA Team" tests/` no devuelve nada).
