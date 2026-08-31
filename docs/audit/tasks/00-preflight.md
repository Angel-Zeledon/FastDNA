# Tarea 00 — Commitear el trabajo pendiente

**Objetivo:** dejar el árbol limpio para que las referencias `archivo:línea`
del resto de tareas no se muevan bajo los pies del ejecutor.

**Hallazgos que resuelve:** ninguno (precondición).
**Prerequisitos:** ninguno.
**Esfuerzo:** 10 min.

## Contexto

El árbol tiene 5 archivos modificados sin commitear, procedentes de una
sesión anterior de trabajo de rendimiento:

```
 M docs/PERFORMANCE_PLAN.md
 M python/fastdna/multiomics.py
 M src/counter.rs
 M src/mem_estimate.rs
 M src/pipeline.rs
```

Todas las demás tareas citan números de línea de `src/counter.rs`,
`src/mem_estimate.rs` y `src/pipeline.rs`. Si estos cambios se revierten o
se commitean a medias, esos números dejan de valer.

## Pasos

1. Comprobar que el árbol está en verde:
   ```bash
   cargo test --all-targets
   ```
   Esperado: `0 failed` en todas las suites.

2. Comprobar que clippy no ha empeorado:
   ```bash
   cargo clippy --all-targets 2>&1 | grep -cE '^warning'
   ```
   Esperado: 8 avisos, todos preexistentes.

3. Commitear:
   ```bash
   git add -A
   git commit -m "Partition the raw buffer before sorting, and model the binned strategy's peak memory"
   ```

## Criterios de aceptación

- `git status --short` no imprime nada.
- `cargo test --all-targets` sigue en verde.

## Qué NO tocar

Nada más. Esta tarea solo commitea lo que ya existe.

## Riesgos

Si los tests no pasan antes de commitear, **detente y repórtalo**. No
commitees un árbol roto para desbloquear el plan.
