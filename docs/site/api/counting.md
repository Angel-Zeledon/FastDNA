# Counting

The core surface: turning a FASTQ(.gz) file into canonical k-mer counts, and
everything that helps decide how to count it. These names are available
directly on the `fastdna` package.

Counting is **exact** under both of FastDNA's strategies — the key really is
the k-mer, not a hash of it, so there is no probability of two k-mers
colliding into one count.

::: fastdna.count

::: fastdna.KmerCounts

::: fastdna.peek

!!! info "What `peek()` returns"

    `Preview` is implemented in the Rust core (`fastdna._core`), so its
    members are not introspectable from the Python source and do not appear
    below. It exposes:

    | Member | Meaning |
    |---|---|
    | `n_reads_sampled` | How many records were actually sampled |
    | `read_length` | `(min, median, max)` read length among them |
    | `gc_content` | GC fraction over the sampled bases |
    | `sample_distinct_kmers` | Exact distinct count — over the sampled prefix only |
    | `suggest_k()` | The largest odd `k` at most `median_read_length / 3`, clamped to `1..=32` |

::: fastdna.estimate_cardinality

::: fastdna.build_info

## Choosing a `min_count` threshold

::: fastdna.spectrum

## Counting sequences already in memory

::: fastdna.interop
