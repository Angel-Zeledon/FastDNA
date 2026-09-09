# Sketching and comparison

A sketch is a fixed-size fingerprint of a sample's canonical k-mer set. It
answers *"how similar are these two samples"*, not *"what exactly is in
each"* — and that restriction is what makes cohort-scale comparison
affordable.

`compare_all()` is the entry point for cohort work: it builds one sketch per
sample and then compares every pair, which costs `O(N)` FASTQ reads plus
`O(N²)` sketch comparisons. Each comparison is `O(sketch_size)`, not
`O(genome size)`. Comparing full k-mer sets pair by pair would instead cost
`O(N²)` FASTQ reads — the exact cost sketching exists to avoid.

`compare_all` gives every pair at once, which is what makes an all-pairs
distance matrix over a cohort affordable: N sketches built once, then
`N*(N-1)/2` comparisons that never touch a FASTQ again.

## MinHash

::: fastdna.sketch

::: fastdna.Sketch

::: fastdna.load_sketch

::: fastdna.sketch_from_kmers

## FracMinHash (scaled)

::: fastdna.frac_sketch

::: fastdna.FracSketch

::: fastdna.load_frac_sketch

## Comparing

::: fastdna.compare

::: fastdna.compare_all
