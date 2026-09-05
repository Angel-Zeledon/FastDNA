"""`fastdna.metagenomics` against Kraken 2, on the same panel and the same reads.

This closes the one hole `docs/validation-real-data.md` names in Track C:
"`fastdna.metagenomics` is the one module whose reference implementation is
not checked head to head here". Track B compared it against *ground truth*
(real E. coli reads, 99.16% of species-level calls correct) but never against
the tool it is a reimplementation of.

## Why not a RefSeq-scale Kraken 2 database

The reason Track C gave for skipping this -- "a fair comparison needs a
Kraken 2 database built over the full NCBI taxonomy" -- had it backwards. A
full-RefSeq Kraken 2 against a 5-genome FastDNA panel measures the
*databases*, and the database would win by orders of magnitude before a
single line of either algorithm mattered. The comparison that isolates the
algorithm is the opposite one: give both tools **the same five genomes, the
same taxonomy tree, and the same reads**, and every difference that remains
is implementation.

## The two arms, and why there are two

`metagenomics.py`'s docstring makes a specific, unmeasured claim: that
storing every exact 64-bit k-mer instead of Kraken 2's spaced-seed
minimizers costs real sensitivity, because "an exact k-mer match has no
tolerance for a substitution". That claim needs the comparison split in two,
because the two halves answer different questions.

* **Arm A -- same algorithm.** Kraken 2 built with `-k 31 -l 31 -s 0`: the
  minimizer is the whole k-mer and there are no spaced positions, so it
  stores exactly the k-mer set FastDNA stores. Two independent
  implementations of one algorithm over one database. Disagreement here is a
  bug in one of them, and this is the arm that either does or does not
  reproduce the KMC3-style *exact equality* result the counting engine got.
* **Arm B -- Kraken 2 as people run it.** Its own defaults, `-k 35 -l 31
  -s 7`. This is the arm that puts a number on the sensitivity trade-off the
  docstring asserts, and it is the honest answer to "should I use this
  instead of Kraken 2".

Both arms are built `--no-masking`. Kraken 2 normally runs `dustmasker` over
the library and drops low-complexity k-mers; FastDNA has no such step, so
leaving masking on would mix a difference in *database contents* into a
comparison meant to be about the algorithm. `--masked` adds a third arm that
turns it back on, for the size of that effect on its own.

## Ground truth

The reads are real: ENA run **DRR002015**, submitted as *Escherichia coli*
2W14. Barring contamination, a species-level call that is not *E. coli* is
wrong, for either tool. That keeps a third question answerable alongside the
agreement rate -- not just "do they agree" but "when they disagree, which
one is right".

The reference *E. coli* is K-12 MG1655, a different strain from the one
sequenced. That is deliberate: a reference identical to the sample would
make both tools look perfect and measure nothing about the strain drift that
is the normal case in practice.

## Not a speed comparison

Kraken 2 has no arm64 build here and runs under x86-64 emulation, while
FastDNA runs native. Wall times are printed because they are what happened,
and they are explicitly not a claim about either tool's speed.

Usage:

    python scripts/validation/metagenomics_vs_kraken2.py
    python scripts/validation/metagenomics_vs_kraken2.py --max-reads 200000
    python scripts/validation/metagenomics_vs_kraken2.py --masked --json k2.json

Downloads ~110 MB on the first run and caches it. Requires Docker.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

#: k for FastDNA and for Arm A. 31 is what Track B used and what the module
#: documents its memory model against.
K = 31

#: The Kraken 2 image. Pinned: `latest` would make a re-run of this script a
#: comparison against a different program.
KRAKEN2_IMAGE = "staphb/kraken2:2.1.3"

#: One complete RefSeq assembly per species, by accession rather than by
#: BV-BRC genome id, so this panel is reconstructible by anyone. Track B's
#: panel was the same five species.
PANEL: List[Tuple[str, int, str]] = [
    ("Escherichia coli", 562,
     "GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2"),
    ("Staphylococcus aureus", 1280,
     "GCF/000/013/425/GCF_000013425.1_ASM1342v1/GCF_000013425.1_ASM1342v1"),
    ("Klebsiella pneumoniae", 573,
     "GCF/000/240/185/GCF_000240185.1_ASM24018v2/GCF_000240185.1_ASM24018v2"),
    ("Pseudomonas aeruginosa", 287,
     "GCF/000/006/765/GCF_000006765.1_ASM676v1/GCF_000006765.1_ASM676v1"),
    ("Salmonella enterica", 28901,
     "GCF/000/006/945/GCF_000006945.2_ASM694v2/GCF_000006945.2_ASM694v2"),
]

_NCBI_BASE = "https://ftp.ncbi.nlm.nih.gov/genomes/all/"

#: Real Illumina WGS reads submitted to ENA as *Escherichia coli* 2W14 --
#: the same accession Track B used, so this run's FastDNA numbers are
#: comparable with the ones already in `docs/validation-real-data.md`.
READS_URL = ("https://ftp.sra.ebi.ac.uk/vol1/fastq/DRR002/DRR002015/"
             "DRR002015_1.fastq.gz")

#: The shared taxonomy: `(tax_id, parent, rank, name)`, real NCBI ids and
#: real ranks. Written out twice -- once as FastDNA's TSV, once as Kraken 2's
#: `nodes.dmp`/`names.dmp` -- from this one definition, because the two tools
#: assigning a read to different nodes because they were given different
#: trees would be a comparison of my two files, not of them.
#:
#: The intermediate ranks are here to make disagreement legible. With only
#: root and five species, every read a tool is unsure about lands on root and
#: all information about *why* is gone. With the real lineage in place, a
#: read the tool can place in Enterobacteriaceae but not in a genus says so.
TAXONOMY: List[Tuple[int, int, str, str]] = [
    (1, 0, "no rank", "root"),
    (2, 1, "superkingdom", "Bacteria"),
    # Pseudomonadota (Proteobacteria): four of the five species.
    (1224, 2, "phylum", "Pseudomonadota"),
    (1236, 1224, "class", "Gammaproteobacteria"),
    (91347, 1236, "order", "Enterobacterales"),
    (543, 91347, "family", "Enterobacteriaceae"),
    (561, 543, "genus", "Escherichia"),
    (562, 561, "species", "Escherichia coli"),
    (590, 543, "genus", "Salmonella"),
    (28901, 590, "species", "Salmonella enterica"),
    (570, 543, "genus", "Klebsiella"),
    (573, 570, "species", "Klebsiella pneumoniae"),
    (72274, 1236, "order", "Pseudomonadales"),
    (135621, 72274, "family", "Pseudomonadaceae"),
    (286, 135621, "genus", "Pseudomonas"),
    (287, 286, "species", "Pseudomonas aeruginosa"),
    # Bacillota (Firmicutes): the out-group, a different phylum entirely.
    (1239, 2, "phylum", "Bacillota"),
    (91061, 1239, "class", "Bacilli"),
    (1385, 91061, "order", "Bacillales"),
    (90964, 1385, "family", "Staphylococcaceae"),
    (1279, 90964, "genus", "Staphylococcus"),
    (1280, 1279, "species", "Staphylococcus aureus"),
]

_NAME_OF = {tax_id: name for tax_id, _, _, name in TAXONOMY}
_NAME_OF[0] = "unclassified"
_RANK_OF = {tax_id: rank for tax_id, _, rank, _ in TAXONOMY}
_RANK_OF[0] = "unclassified"
_SPECIES_IDS = {tax_id for tax_id, _, rank, _ in TAXONOMY if rank == "species"}

#: The truth for DRR002015: every species-level call that is not this is a
#: false positive.
TRUTH_TAX_ID = 562


# ---------------------------------------------------------------- downloads


def _download(url: str, dest: Path) -> Path:
    """Downloads to a temporary name and renames, so an interrupted download
    can never be mistaken for a cached file on the next run."""
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    print(f"  downloading {dest.name} ...", flush=True)
    with urllib.request.urlopen(url, timeout=300) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    partial.rename(dest)
    return dest


def fetch_inputs(work: Path) -> Tuple[Path, Path]:
    """The panel FASTA and the read file, both cached.

    The panel is written as one FASTA with every header carrying
    `|kraken:taxid|<id>`. Both tools read that same file: Kraken 2's
    `add_to_library` parses the taxid out of the sequence id, and FastDNA
    takes the whole first whitespace-delimited token as the sequence id,
    which the taxonomy TSV then maps. One file, no chance of the two tools
    being given different sequence.
    """
    panel = work / "panel.fasta"
    if not panel.is_file():
        parts = []
        for species, tax_id, accession in PANEL:
            local = _download(_NCBI_BASE + accession + "_genomic.fna.gz",
                              work / "refs" / (accession.rsplit("/", 1)[-1] + ".fna.gz"))
            with gzip.open(local, "rt") as handle:
                for line in handle:
                    if line.startswith(">"):
                        fields = line[1:].rstrip("\n").split(None, 1)
                        description = fields[1] if len(fields) > 1 else ""
                        parts.append(f">{fields[0]}|kraken:taxid|{tax_id} {description}\n")
                    else:
                        parts.append(line)
            print(f"  {species:<24} taxid {tax_id:<6} {local.name}")
        panel.write_text("".join(parts))
    reads = _download(READS_URL, work / "DRR002015_1.fastq.gz")
    return panel, reads


def plain_fastq(reads_gz: Path, work: Path, max_reads: int) -> Path:
    """The reads as an uncompressed FASTQ, optionally truncated to the first
    `max_reads`.

    Decompressed once rather than handed to each tool as `.gz`: Kraken 2's
    handling of compressed input has varied across versions, and neither
    tool's gzip path is what is being measured. The *first* n reads rather
    than a random n, when truncating -- both tools then see byte-identical
    input, and for an agreement measurement between two deterministic
    classifiers a random subsample buys nothing a prefix does not.
    """
    out = work / (f"reads_{max_reads}.fastq" if max_reads else "reads_all.fastq")
    if out.is_file() and out.stat().st_size > 0:
        return out
    partial = out.with_suffix(".partial")
    limit = max_reads * 4 if max_reads else None
    with gzip.open(reads_gz, "rt") as handle, partial.open("w") as sink:
        for i, line in enumerate(handle):
            if limit is not None and i >= limit:
                break
            sink.write(line)
    partial.rename(out)
    return out


def sequence_ids_by_taxon(panel: Path) -> Dict[int, List[str]]:
    """Which sequence ids belong to which taxon, read back out of the FASTA
    that was just written -- rather than remembered while writing it -- so
    the taxonomy handed to FastDNA describes the file that actually exists."""
    by_taxon: Dict[int, List[str]] = {}
    with panel.open() as handle:
        for line in handle:
            if line.startswith(">"):
                seq_id = line[1:].split(None, 1)[0]
                tax_id = int(seq_id.rsplit("|", 1)[1])
                by_taxon.setdefault(tax_id, []).append(seq_id)
    return by_taxon


# ----------------------------------------------------------------- fastdna


def write_fastdna_taxonomy(path: Path, by_taxon: Dict[int, List[str]]) -> None:
    lines = ["tax_id\tparent_tax_id\trank\tname\tsequence_ids"]
    for tax_id, parent, rank, name in TAXONOMY:
        lines.append(f"{tax_id}\t{parent}\t{rank}\t{name}\t"
                     + ";".join(by_taxon.get(tax_id, [])))
    path.write_text("\n".join(lines) + "\n")


def run_fastdna(panel: Path, taxonomy: Path, reads: Path, k: int) -> Dict:
    from fastdna.metagenomics import build_database

    started = time.monotonic()
    db = build_database(panel, taxonomy, k=k)
    build_seconds = time.monotonic() - started

    started = time.monotonic()
    calls = db.classify(reads, confidence_threshold=0.0)
    classify_seconds = time.monotonic() - started

    return {
        "arm": f"FastDNA k={k}",
        "n_kmers": int(db.n_kmers),
        "database_bytes": int(db.memory_bytes),
        "build_seconds": round(build_seconds, 1),
        "classify_seconds": round(classify_seconds, 1),
        "table": calls,
    }


# ----------------------------------------------------------------- kraken 2


def write_kraken_taxonomy(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    nodes, names = [], []
    for tax_id, parent, rank, name in TAXONOMY:
        # Kraken 2 reads fields 0/1/2 of a `\t|\t`-separated line as
        # id/parent/rank; the trailing columns exist only because dmp files
        # have them. Root is self-parented, as in NCBI's own nodes.dmp.
        nodes.append(f"{tax_id}\t|\t{parent or tax_id}\t|\t{rank}\t|\t-\t|\t0\t|")
        names.append(f"{tax_id}\t|\t{name}\t|\t\t|\tscientific name\t|")
    (directory / "nodes.dmp").write_text("\n".join(nodes) + "\n")
    (directory / "names.dmp").write_text("\n".join(names) + "\n")


def run_kraken2(work: Path, panel: Path, reads: Path, *, label: str, kmer_len: int,
                minimizer_len: int, minimizer_spaces: int, masked: bool,
                threads: int) -> Dict:
    """Builds a Kraken 2 database over the panel and classifies the reads.

    The build is cached by its parameters: three arms over the same five
    genomes would otherwise rebuild the same table three times, and a rerun
    to change only the analysis would rebuild all of them.
    """
    slug = f"k{kmer_len}_l{minimizer_len}_s{minimizer_spaces}{'_masked' if masked else ''}"
    db_dir = work / f"k2db_{slug}"
    output = work / f"k2_{slug}_{reads.name}.out"
    stamp = db_dir / "hash.k2d"

    write_kraken_taxonomy(db_dir / "taxonomy")
    mask_flag = "" if masked else "--no-masking"

    build_seconds = 0.0
    if not stamp.is_file():
        script = f"""
set -euo pipefail
cd /work
kraken2-build --db {db_dir.name} --add-to-library panel.fasta {mask_flag} --threads {threads}
kraken2-build --db {db_dir.name} --build {mask_flag} \
    --kmer-len {kmer_len} --minimizer-len {minimizer_len} \
    --minimizer-spaces {minimizer_spaces} --threads {threads}
"""
        started = time.monotonic()
        _docker(work, script)
        build_seconds = time.monotonic() - started

    started = time.monotonic()
    _docker(work, f"""
set -euo pipefail
cd /work
kraken2 --db {db_dir.name} --threads {threads} --confidence 0.0 \
    --output {output.name} --report {output.name}.report {reads.name}
""")
    classify_seconds = time.monotonic() - started

    return {
        "arm": label,
        "n_kmers": _kraken_distinct_minimizers(work, db_dir, threads),
        "database_bytes": sum(f.stat().st_size for f in db_dir.glob("*.k2d")),
        "build_seconds": round(build_seconds, 1),
        "classify_seconds": round(classify_seconds, 1),
        "output": output,
    }


def _docker(work: Path, script: str) -> str:
    result = subprocess.run(
        ["docker", "run", "--rm", "--platform", "linux/amd64",
         "-v", f"{work}:/work", "-w", "/work", KRAKEN2_IMAGE, "bash", "-c", script],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-4000:] + "\n" + result.stderr[-4000:] + "\n")
        raise RuntimeError(f"kraken2 container failed ({result.returncode})")
    return result.stdout


def _kraken_distinct_minimizers(work: Path, db_dir: Path, threads: int) -> Optional[int]:
    """How many distinct keys the database holds, from `kraken2-inspect`.

    Kraken 2 stores *minimizers*, so this equals FastDNA's `n_kmers` only in
    Arm A, where `-l` equals `-k` and every k-mer is its own minimizer. That
    makes it a real equality check there -- two independent implementations
    reducing the same five genomes to the same number of distinct canonical
    31-mers -- and deliberately not comparable in Arm B, where a smaller
    number is the whole point of a minimizer.

    Returns `None` rather than raising if inspect fails: a missing count
    should not throw away a completed classification run.
    """
    try:
        report = _docker(work, f"kraken2-inspect --db /work/{db_dir.name} "
                               f"--threads {threads} --skip-counts")
    except RuntimeError:
        return None
    total = 0
    for line in report.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split("\t")
        # pct, clade-cumulative count, this-node count, rank, taxid, name
        if len(fields) >= 3 and fields[2].strip().isdigit():
            total += int(fields[2])
    return total or None


# --------------------------------------------------------------- comparison


def kraken_calls(output: Path) -> Iterator[Tuple[str, int]]:
    """`(read_id, tax_id)` per line of a Kraken 2 output file.

    Kraken 2 writes `C/U <read_id> <taxid> <length> <k-mer map>`, and taxid
    is `0` for an unclassified read -- the same convention `classify()`
    returns, which is why the two are directly comparable without a mapping
    step.
    """
    with output.open() as handle:
        for line in handle:
            fields = line.split("\t", 3)
            if len(fields) < 3:
                continue
            yield fields[1], int(fields[2])


def compare(fastdna_result: Dict, kraken_result: Dict) -> Dict:
    """Per-read agreement, streamed.

    Positional rather than a join on read id: both tools emit one row per
    read in input order, so the ids must line up, and checking that they do
    at every row is a stronger assertion than a join (a join would silently
    tolerate one tool having dropped or reordered reads, which is exactly the
    kind of thing worth catching). The ids are compared, not assumed.
    """
    table = fastdna_result["table"]
    ours_ids = table.column("read_id")
    ours_tax = table.column("tax_id")

    confusion: Dict[Tuple[int, int], int] = {}
    stream = kraken_calls(kraken_result["output"])
    n = 0
    for id_chunk, tax_chunk in zip(ours_ids.chunks, ours_tax.chunks):
        for read_id, tax_id in zip(id_chunk.to_pylist(), tax_chunk.to_pylist()):
            try:
                their_id, their_tax = next(stream)
            except StopIteration:
                raise RuntimeError(
                    f"kraken2 produced fewer rows than FastDNA (stopped at read {n})")
            if their_id != read_id:
                raise RuntimeError(
                    f"read {n}: FastDNA says {read_id!r}, kraken2 says {their_id!r} -- "
                    "the two outputs are not in the same order, so a positional "
                    "comparison would be meaningless")
            confusion[(tax_id, their_tax)] = confusion.get((tax_id, their_tax), 0) + 1
            n += 1
    if next(stream, None) is not None:
        raise RuntimeError("kraken2 produced more rows than FastDNA")

    agree = sum(count for (a, b), count in confusion.items() if a == b)
    return {"n_reads": n, "agreement": agree / n if n else 0.0, "confusion": confusion}


def arm_summary(tax_ids: Dict[int, int], n_reads: int) -> Dict:
    """The three numbers that matter per arm: how much was classified, how
    much reached a species, and how much of that species-level call was the
    species the reads actually are."""
    species_calls = {t: c for t, c in tax_ids.items() if t in _SPECIES_IDS}
    n_species = sum(species_calls.values())
    return {
        "n_reads": n_reads,
        "unclassified": tax_ids.get(0, 0),
        "unclassified_pct": 100.0 * tax_ids.get(0, 0) / n_reads if n_reads else 0.0,
        "species_level": n_species,
        "species_level_pct": 100.0 * n_species / n_reads if n_reads else 0.0,
        "correct_species": species_calls.get(TRUTH_TAX_ID, 0),
        "species_precision": (100.0 * species_calls.get(TRUTH_TAX_ID, 0) / n_species
                              if n_species else float("nan")),
        "by_tax_id": {str(t): c for t, c in sorted(tax_ids.items(), key=lambda kv: -kv[1])},
    }


def tally_fastdna(table) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for chunk in table.column("tax_id").chunks:
        for tax_id in chunk.to_pylist():
            counts[tax_id] = counts.get(tax_id, 0) + 1
    return counts


def tally_kraken(output: Path) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for _, tax_id in kraken_calls(output):
        counts[tax_id] = counts.get(tax_id, 0) + 1
    return counts


def print_arm(label: str, result: Dict, summary: Dict) -> None:
    print(f"\n{label}")
    print(f"  database            {result['database_bytes'] / 1e6:,.1f} MB"
          + (f", {result['n_kmers']:,} distinct k-mers" if result.get("n_kmers") else ""))
    print(f"  build / classify    {result['build_seconds']:.0f}s / "
          f"{result['classify_seconds']:.0f}s")
    print(f"  unclassified        {summary['unclassified']:,} "
          f"({summary['unclassified_pct']:.2f}%)")
    print(f"  species-level call  {summary['species_level']:,} "
          f"({summary['species_level_pct']:.2f}%)")
    print(f"  of those, E. coli   {summary['correct_species']:,} "
          f"({summary['species_precision']:.2f}% precision)")
    top = list(summary["by_tax_id"].items())[:6]
    print("  calls               "
          + ", ".join(f"{_NAME_OF.get(int(t), t)} {c:,}" for t, c in top))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work-dir", type=Path,
                        default=Path.home() / ".fastdna" / "validation" / "kraken2")
    parser.add_argument("--max-reads", type=int, default=0,
                        help="classify only the first N reads (0 = all of them)")
    parser.add_argument("--threads", type=int, default=4,
                        help="threads for kraken2; FastDNA's classify is single-threaded")
    parser.add_argument("--masked", action="store_true",
                        help="add a third arm with Kraken 2's dustmasker step left on")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--k", type=int, default=K)
    args = parser.parse_args()

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)

    print("inputs")
    panel, reads_gz = fetch_inputs(work)
    reads = plain_fastq(reads_gz, work, args.max_reads)
    print(f"  panel  {panel} ({panel.stat().st_size / 1e6:.1f} MB)")
    print(f"  reads  {reads.name} ({reads.stat().st_size / 1e6:.1f} MB)")

    taxonomy = work / "panel_taxonomy.tsv"
    write_fastdna_taxonomy(taxonomy, sequence_ids_by_taxon(panel))

    print("\nFastDNA ...", flush=True)
    ours = run_fastdna(panel, taxonomy, reads, args.k)
    our_tally = tally_fastdna(ours["table"])
    n_reads = sum(our_tally.values())
    our_summary = arm_summary(our_tally, n_reads)

    arms = [(
        "Arm A -- Kraken 2, same algorithm (-k 31 -l 31 -s 0)",
        dict(label="kraken2 k=31 l=31 s=0", kmer_len=args.k, minimizer_len=args.k,
             minimizer_spaces=0, masked=False),
    ), (
        "Arm B -- Kraken 2, its own defaults (-k 35 -l 31 -s 7)",
        dict(label="kraken2 defaults", kmer_len=35, minimizer_len=31,
             minimizer_spaces=7, masked=False),
    )]
    if args.masked:
        arms.append((
            "Arm C -- Kraken 2 defaults with masking on",
            dict(label="kraken2 defaults, masked", kmer_len=35, minimizer_len=31,
                 minimizer_spaces=7, masked=True),
        ))

    rows = {"fastdna": {**{kk: vv for kk, vv in ours.items() if kk != "table"},
                        **our_summary}}
    print_arm(f"FastDNA k={args.k} (exact 64-bit k-mers)", ours, our_summary)

    for heading, kwargs in arms:
        print(f"\n{heading} ...", flush=True)
        theirs = run_kraken2(work, panel, reads, threads=args.threads, **kwargs)
        their_tally = tally_kraken(theirs["output"])
        their_summary = arm_summary(their_tally, sum(their_tally.values()))
        print_arm(heading, theirs, their_summary)

        agreement = compare(ours, theirs)
        print(f"  per-read agreement  {100 * agreement['agreement']:.4f}% "
              f"over {agreement['n_reads']:,} reads")
        disagreements = sorted(
            ((c, a, b) for (a, b), c in agreement["confusion"].items() if a != b),
            reverse=True,
        )[:5]
        for count, a, b in disagreements:
            print(f"    {count:>9,}  FastDNA {_NAME_OF.get(a, a):<24} "
                  f"kraken2 {_NAME_OF.get(b, b)}")

        rows[kwargs["label"]] = {
            **{kk: vv for kk, vv in theirs.items() if kk != "output"},
            **their_summary,
            "agreement_with_fastdna": agreement["agreement"],
            "confusion": {f"{a}->{b}": c for (a, b), c in agreement["confusion"].items()},
        }
        rows[kwargs["label"]]["output"] = str(theirs["output"])

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, default=str))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
