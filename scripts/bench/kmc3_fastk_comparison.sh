#!/usr/bin/env bash
# scripts/bench/kmc3_fastk_comparison.sh
#
# Exact steps used to produce the README's large-scale (2.14 GB) comparison
# against KMC3 and FASTK. Run inside WSL2 (Ubuntu); FastDNA's own side of
# the comparison is the native Windows release binary, run separately
# (`cargo build --release`, then `fastdna.exe --input ... --output ...`).
#
# This is a record of what was actually run, not a one-command installer:
# KMC3 and FASTK are fetched from their own upstream releases/source, which
# this script does not vendor or mirror.
set -euo pipefail

WORKDIR="${1:-$HOME/fastdna-bench}"
mkdir -p "$WORKDIR/tools"
cd "$WORKDIR"

# --- KMC3 (prebuilt release binary) ---------------------------------------
if [ ! -x tools/bin/kmc ]; then
  curl -sL -o tools/kmc.tar.gz \
    https://github.com/refresh-bio/KMC/releases/download/v3.2.4/KMC3.2.4.linux.x64.tar.gz
  mkdir -p tools && tar xzf tools/kmc.tar.gz -C tools
  chmod +x tools/bin/kmc tools/bin/kmc_tools tools/bin/kmc_dump
fi

# --- FASTK (built from source; needs build-essential/zlib/bzip2/lzma/curl/ssl) ---
if [ ! -x tools/FASTK/FastK ]; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq build-essential zlib1g-dev git \
    libbz2-dev liblzma-dev libcurl4-openssl-dev libssl-dev autoconf
  git clone -q https://github.com/thegenemyers/FASTK.git tools/FASTK
  make -C tools/FASTK -j"$(nproc)"
fi

# --- Generate the same 2.14 GB dataset FastDNA's Windows side counts -----
# Same generator, same seed, run from WSL's own Python (needs numpy) so
# the file lands on WSL-native ext4, not the slower /mnt/c 9P mount --
# this matters for the Linux tools' own I/O time, not for correctness:
# the generator is deterministic, so this is byte-identical to a copy
# generated on the Windows side with the same arguments and seed.
python3 -m venv .venv 2>/dev/null || true
.venv/bin/pip install -q numpy
.venv/bin/python "$(dirname "$0")/generate_reads_large.py" \
  35000000 30 150 bench2gb.fastq 9001 200000

K=31
THREADS="$(nproc)"

# --- KMC3 ---
# -ci1: do not exclude singletons (KMC's own default excludes k-mers seen
# only once; FastDNA's default min_count=1 does not, so this matches it).
mkdir -p kmc_work
/usr/bin/time -v tools/bin/kmc -k$K -t$THREADS -m10 -ci1 -fq \
  bench2gb.fastq kmc_out kmc_work

# --- FASTK ---
# -t1: keep k-mers seen >=1 times (FASTK's own default), matching min_count=1.
mkdir -p fastk_work
/usr/bin/time -v tools/FASTK/FastK -k$K -T$THREADS -M10 -t1 \
  -P fastk_work -N fastk_out bench2gb.fastq
tools/FASTK/Histex -h1:100000 -A fastk_out > fastk_hist.txt

echo "KMC3 distinct k-mers:"
head -20 kmc_stdout.log 2>/dev/null || true
echo "FASTK distinct k-mers (sum of fastk_hist.txt column 2):"
python3 -c "print(sum(int(l.split()[1]) for l in open('fastk_hist.txt') if l.split() and l.split()[0].isdigit()))"
