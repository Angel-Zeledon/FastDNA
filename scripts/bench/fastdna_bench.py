import sys
import time
import fastdna

path = sys.argv[1]
k = int(sys.argv[2])
threads = sys.argv[3]
threads = None if threads == "none" else int(threads)
repeats = int(sys.argv[4]) if len(sys.argv) > 4 else 3

times = []
r = None
for i in range(repeats):
    t0 = time.perf_counter()
    r = fastdna.count(path, k=k, threads=threads)
    times.append(time.perf_counter() - t0)

times.sort()
median = times[len(times) // 2]
print(
    f"threads={threads} k={k} reads_file={path} "
    f"distinct={r.distinct_kmers} total={r.total_kmers} "
    f"times={[round(t,4) for t in times]} median={median:.4f}s"
)
