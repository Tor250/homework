import time
import random
import os
from lsm_v3 import LSMTree
import shutil

def benchmark_lsm():
    tree_path = "bench_data"
    if os.path.exists(tree_path):
        for f in os.listdir(tree_path):
            os.remove(os.path.join(tree_path, f))
    tree = LSMTree(tree_path, mem_limit=500, level_factor=10)

    N = 50000
    keys = [f"key{i}" for i in range(N)]
    values = [f"value{i}".encode() for i in range(N)]
    start = time.time()
    for k, v in zip(keys, values):
        tree.put(k, v)
    tree.close()
    print(f"Insertion of {N} items: {time.time() - start:.3f}s")
    random_keys = random.sample(keys, 5000)
    start = time.time()
    for k in random_keys:
        v = tree.get(k)
        assert v is not None
    print(f"Point reads (5000 random keys): {time.time() - start:.3f}s")
    ranges = [(f"key{i}", f"key{i+100}") for i in range(0, N, 1000)]
    start = time.time()
    for start_key, end_key in ranges:
        result = tree.range(start_key, end_key)
        assert all(start_key <= k <= end_key for k, _ in result)
    print(f"Range reads ({len(ranges)} ranges of 100 keys): {time.time() - start:.3f}s")
    tree.report_stats()
    shutil.rmtree("bench_data", ignore_errors=True)

if __name__ == "__main__":
    benchmark_lsm()