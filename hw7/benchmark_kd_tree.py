import csv
import os
import random
import time

from hw7.kd_tree import DiskKDTree


def _generate_points(rng, count, dims):
    return [tuple(rng.uniform(-1000.0, 1000.0) for _ in range(dims)) for _ in range(count)]


def benchmark_dimensions(
    dimensions=(2, 4, 8, 16, 32),
    point_count=4000,
    query_count=1500,
    seed=42,
    output_dir="data",
):
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)
    results = []

    for dims in dimensions:
        points = _generate_points(rng, point_count, dims)
        queries = _generate_points(rng, query_count, dims)
        tree_path = os.path.join(output_dir, f"hw7_kdtree_{dims}d.bin")

        tree = DiskKDTree(tree_path, clear_on_init=True)
        start = time.perf_counter()
        tree.build(points)
        build_seconds = time.perf_counter() - start

        # Открываем дерево с диска, чтобы замер был честным.
        reopened = DiskKDTree(tree_path)
        reopened.reset_stats()
        start = time.perf_counter()
        for query in queries:
            reopened.nearest_neighbor(query)
        query_seconds = time.perf_counter() - start

        query_ms = query_seconds * 1000.0 / query_count
        avg_reads = reopened.read_count / query_count
        results.append((dims, build_seconds, query_seconds, query_ms, avg_reads))
        print(
            f"k={dims:>2}: build {build_seconds:.3f}s, "
            f"{query_count} NN queries {query_seconds:.3f}s ({query_ms:.3f} ms/query), "
            f"{avg_reads:.1f} reads/query"
        )

    csv_path = os.path.join(output_dir, "hw7_kdtree_benchmark.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["dimensions", "build_seconds", "query_seconds", "query_ms_per_query", "avg_node_reads_per_query"])
        writer.writerows(results)

    print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    benchmark_dimensions()
