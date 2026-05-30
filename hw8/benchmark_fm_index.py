import csv
import os
import random
import string
import time

from hw8.fm_index import FMIndex


def _generate_text(rng, length):
    alphabet = string.ascii_lowercase + " "
    return "".join(rng.choice(alphabet) for _ in range(length))


def _generate_queries(rng, text, pattern_length, query_count):
    alphabet = string.ascii_lowercase + " "
    queries = []
    for _ in range(query_count):
        if len(text) >= pattern_length and rng.random() < 0.6:
            start = rng.randint(0, len(text) - pattern_length)
            queries.append(text[start : start + pattern_length])
        else:
            queries.append("".join(rng.choice(alphabet) for _ in range(pattern_length)))
    return queries


def benchmark_pattern_lengths(
    text_length=60000,
    pattern_lengths=(2, 4, 8, 12, 16, 24),
    query_count=1000,
    seed=42,
    output_dir="data",
):
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)
    text = _generate_text(rng, text_length)

    start = time.perf_counter()
    index = FMIndex(text, checkpoint_step=32)
    build_seconds = time.perf_counter() - start
    print(f"Build {text_length} chars took {build_seconds:.3f}s")

    results = []
    for pattern_length in pattern_lengths:
        queries = _generate_queries(rng, text, pattern_length, query_count)
        start = time.perf_counter()
        for query in queries:
            index.search(query)
        elapsed = time.perf_counter() - start
        ms_per_query = elapsed * 1000.0 / query_count
        results.append((pattern_length, elapsed, ms_per_query))
        print(
            f"pattern len {pattern_length:>2}: {query_count} searches "
            f"{elapsed:.3f}s ({ms_per_query:.3f} ms/query)"
        )

    csv_path = os.path.join(output_dir, "hw8_fm_index_benchmark.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["pattern_length", "query_seconds", "query_ms_per_query"])
        writer.writerows(results)

    print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    benchmark_pattern_lengths()
