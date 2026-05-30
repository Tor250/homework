import random
import time

from pyroaring import BitMap

from hw5.positional_index import HW5PositionalIndex, tokenize


def _naive_phrase_positions(tokens, phrase_terms): # возвращает список позиций начала фразы в списке токенов документа
    if not phrase_terms:
        return []
    window = len(phrase_terms)
    matches = []
    for start in range(len(tokens) - window + 1):
        if tokens[start:start + window] == phrase_terms:
            matches.append(start)
    return matches


def _naive_phrase_search(docs, phrase): # возвращает (bitmap, positions_map, counts) для фразы в списке документов, где bitmap - BitMap с doc_id документов, содержащих фразу, positions_map - dict of doc_id -> list of позиций начала фразы в документе, counts - dict of doc_id -> количество вхождений фразы в документ
    phrase_terms = tokenize(phrase, remove_stopwords=False)
    positions = {}
    for doc_id, text in enumerate(docs):
        doc_tokens = tokenize(text, remove_stopwords=False)
        starts = _naive_phrase_positions(doc_tokens, phrase_terms)
        if starts:
            positions[doc_id] = starts
    bitmap = BitMap(positions.keys())
    counts = {doc_id: len(starts) for doc_id, starts in positions.items()}
    return bitmap, positions, counts

def benchmark_build(doc_count=10000, doc_len=60): # проверяет, что построение индекса с заданным количеством документов и длиной каждого документа выполняется за разумное время
    idx = HW5PositionalIndex(
        use_lsm=True,
        lsm_path="data/hw5_benchmark_lsm",
        clear_on_init=True,
        lsm_auto_flush_docs=200,
    )
    start = time.time()
    for doc_id in range(doc_count):
        tokens = [f"word{random.randint(0, 2000)}" for _ in range(doc_len)]
        if doc_id % 50 == 0:
            insert_at = random.randint(0, doc_len - 4)
            tokens[insert_at:insert_at + 4] = ["alpha", "beta", "gamma", "delta"]
        idx.add_document(" ".join(tokens))
    elapsed = time.time() - start
    print(f"Build {doc_count} docs took {elapsed:.2f}s")
    return idx


def benchmark_phrase_queries(idx, query_count=1000): # проверяет, что выполнение заданного количества запросов на поиск фразы выполняется за разумное время и возвращает результаты
    queries = []
    for i in range(query_count):
        if i % 3 == 0:
            queries.append("alpha beta gamma delta")
        else:
            left = random.randint(0, 1997)
            queries.append(f"word{left} word{left + 1}")

    start = time.time()
    total_hits = 0
    for query in queries:
        total_hits += len(idx.search_phrase(query))
    elapsed = time.time() - start
    avg_ms = elapsed * 1000.0 / max(query_count, 1)
    print(f"{query_count} phrase queries took {elapsed:.2f}s ({avg_ms:.2f} ms/query), total hits={total_hits}")

def benchmark_comparison(doc_count=1000, query_phrase="alpha beta gamma delta"): # сравнивает время выполнения поиска фразы в позиционном индексе с наивным поиском по всем документам, проверяя, что результаты совпадают и что позиционный индекс работает быстрее
    docs_texts = []
    for doc_id in range(doc_count):
        tokens = [f"word{random.randint(0, 500)}" for _ in range(50)]
        if doc_id % 10 == 0:
            tokens[10:14] = ["alpha", "beta", "gamma", "delta"]
        docs_texts.append(" ".join(tokens))
    idx = HW5PositionalIndex(use_lsm=False, clear_on_init=True)
    for text in docs_texts:
        idx.add_document(text)
    start_idx = time.time()
    res_idx = idx.search_phrase(query_phrase)
    end_idx = time.time()
    time_idx = (end_idx - start_idx) * 1000
    start_naive = time.time()
    res_naive, _, _ = _naive_phrase_search(docs_texts, query_phrase)
    end_naive = time.time()
    time_naive = (end_naive - start_naive) * 1000
    assert res_idx == res_naive, "Ошибка: результаты поиска не совпадают!"
    print(f"Positional Index Time: {time_idx:.4f} ms")
    print(f"Naive Search Time:    {time_naive:.4f} ms")
    print(f"Speedup: {time_naive / time_idx:.1f}x faster")



if __name__ == "__main__":
    random.seed(42)
    index = benchmark_build()
    benchmark_phrase_queries(index)
    benchmark_comparison(2000)
    index.close()
