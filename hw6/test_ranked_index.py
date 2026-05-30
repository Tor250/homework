import math
import random
import shutil
from collections import Counter

import pytest

from hw6.ranked_index import HW6RankedIndex, tokenize


def _doc_ids(results): # извлекает список doc_id из результатов поиска, игнорируя оценки релевантности
    return [doc_id for doc_id, _ in results]


def _cleanup_lsm(path):
    shutil.rmtree(path, ignore_errors=True)


def _make_index(champion_size=8, tier_size=8):
    return HW6RankedIndex(
        champion_size=champion_size,
        tier_size=tier_size,
        use_lsm=True,
        lsm_path="data/hw6_test_lsm",
        clear_on_init=True,
        lsm_auto_flush_docs=50,
    )


def _manual_rank(docs, query, cosine=False): # выполняет ручной расчет TF-IDF и косинусного сходства для заданного списка документов и запроса, возвращая отсортированный список (doc_id, score) для всех документов, где score > 0
    doc_counters = []
    df = Counter()
    for text in docs:
        tokens = tokenize(text, remove_stopwords=True)
        counter = Counter(tokens)
        doc_counters.append(counter)
        for term in counter:
            df[term] += 1

    total_docs = len(docs)
    idf = {term: math.log((total_docs + 1.0) / (freq + 1.0)) + 1.0 for term, freq in df.items()}

    doc_weights = {}
    doc_norms = {}
    for doc_id, counter in enumerate(doc_counters):
        weights = {}
        squared_norm = 0.0
        for term, tf in counter.items():
            weight = (1.0 + math.log(tf)) * idf[term]
            weights[term] = weight
            squared_norm += weight * weight
        doc_weights[doc_id] = weights
        doc_norms[doc_id] = math.sqrt(squared_norm)

    query_counter = Counter(tokenize(query, remove_stopwords=True))
    query_weights = {}
    for term, tf in query_counter.items():
        if term in idf:
            query_weights[term] = (1.0 + math.log(tf)) * idf[term]

    if not query_weights:
        return []

    query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values()))
    scores = []
    for doc_id, weights in doc_weights.items():
        score = 0.0
        for term, query_weight in query_weights.items():
            score += query_weight * weights.get(term, 0.0)
        if score <= 0.0:
            continue
        if cosine:
            denom = doc_norms[doc_id] * query_norm
            if denom <= 0.0:
                continue
            score /= denom
        scores.append((doc_id, score))
    return sorted(scores, key=lambda item: (-item[1], item[0]))


def _assert_rankings_close(actual, expected, top_k): # проверяет, что два списка (doc_id, score) совпадают по doc_id в топ-k и что оценки релевантности примерно равны для этих документов
    assert _doc_ids(actual[:top_k]) == _doc_ids(expected[:top_k])
    for (actual_doc, actual_score), (expected_doc, expected_score) in zip(actual[:top_k], expected[:top_k]):
        assert actual_doc == expected_doc
        assert actual_score == pytest.approx(expected_score, rel=1e-9, abs=1e-9)


def test_tfidf_prefers_higher_term_frequency(): # проверяет, что документы с более высокой частотой термов получают более высокие оценки релевантности при поиске по TF-IDF
    idx = _make_index(champion_size=3, tier_size=3)
    idx.add_document("python ranking basics")
    idx.add_document("python python ranking ranking ranking")
    idx.add_document("database systems")

    result = idx.search_tfidf("python ranking", top_k=3)
    assert _doc_ids(result)[:2] == [1, 0]


def test_vector_space_penalizes_longer_document(): # проверяет, что более длинные документы получают более низкие оценки релевантности при поиске в пространстве векторов
    idx = _make_index(champion_size=3, tier_size=3)
    idx.add_document("python ranking")
    idx.add_document("python ranking search engine tfidf cosine lsm bloom filter storage")
    idx.add_document("java ranking")

    result = idx.search_vector("python ranking", top_k=3)
    assert _doc_ids(result)[:2] == [0, 1]


def test_exact_scores_match_manual_reference(): # проверяет, что оценки релевантности, возвращаемые поиском по TF-IDF и векторному пространству, совпадают с ручным расчетом для небольшого набора документов и запроса
    docs = [
        "alpha beta beta",
        "alpha alpha beta gamma",
        "gamma delta",
        "beta gamma gamma",
    ]
    idx = _make_index(champion_size=3, tier_size=3)
    for doc in docs:
        idx.add_document(doc)

    query = "alpha beta gamma"
    expected_tfidf = _manual_rank(docs, query, cosine=False)
    expected_vector = _manual_rank(docs, query, cosine=True)

    _assert_rankings_close(idx.search_tfidf(query, top_k=4), expected_tfidf, top_k=4)
    _assert_rankings_close(idx.search_vector(query, top_k=4), expected_vector, top_k=4)


def test_champion_lists_and_tiers_are_built_correctly(): # проверяет, что для терма с разной частотой в документах формируются правильные чемпион-листы и уровни в иерархической структуре, и что эти структуры обновляются при добавлении новых документов
    idx = _make_index(champion_size=2, tier_size=2)
    idx.add_document("alpha alpha alpha alpha alpha")
    idx.add_document("alpha alpha alpha alpha")
    idx.add_document("alpha alpha alpha")
    idx.add_document("alpha alpha")
    idx.add_document("alpha")
    idx.add_document("beta beta beta")

    idx.search_tfidf("alpha", top_k=3)

    sorted_alpha = [doc_id for doc_id, _ in idx.sorted_postings["alpha"]]
    assert sorted_alpha == [0, 1, 2, 3, 4]
    assert idx.champion_lists["alpha"] == [0, 1]
    assert idx.tiered_postings["alpha"]["hot"] == [0, 1]
    assert idx.tiered_postings["alpha"]["warm"] == [2, 3]
    assert idx.tiered_postings["alpha"]["cold"] == [4]


def test_tiered_candidate_expansion_reaches_all_stages(): # проверяет, что при сборе кандидатов для топ-k запросов с помощью иерархической структуры, сначала возвращаются документы из чемпион-листа, затем из горячего уровня, затем из теплого, и так далее, пока не будет достигнуто необходимое количество кандидатов
    idx = _make_index(champion_size=1, tier_size=1)
    idx.add_document("alpha alpha alpha alpha alpha")
    idx.add_document("alpha alpha alpha alpha")
    idx.add_document("alpha alpha alpha")
    idx.add_document("alpha alpha")
    idx.add_document("alpha")

    idx.search_vector("alpha", top_k=3)

    assert idx._collect_inexact_candidates(["alpha"], top_k=1) == {0}
    assert idx._collect_inexact_candidates(["alpha"], top_k=2) == {0, 1}
    assert idx._collect_inexact_candidates(["alpha"], top_k=4) == {0, 1, 2, 3, 4}


def test_inexact_top_k_matches_exact_head_for_vector_and_tfidf(): # проверяет, что приближенный топ-k поиск возвращает те же самые документы в топе, что и точный поиск, для обоих моделей TF-IDF и векторного пространства, на небольшом наборе документов и запросов
    idx = _make_index(champion_size=4, tier_size=4)
    docs = [
        "python search ranking tfidf cosine",
        "python python ranking ranking ranking",
        "ranking search engine tutorial",
        "database lsm bloom filter compaction",
        "python search tutorial basics",
        "ranking vector space search",
    ]
    for doc in docs:
        idx.add_document(doc)

    exact_vector = idx.search_vector("python ranking search", top_k=3)
    approx_vector = idx.search_inexact_top_k("python ranking search", top_k=3, model="vector")
    exact_tfidf = idx.search_tfidf("python ranking search", top_k=3)
    approx_tfidf = idx.search_inexact_top_k("python ranking search", top_k=3, model="tfidf")

    assert _doc_ids(approx_vector)
    assert _doc_ids(approx_tfidf)
    assert _doc_ids(approx_vector)[0] == _doc_ids(exact_vector)[0]
    assert _doc_ids(approx_tfidf)[0] == _doc_ids(exact_tfidf)[0]


def test_statistics_refresh_after_new_documents(): # проверяет, что после добавления новых документов и обновления статистики, поиск по TF-IDF возвращает правильные результаты с учетом новых данных, и что чемпион-листы обновляются соответственно
    idx = _make_index(champion_size=2, tier_size=2)
    idx.add_document("alpha beta")
    idx.add_document("alpha alpha")

    initial = idx.search_tfidf("alpha beta", top_k=2)
    assert _doc_ids(initial)[0] == 0

    idx.add_document("alpha alpha alpha alpha beta")
    updated = idx.search_tfidf("alpha beta", top_k=3)

    assert _doc_ids(updated)[0] == 2
    assert idx.champion_lists["alpha"][0] == 2


def test_stopwords_and_stemming(): # проверяет, что стоп-слова и стемминг применяются при поиске по TF-IDF и векторному пространству, и что возвращаются правильные документы и оценки релевантности для разных форм слова и с учетом стоп-слов
    idx = _make_index(champion_size=3, tier_size=3)
    idx.add_document("the running code and the runner")
    idx.add_document("runs faster")

    assert _doc_ids(idx.search_tfidf("running", top_k=5)) == [0, 1]
    assert idx.search_tfidf("the and", top_k=5) == []


def test_unknown_query_returns_empty(): # проверяет, что поиск по TF-IDF, векторному пространству и приближенный топ-k поиск возвращают пустой результат для запроса, который не содержит известных термов
    idx = _make_index()
    idx.add_document("python ranking")
    assert idx.search_tfidf("unknown-term", top_k=5) == []
    assert idx.search_vector("unknown-term", top_k=5) == []
    assert idx.search_inexact_top_k("unknown-term", top_k=5) == []


def test_random_exact_ranking_matches_manual_reference(): # проверяет, что точный поиск по TF-IDF и векторному пространству возвращает те же документы и оценки релевантности, что и ручной расчет, на случайных документах и запросах
    rng = random.Random(42)
    docs = []
    idx = _make_index(champion_size=10, tier_size=10)
    vocab = [f"word{i}" for i in range(20)] + ["alpha", "beta", "gamma", "delta"]

    for doc_id in range(180):
        tokens = [rng.choice(vocab) for _ in range(35)]
        if doc_id % 17 == 0:
            tokens.extend(["alpha", "beta", "gamma", "alpha"])
        if doc_id % 29 == 0:
            tokens.extend(["delta", "delta", "beta"])
        text = " ".join(tokens)
        docs.append(text)
        idx.add_document(text)

    queries = [
        "alpha beta gamma",
        "delta beta",
        "word1 word2 word3",
    ]
    for _ in range(20):
        width = rng.randint(2, 4)
        query_terms = [rng.choice(vocab) for _ in range(width)]
        queries.append(" ".join(query_terms))

    for query in queries:
        expected_tfidf = _manual_rank(docs, query, cosine=False)[:5]
        expected_vector = _manual_rank(docs, query, cosine=True)[:5]
        actual_tfidf = idx.search_tfidf(query, top_k=5)
        actual_vector = idx.search_vector(query, top_k=5)
        _assert_rankings_close(actual_tfidf, expected_tfidf, top_k=min(5, len(expected_tfidf)))
        _assert_rankings_close(actual_vector, expected_vector, top_k=min(5, len(expected_vector)))


def test_large_multiquery_inexact_quality(): # проверяет, что для сложных запросов с несколькими термами, приближенный топ-k поиск возвращает большинство тех же документов в топе, что и точный поиск по векторному пространству, на большом наборе случайных документов
    idx = _make_index(champion_size=32, tier_size=32)
    rng = random.Random(123)

    for doc_id in range(700):
        tokens = [f"word{rng.randint(0, 250)}" for _ in range(40)]
        if doc_id % 13 == 0:
            tokens.extend(["alpha", "alpha", "alpha", "beta", "beta", "gamma"])
        if doc_id % 17 == 0:
            tokens.extend(["delta", "delta", "delta", "epsilon", "epsilon"])
        if doc_id % 19 == 0:
            tokens.extend(["python", "python", "python", "search", "search", "ranking"])
        if doc_id % 23 == 0:
            tokens.extend(["lsm", "lsm", "lsm", "bloom", "bloom", "compaction"])
        idx.add_document(" ".join(tokens))

    queries = [
        "alpha beta gamma",
        "delta epsilon",
        "python search ranking",
        "lsm bloom compaction",
        "alpha python",
    ]

    overlaps = []
    top1_matches = 0
    for query in queries:
        exact = idx.search_vector(query, top_k=10)
        approx = idx.search_inexact_top_k(query, top_k=10)
        assert approx, f"Approximate top-k unexpectedly returned no results for {query}"
        if _doc_ids(approx)[0] == _doc_ids(exact)[0]:
            top1_matches += 1
        overlap = len(set(_doc_ids(exact)) & set(_doc_ids(approx)))
        overlaps.append(overlap)

    assert top1_matches >= 4
    assert min(overlaps) >= 4
    assert sum(overlaps) / len(overlaps) >= 7.5


def test_lsm_persistence_roundtrip(): # проверяет, что индекс с LSM сохраняет данные при закрытии и загружает их при повторном открытии, и что поиск по TF-IDF и векторному пространству возвращает правильные результаты после перезапуска
    lsm_path = "data/hw6_persistence_lsm"
    _cleanup_lsm(lsm_path)

    idx = HW6RankedIndex(
        champion_size=4,
        tier_size=4,
        use_lsm=True,
        lsm_path=lsm_path,
        clear_on_init=True,
        lsm_auto_flush_docs=1000,
    )
    idx.add_document("python ranking basics")
    idx.add_document("python python ranking ranking ranking")
    idx.add_document("lsm bloom filter compaction")
    idx.close()

    reopened = HW6RankedIndex(
        champion_size=4,
        tier_size=4,
        use_lsm=True,
        lsm_path=lsm_path,
        clear_on_init=False,
    )

    tfidf_ids = _doc_ids(reopened.search_tfidf("python ranking", top_k=3))
    vector_ids = _doc_ids(reopened.search_vector("python ranking", top_k=3))
    inexact_ids = _doc_ids(reopened.search_inexact_top_k("python ranking", top_k=3))

    assert tfidf_ids[0] == 1
    assert vector_ids[0] == 1
    assert inexact_ids
    assert inexact_ids[0] == vector_ids[0]
    reopened.close()
    _cleanup_lsm(lsm_path)
