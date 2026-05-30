import math
import json
import re
import shutil
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer

from hw1.lsm_v3 import LSMTree

stemmer = PorterStemmer()
STOP_WORDS = {
    "the", "a", "and", "is", "in", "on", "of", "to", "for", "with", "by",
    "и", "в", "во", "на", "с", "со", "к", "ко", "по", "из", "у", "за", "под", "о", "об",
}
_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё_-]+")


def _normalize_token(token: str) -> str:
    lowered = token.lower()
    if lowered.isascii() and any("a" <= ch <= "z" for ch in lowered):
        return stemmer.stem(lowered)
    return lowered


def tokenize(text: str, remove_stopwords: bool = True):
    if not text or not isinstance(text, str):
        return []
    tokens = []
    for raw in _WORD_RE.findall(text):
        normalized = _normalize_token(raw)
        if remove_stopwords and normalized in STOP_WORDS:
            continue
        tokens.append(normalized)
    return tokens


class HW6RankedIndex:
    def __init__(
        self,
        champion_size: int = 8,
        tier_size: int = 8,
        use_lsm: bool = True,
        lsm_path: str = "data/hw6_ranked_lsm",
        clear_on_init: bool = False,
        lsm_mem_limit: int = 256,
        lsm_auto_flush_docs: int = 200,
    ):
        self.champion_size = champion_size # размер чемпион-листа
        self.tier_size = tier_size # размер каждого уровня в иерархической структуре

        self.docs = {} # doc_id -> текст документа
        self.doc_tokens = {} # doc_id -> список токенов документа
        self.doc_lengths = {} # doc_id -> количество токенов в документе
        self.postings = defaultdict(dict) # term -> dict of doc_id -> tf (частота терма в документе)
        self.doc_count = 0 # общее количество документов в индексе

        self.df = {} # term -> document frequency (количество документов, содержащих терм)
        self.idf = {} # term -> inverse document frequency (логарифм отношения общего количества документов к количеству документов с термом)
        self.term_weights = {} # term -> dict of doc_id -> weight (вес терма в документе, например, TF-IDF)
        self.doc_norms = {} # doc_id -> норма вектора документа (для косинусного ранжирования)
        self.sorted_postings = {} # term -> список (doc_id, weight) для всех документов с термом, отсортированный по убыванию веса
        self.champion_lists = {} # term -> список doc_id для чемпион-листа терма (топ-k документов с наибольшим весом для терма)
        self.tiered_postings = {} # term -> dict of tier_name -> список doc_id для каждого уровня иерархической структуры (например, "hot", "warm", "cold")
        self._stats_dirty = True # флаг, указывающий, что статистика (df, idf, term_weights, doc_norms, sorted_postings, champion_lists, tiered_postings) устарела и требует обновления
        self.use_lsm = use_lsm # флаг, указывающий, использовать ли LSM-дерево для хранения данных на диске
        self.lsm_path = lsm_path # путь к директории для хранения данных LSM-дерева
        self.lsm_auto_flush_docs = lsm_auto_flush_docs # количество добавленных документов, после которого автоматически вызывается flush() для сохранения данных в LSM-дерево
        self._dirty_terms = set() # множество термов, для которых были изменены постинг-листы и которые требуют сохранения в LSM-дереве при flush()
        self._dirty_docs = set() # множество doc_id документов, которые были добавлены или изменены и требуют сохранения в LSM-дереве при flush()
        self._docs_since_flush = 0 # счетчик добавленных документов с последнего flush()
        self.lsm = None # экземпляр LSM-дерева для хранения данных на диске, если use_lsm=True

        if self.use_lsm:
            if clear_on_init:
                shutil.rmtree(self.lsm_path, ignore_errors=True)
            self.lsm = LSMTree(path=self.lsm_path, mem_limit=lsm_mem_limit)
            self._load_from_lsm()

    @staticmethod
    def _posting_key(term: str):
        return "p:" + term

    @staticmethod
    def _doc_key(doc_id: int):
        return "__doc__:" + str(doc_id)

    @staticmethod
    def _meta_key():
        return "__meta__:doc_count"

    def _serialize_tf_map(self, posting_dict):
        encoded = {}
        for doc_id, tf in posting_dict.items():
            encoded[str(doc_id)] = tf
        return json.dumps(encoded, separators=(",", ":")).encode("utf-8")

    def _deserialize_tf_map(self, blob):
        if blob is None:
            return {}
        try:
            raw = json.loads(blob.decode("utf-8"))
        except Exception:
            return {}

        decoded = {}
        for doc_id_raw, tf_raw in raw.items():
            try:
                doc_id = int(doc_id_raw)
            except Exception:
                continue
            if not isinstance(tf_raw, int):
                continue
            if tf_raw <= 0:
                continue
            decoded[doc_id] = tf_raw
        return decoded

    def _load_from_lsm(self): # загружает данные из LSM-дерева в память при инициализации, восстанавливая документы, постинг-листы и статистику, и устанавливает doc_count на основе сохраненных данных
        if self.lsm is None:
            return

        self.docs.clear()
        self.doc_tokens.clear()
        self.doc_lengths.clear()
        self.postings.clear()
        self.doc_count = 0

        items = self.lsm.range("", "\uffff")
        max_doc_id = -1
        saved_doc_count = None

        for key, value in items:
            if value is None:
                continue
            if key == self._meta_key():
                try:
                    saved_doc_count = int(value.decode("utf-8"))
                except Exception:
                    saved_doc_count = None
                continue
            if key.startswith("__doc__:"):
                doc_id_raw = key.split(":", 1)[1]
                try:
                    doc_id = int(doc_id_raw)
                except Exception:
                    continue
                try:
                    text = value.decode("utf-8")
                except Exception:
                    text = ""
                self.docs[doc_id] = text
                tokens = tokenize(text, remove_stopwords=True)
                self.doc_tokens[doc_id] = tokens
                self.doc_lengths[doc_id] = len(tokens)
                if doc_id > max_doc_id:
                    max_doc_id = doc_id
                continue
            if key.startswith("p:"):
                term = key[2:]
                posting_dict = self._deserialize_tf_map(value)
                if not posting_dict:
                    continue
                for doc_id, tf in posting_dict.items():
                    self.postings[term][doc_id] = tf
                    if doc_id > max_doc_id:
                        max_doc_id = doc_id

        if saved_doc_count is not None:
            self.doc_count = saved_doc_count
        else:
            if max_doc_id >= 0:
                self.doc_count = max_doc_id + 1
        self._stats_dirty = True

    @staticmethod
    def _tf_weight(tf: int) -> float: # преобразует частоту терма в документе (tf) в вес терма
        if tf <= 0:
            return 0.0
        return 1.0 + math.log(tf)

    def flush(self): # сохраняет измененные документы и постинг-листы в LSM-дерево, если use_lsm=True, и обновляет doc_count в метаданных, а также очищает множества _dirty_terms и _dirty_docs и сбрасывает счетчик _docs_since_flush
        if not self.use_lsm:
            return
        if self.lsm is None:
            return

        for doc_id in sorted(self._dirty_docs):
            text = self.docs.get(doc_id, "")
            self.lsm.put(self._doc_key(doc_id), text.encode("utf-8"))

        for term in sorted(self._dirty_terms):
            posting_dict = self.postings.get(term, {})
            blob = self._serialize_tf_map(posting_dict)
            self.lsm.put(self._posting_key(term), blob)

        self.lsm.put(self._meta_key(), str(self.doc_count).encode("utf-8"))
        self._dirty_terms.clear()
        self._dirty_docs.clear()
        self._docs_since_flush = 0

    def close(self):
        if self.use_lsm and self.lsm is not None:
            self.flush()
            self.lsm.close()

    def add_document(self, text: str): # добавляет новый документ в индекс, обновляя словарь документов, постинг-листы, статистику и устанавливая флаги грязных данных для последующего сохранения в LSM-дереве при flush()
        doc_id = self.doc_count
        self.doc_count += 1

        tokens = tokenize(text, remove_stopwords=True)
        tf_counter = Counter(tokens)

        self.docs[doc_id] = text
        self.doc_tokens[doc_id] = tokens
        self.doc_lengths[doc_id] = len(tokens)

        for term, tf in tf_counter.items():
            self.postings[term][doc_id] = tf
            if self.use_lsm:
                self._dirty_terms.add(term)

        self._stats_dirty = True
        if self.use_lsm:
            self._dirty_docs.add(doc_id)
            self._docs_since_flush += 1
            if self._docs_since_flush >= self.lsm_auto_flush_docs:
                self.flush()
        return doc_id

    def _ensure_statistics(self): # проверяет флаг _stats_dirty и, если он установлен, пересчитывает статистику (df, idf, term_weights, doc_norms, sorted_postings, champion_lists, tiered_postings) на основе текущих данных в self.postings и self.docs, и затем сбрасывает флаг _stats_dirty
        if not self._stats_dirty:
            return

        self.df = {}
        self.idf = {}
        self.term_weights = {}
        self.doc_norms = {doc_id: 0.0 for doc_id in self.docs}
        self.sorted_postings = {}
        self.champion_lists = {}
        self.tiered_postings = {}

        total_docs = max(self.doc_count, 1)
        for term, posting_dict in self.postings.items():
            doc_freq = len(posting_dict)
            self.df[term] = doc_freq
            idf = math.log((total_docs + 1.0) / (doc_freq + 1.0)) + 1.0
            self.idf[term] = idf

            weights = {}
            for doc_id, tf in posting_dict.items():
                weight = self._tf_weight(tf) * idf
                weights[doc_id] = weight
                self.doc_norms[doc_id] += weight * weight

            self.term_weights[term] = weights
            ranked_docs = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
            self.sorted_postings[term] = ranked_docs
            champion_cutoff = min(self.champion_size, len(ranked_docs))
            tier_cutoff = min(champion_cutoff + self.tier_size, len(ranked_docs))
            self.champion_lists[term] = [doc_id for doc_id, _ in ranked_docs[:champion_cutoff]]
            self.tiered_postings[term] = {
                "hot": [doc_id for doc_id, _ in ranked_docs[:champion_cutoff]],
                "warm": [doc_id for doc_id, _ in ranked_docs[champion_cutoff:tier_cutoff]],
                "cold": [doc_id for doc_id, _ in ranked_docs[tier_cutoff:]],
            }

        for doc_id, squared_norm in list(self.doc_norms.items()):
            self.doc_norms[doc_id] = math.sqrt(squared_norm)

        self._stats_dirty = False

    def _query_weights(self, query: str): # принимает строку запроса, токенизирует ее, удаляет стоп-слова и применяет стемминг, а затем для каждого терма в запросе вычисляет вес (например, TF-IDF) на основе статистики индекса, возвращая словарь term -> weight для термов в запросе
        terms = tokenize(query, remove_stopwords=True)
        if not terms:
            return terms, {}
        self._ensure_statistics()
        counter = Counter(terms)
        weights = {}
        for term, tf in counter.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            weights[term] = self._tf_weight(tf) * idf
        return terms, weights

    def _score_candidates(self, query_weights, candidate_docs=None, cosine: bool = False):
        scores = defaultdict(float)
        if candidate_docs is not None:
            candidate_docs = set(candidate_docs)

        for term, query_weight in query_weights.items():
            term_weights = self.term_weights.get(term, {})
            if candidate_docs is None:
                iterator = term_weights.items()
            else:
                iterator = ((doc_id, term_weights[doc_id]) for doc_id in candidate_docs if doc_id in term_weights)

            for doc_id, doc_weight in iterator:
                scores[doc_id] += query_weight * doc_weight

        if cosine and scores:
            query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values()))
            if query_norm > 0:
                for doc_id in list(scores):
                    doc_norm = self.doc_norms.get(doc_id, 0.0)
                    if doc_norm > 0:
                        scores[doc_id] /= doc_norm * query_norm
                    else:
                        del scores[doc_id]

        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))

    def search_tfidf(self, query: str, top_k: int = 10): # выполняет поиск по TF-IDF, возвращая список doc_id для топ-k документов с наивысшими оценками релевантности для данного запроса, используя точный поиск по всем документам
        _, query_weights = self._query_weights(query)
        if not query_weights:
            return []
        return self._score_candidates(query_weights, cosine=False)[:top_k]

    def search_vector(self, query: str, top_k: int = 10): # выполняет поиск по векторной модели, возвращая список doc_id для топ-k документов с наивысшими оценками релевантности для данного запроса, используя косинусное ранжирование
        _, query_weights = self._query_weights(query)
        if not query_weights:
            return []
        return self._score_candidates(query_weights, cosine=True)[:top_k]

    def _collect_inexact_candidates(self, query_terms, top_k: int): # для данного списка термов в запросе и количества top_k, собирает множество candidate_docs, сначала добавляя документы из чемпион-листов для термов, затем из следующих уровней иерархической структуры, пока не будет собрано достаточно кандидатов (например, target_size = max(top_k * 3, champion_size + tier_size)), и возвращает множество candidate_docs для последующего оценивания
        self._ensure_statistics()
        candidates = set()
        for term in query_terms:
            candidates.update(self.tiered_postings.get(term, {}).get("hot", []))
        if len(candidates) >= top_k:
            return candidates

        for term in query_terms:
            candidates.update(self.tiered_postings.get(term, {}).get("warm", []))
        if len(candidates) >= top_k:
            return candidates

        target_size = max(top_k * 3, self.champion_size + self.tier_size)
        for term in query_terms:
            for doc_id in self.tiered_postings.get(term, {}).get("cold", []):
                candidates.add(doc_id)
                if len(candidates) >= target_size:
                    break
            if len(candidates) >= target_size:
                break
        return candidates

    def search_inexact_top_k(self, query: str, top_k: int = 10, model: str = "vector"): # выполняет приближенный топ-k поиск, сначала собирая кандидатов из чемпион-листов и уровней иерархической структуры для термов в запросе, а затем оценивая их с помощью TF-IDF или косинусного ранжирования (в зависимости от параметра model), и возвращая список doc_id для топ-k документов с наивысшими оценками релевантности для данного запроса
        query_terms, query_weights = self._query_weights(query)
        if not query_weights:
            return []

        candidates = self._collect_inexact_candidates(query_terms, top_k)
        use_cosine = True
        if model == "tfidf":
            use_cosine = False
        scored = self._score_candidates(query_weights, candidate_docs=candidates, cosine=use_cosine)
        return scored[:top_k]

    def get_document(self, doc_id: int):
        return self.docs.get(doc_id)
