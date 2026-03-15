import re
from collections import defaultdict
from pyroaring import BitMap
from nltk.stem import PorterStemmer
from hw1.lsm_v3 import LSMTree

stemmer = PorterStemmer()
STOP_WORDS = {"the", "a", "and", "is", "in", "on", "of", "to", "for", "with", "by"}
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str):
    if not text or not isinstance(text, str):
        return []
    words = _WORD_RE.findall(text.lower())
    words = [w for w in words if w not in STOP_WORDS]
    return [stemmer.stem(w) for w in words]


class InMemoryInvertedIndex:
    def __init__(self):
        self.index = defaultdict(BitMap)
        self.doc_id = 0

    def add_document(self, text: str):
        tokens = tokenize(text)
        for t in tokens:
            self.index[t].add(self.doc_id)
        self.doc_id += 1

    def query_and(self, *terms):
        if not terms:
            return BitMap()
        result = None
        for t in terms:
            key = stemmer.stem(t)
            bm = self.index.get(key, BitMap())
            result = bm if result is None else (result & bm)
        return result or BitMap()

    def query_or(self, *terms):
        result = BitMap()
        for t in terms:
            key = stemmer.stem(t)
            result |= self.index.get(key, BitMap())
        return result

    def query_and_not(self, term1, term2):
        bm1 = self.index.get(stemmer.stem(term1), BitMap())
        bm2 = self.index.get(stemmer.stem(term2), BitMap())
        return bm1 - bm2


class LSMInvertedIndex:
    def __init__(self, mem_limit: int = 1000, lsm_path: str = "lsm_index"):
        self.mem = defaultdict(BitMap)
        self.mem_postings = 0
        self.mem_limit = mem_limit
        self.doc_id = 0
        self.lsm = LSMTree(lsm_path, mem_limit=max(16, mem_limit // 4))

    def add_document(self, text: str):
        tokens = tokenize(text)
        for t in tokens:
            self.mem[t].add(self.doc_id)
            self.mem_postings += 1
        self.doc_id += 1

        if self.mem_postings >= self.mem_limit:
            self._flush()

    def _flush(self):
        if not self.mem:
            return

        for stem_term, bm in self.mem.items():
            existing_blob = self.lsm.get(stem_term)
            if existing_blob:
                try:
                    existing_bm = BitMap.deserialize(existing_blob)
                except Exception:
                    existing_bm = BitMap()
                combined = bm | existing_bm
            else:
                combined = bm
            self.lsm.put(stem_term, combined.serialize())

        self.mem.clear()
        self.mem_postings = 0

    def _get_bitmap(self, term: str):
        key = stemmer.stem(term)
        res = BitMap()
        if key in self.mem:
            res |= self.mem[key]
        blob = self.lsm.get(key)
        if blob:
            try:
                res |= BitMap.deserialize(blob)
            except Exception:
                pass
        return res

    def query_and(self, *terms):
        if not terms:
            return BitMap()
        result = None
        for t in terms:
            bm = self._get_bitmap(t)
            result = bm if result is None else (result & bm)
        return result or BitMap()

    def query_or(self, *terms):
        result = BitMap()
        for t in terms:
            result |= self._get_bitmap(t)
        return result

    def query_and_not(self, term1, term2):
        bm1 = self._get_bitmap(term1)
        bm2 = self._get_bitmap(term2)
        return bm1 - bm2

    def close(self):
        self._flush()
        if hasattr(self.lsm, "close"):
            try:
                self.lsm.close()
            except Exception:
                pass