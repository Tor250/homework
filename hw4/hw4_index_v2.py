import re
import shutil
from datetime import datetime
from collections import defaultdict
from pyroaring import BitMap
from nltk.stem import PorterStemmer
from hw1.lsm_v3 import LSMTree

stemmer = PorterStemmer()
STOP_WORDS = {"the", "a", "and", "is", "in", "on", "of", "to", "for", "with", "by"}
_WORD_RE = re.compile(r"[A-Za-z0-9_-]+")

def tokenize(text: str):
    if not text or not isinstance(text, str):
        return []
    words = _WORD_RE.findall(text.lower())
    words = [w for w in words if w not in STOP_WORDS]
    return [stemmer.stem(w) for w in words]


class HW4InvertedIndex:
    def __init__(self, lsm_path='data/lsm_index', clear_on_init=False):
        if clear_on_init:
            shutil.rmtree(lsm_path, ignore_errors=True)
        
        self.lsm_path = lsm_path
        self.lsm = LSMTree(path=lsm_path)
        self.docs = {}
        self.doc_dates = {}
        self.doc_count = 0

    def add_document(self, text: str, start_date: str, end_date: str = None):
        doc_id = self.doc_count
        self.doc_count += 1
        self.docs[doc_id] = text

        try:
            start_dt = datetime.fromisoformat(start_date)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid start_date: {start_date}")
        end_dt = None
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid end_date: {end_date}")
        self.doc_dates[doc_id] = (start_dt, end_dt)

        tokens = tokenize(text)
        for t in tokens:
            current = self._get_term_bitmap(t)
            current.add(doc_id)
            self.lsm.put(t, current.serialize())
        return doc_id


    def _get_term_bitmap(self, term: str) -> BitMap:
        stemmed = stemmer.stem(term)
        val = self.lsm.get(stemmed)
        if val is None:
            return BitMap()
        try:
            return BitMap.deserialize(val)
        except Exception:
            return BitMap()


    def query_and(self, *terms):
        if not terms:
            return BitMap()
        result = self._get_term_bitmap(terms[0]).copy()
        for t in terms[1:]:
            result &= self._get_term_bitmap(t)
        return result

    def query_or(self, *terms):
        result = BitMap()
        for t in terms:
            result |= self._get_term_bitmap(t)
        return result

    def query_and_not(self, include_term: str, exclude_term: str):
        return self._get_term_bitmap(include_term) - self._get_term_bitmap(exclude_term)


    def _filter_by_date(self, doc_ids: BitMap, start: datetime, end: datetime, mode: str) -> BitMap:
        filtered = BitMap()
        for doc_id in doc_ids:
            if doc_id not in self.doc_dates:
                continue
            doc_start, doc_end = self.doc_dates[doc_id]
            effective_end = doc_end if doc_end else datetime.max
            if mode == "valid":
                if doc_start <= end and effective_end >= start:
                    filtered.add(doc_id)
            elif mode == "appeared":
                if start <= doc_start <= end:
                    filtered.add(doc_id)
            else:
                raise ValueError(f"Unknown mode: {mode}")
        return filtered

    def query_and_date(self, terms=None, start: str = None, end: str = None, mode: str = "valid"):
        dt_start = datetime.fromisoformat(start) if start else None
        dt_end = datetime.fromisoformat(end) if end else None

        if terms:
            doc_ids = self.query_and(*terms)
        else:
            doc_ids = BitMap(range(self.doc_count))

        if dt_start is not None or dt_end is not None:
            effective_start = dt_start or datetime.min
            effective_end = dt_end or datetime.max
            return self._filter_by_date(doc_ids, effective_start, effective_end, mode)
        return doc_ids

    def query_complex(self, expr, start=None, end=None, mode="valid"):
        dt_start = datetime.fromisoformat(start) if start else None
        dt_end = datetime.fromisoformat(end) if end else None
        result = self._eval_expr(expr)
        if dt_start or dt_end:
            effective_start = dt_start or datetime.min
            effective_end = dt_end or datetime.max
            result = self._filter_by_date(result, effective_start, effective_end, mode)
        return result

    def query_boolean(self, expr, start=None, end=None, mode="valid"):
        res = self.query_complex(expr)
        if start or end:
            dt_start = datetime.fromisoformat(start) if start else datetime.min
            dt_end = datetime.fromisoformat(end) if end else datetime.max
            return self._filter_by_date(res, dt_start, dt_end, mode)
        return res

    def _eval_expr(self, expr):
        if isinstance(expr, str):
            return self._get_term_bitmap(expr).copy()
        if not isinstance(expr, tuple) or len(expr) == 0:
            return BitMap()
        op = expr[0]

        if op == "TERM":
            return self._get_term_bitmap(expr[1]).copy()
        elif op == "AND":
            if len(expr) < 2:
                return BitMap(range(self.doc_count))
            result = self._eval_expr(expr[1]).copy()
            for subexpr in expr[2:]:
                result &= self._eval_expr(subexpr)
            return result
        elif op == "OR":
            result = BitMap()
            for subexpr in expr[1:]:
                result |= self._eval_expr(subexpr)
            return result
        elif op == "NOT":
            if len(expr) != 2:
                raise ValueError("NOT requires exactly one argument")
            all_docs = BitMap(range(self.doc_count))
            return all_docs - self._eval_expr(expr[1])
        elif op == "AND_NOT":
            if len(expr) != 3:
                raise ValueError("AND_NOT requires two arguments")
            return self._eval_expr(expr[1]) - self._eval_expr(expr[2])
        elif op == "DATE":
            if len(expr) != 5:
                raise ValueError("DATE requires start, end, mode")
            _, start_str, end_str, mode = expr
            dt_start = datetime.fromisoformat(start_str) if start_str else None
            dt_end = datetime.fromisoformat(end_str) if end_str else None
            all_docs = BitMap(range(self.doc_count))
            effective_start = dt_start or datetime.min
            effective_end = dt_end or datetime.max
            return self._filter_by_date(all_docs, effective_start, effective_end, mode)
        elif op == "AND_DATE":
            if len(expr) < 5:
                raise ValueError("AND_DATE requires term_expr, start, end, mode")
            term_expr = expr[1]
            start_str, end_str, mode = expr[2], expr[3], expr[4]
            if isinstance(term_expr, str):
                doc_ids = self._get_term_bitmap(term_expr).copy()
            else:
                doc_ids = self._eval_expr(term_expr)
            dt_start = datetime.fromisoformat(start_str) if start_str else None
            dt_end = datetime.fromisoformat(end_str) if end_str else None
            effective_start = dt_start or datetime.min
            effective_end = dt_end or datetime.max
            return self._filter_by_date(doc_ids, effective_start, effective_end, mode)
        else:
            raise ValueError(f"Unknown operator: {op}")

    def get_document_count(self):
        return self.doc_count

    def get_term_frequency(self, term: str):
        return len(self._get_term_bitmap(term))

    def close(self):
        if hasattr(self.lsm, 'close'):
            self.lsm.close()

    def cleanup(self):
        shutil.rmtree(self.lsm_path, ignore_errors=True)