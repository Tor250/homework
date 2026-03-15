import re
from collections import defaultdict
from datetime import datetime
from pyroaring import BitMap
from nltk.stem import PorterStemmer

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
    def __init__(self):
        self.index = defaultdict(BitMap)
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
            raise ValueError(f"Invalid start_date format: {start_date}")
        end_dt = None
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid end_date format: {end_date}")
        self.doc_dates[doc_id] = (start_dt, end_dt)

        tokens = tokenize(text)
        for t in tokens:
            self.index[t].add(doc_id)
        return doc_id

    def query_and(self, *terms):
        if not terms:
            return BitMap()
        result = self.index.get(stemmer.stem(terms[0]), BitMap()).copy()
        for t in terms[1:]:
            result &= self.index.get(stemmer.stem(t), BitMap())
        return result

    def query_or(self, *terms):
        result = BitMap()
        for t in terms:
            result |= self.index.get(stemmer.stem(t), BitMap())
        return result

    def query_and_not(self, include_term: str, exclude_term: str):
        bm1 = self.index.get(stemmer.stem(include_term), BitMap())
        bm2 = self.index.get(stemmer.stem(exclude_term), BitMap())
        return bm1 - bm2

    def _filter_by_date(self, doc_ids: BitMap, start: datetime, end: datetime, mode: str) -> BitMap:
        filtered = BitMap()
        for doc_id in doc_ids:
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
            return self.index.get(stemmer.stem(expr), BitMap()).copy()
        
        if not isinstance(expr, tuple) or len(expr) == 0:
            return BitMap()
        
        op = expr[0]
        
        if op == "TERM":
            term = expr[1]
            return self.index.get(stemmer.stem(term), BitMap()).copy()
        
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
                doc_ids = self.index.get(stemmer.stem(term_expr), BitMap()).copy()
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
        return len(self.index.get(stemmer.stem(term), BitMap()))