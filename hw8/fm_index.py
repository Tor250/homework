from collections import Counter
from typing import Dict, List, Optional


def _build_suffix_array(text: str) -> List[int]:
    text_with_sentinel = text + "\0"
    size = len(text_with_sentinel)

    if size == 1:
        return [0]

    suffix_array = list(range(size))
    ranks = [ord(char) for char in text_with_sentinel]
    next_ranks = [0] * size
    step = 1

    while True:
        # Сортировка удвоением держит код простым.
        suffix_array.sort(
            key=lambda index: (
                ranks[index],
                ranks[index + step] if index + step < size else -1,
            )
        )

        next_ranks[suffix_array[0]] = 0
        classes = 0
        for position in range(1, size):
            current = suffix_array[position]
            previous = suffix_array[position - 1]
            current_key = (
                ranks[current],
                ranks[current + step] if current + step < size else -1,
            )
            previous_key = (
                ranks[previous],
                ranks[previous + step] if previous + step < size else -1,
            )
            if current_key != previous_key:
                classes += 1
            next_ranks[current] = classes

        ranks, next_ranks = next_ranks, ranks
        if classes == size - 1:
            break
        step <<= 1

    return suffix_array


class FMIndex:
    def __init__(self, text: Optional[str] = None, checkpoint_step: int = 32):
        self.checkpoint_step = max(1, int(checkpoint_step))
        self.text_length = 0
        self.suffix_array: List[int] = []
        self.bwt: List[str] = []
        self._symbols: List[str] = []
        self._symbol_to_index: Dict[str, int] = {}
        self._first_occurrence: Dict[str, int] = {}
        self._checkpoint_counts: Dict[str, List[int]] = {}

        if text is not None:
            self.build(text)

    def build(self, text: str):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if "\0" in text:
            raise ValueError("text must not contain NUL characters")

        self.text_length = len(text)
        self.suffix_array = _build_suffix_array(text)
        self.bwt = [text[index - 1] if index > 0 else "\0" for index in self.suffix_array]

        counts = Counter(self.bwt)
        self._symbols = sorted(counts)
        self._symbol_to_index = {symbol: idx for idx, symbol in enumerate(self._symbols)}

        self._first_occurrence = {}
        running = 0
        for symbol in self._symbols:
            self._first_occurrence[symbol] = running
            running += counts[symbol]

        # Контрольные точки считают occ без полного прохода по BWT.
        block_count = (len(self.bwt) + self.checkpoint_step - 1) // self.checkpoint_step
        self._checkpoint_counts = {symbol: [0] * block_count for symbol in self._symbols}
        current_counts = [0] * len(self._symbols)

        for position, symbol in enumerate(self.bwt):
            if position % self.checkpoint_step == 0:
                block = position // self.checkpoint_step
                for idx, value in enumerate(current_counts):
                    self._checkpoint_counts[self._symbols[idx]][block] = value
            current_counts[self._symbol_to_index[symbol]] += 1

        return self

    def _occ(self, symbol: str, position: int) -> int:
        if position < 0:
            return 0

        counts = self._checkpoint_counts.get(symbol)
        symbol_index = self._symbol_to_index.get(symbol)
        if counts is None or symbol_index is None:
            return 0

        block = position // self.checkpoint_step
        count = counts[block]
        start = block * self.checkpoint_step
        stop = min(position, len(self.bwt) - 1)
        for index in range(start, stop + 1):
            if self.bwt[index] == symbol:
                count += 1
        return count

    def search(self, pattern: str):
        if not isinstance(pattern, str):
            raise TypeError("pattern must be a string")
        if "\0" in pattern:
            raise ValueError("pattern must not contain NUL characters")
        if not pattern:
            return list(range(self.text_length))
        if not self.suffix_array:
            return []

        left = 0
        right = len(self.bwt)

        for symbol in reversed(pattern):
            first = self._first_occurrence.get(symbol)
            if first is None:
                return []

            # Обратный поиск сжимает интервал справа налево.
            left = first + self._occ(symbol, left - 1)
            right = first + self._occ(symbol, right - 1)
            if left >= right:
                return []

        positions = [index for index in self.suffix_array[left:right] if index < self.text_length]
        positions.sort()
        return positions

    def count(self, pattern: str) -> int:
        return len(self.search(pattern))

    locate = search
