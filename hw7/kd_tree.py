import math
import os
import struct
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class _StoredNode:
    point_id: int
    coords: Tuple[float, ...]
    axis: int
    left_id: int
    right_id: int


class DiskKDTree:
    MAGIC = b"KD7TREE1"
    VERSION = 1
    HEADER_FORMAT = "<8sIIIQqQ"
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

    def __init__(self, path: str = "data/hw7_kdtree.bin", dims: Optional[int] = None, clear_on_init: bool = False):
        self.path = path
        self.dims = dims
        self.node_size = self._calc_node_size(dims) if dims is not None else None
        self.node_count = 0
        self.point_count = 0
        self.root_id = -1
        self.read_count = 0
        self.write_count = 0

        if clear_on_init:
            self._remove_existing_path()
        elif os.path.exists(self.path):
            self._load_metadata()

    @classmethod
    def _calc_node_size(cls, dims: Optional[int]) -> int:
        if dims is None:
            return 0
        return struct.calcsize(f"<BIqqq{dims}d")

    @property
    def _node_format(self) -> str:
        if self.dims is None:
            raise ValueError("Tree dimensions are unknown")
        return f"<BIqqq{self.dims}d"

    def _remove_existing_path(self) -> None:
        if not os.path.exists(self.path):
            return
        if os.path.isdir(self.path):
            for root, dirs, files in os.walk(self.path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.path)
        else:
            os.remove(self.path)

    def _load_metadata(self) -> None:
        with open(self.path, "rb") as file:
            header = file.read(self.HEADER_SIZE)
        if len(header) != self.HEADER_SIZE:
            raise ValueError("Invalid kd-tree file: truncated header")

        magic, version, dims, node_size, node_count, root_id, point_count = struct.unpack(self.HEADER_FORMAT, header)
        if magic != self.MAGIC:
            raise ValueError("Invalid kd-tree file signature")
        if version != self.VERSION:
            raise ValueError(f"Unsupported kd-tree version: {version}")

        expected_node_size = self._calc_node_size(dims)
        if node_size != expected_node_size:
            raise ValueError("Invalid kd-tree file: node size mismatch")

        self.dims = dims
        self.node_size = node_size
        self.node_count = node_count
        self.root_id = root_id
        self.point_count = point_count

    def _normalize_point(self, point: Sequence[float]) -> Tuple[float, ...]:
        coords = tuple(float(value) for value in point)
        if self.dims is None:
            self.dims = len(coords)
            self.node_size = self._calc_node_size(self.dims)
        elif len(coords) != self.dims:
            raise ValueError(f"Point dimension mismatch: expected {self.dims}, got {len(coords)}")
        return coords

    def reset_stats(self) -> None:
        self.read_count = 0
        self.write_count = 0

    def _write_header(self, file) -> None:
        header = struct.pack(
            self.HEADER_FORMAT,
            self.MAGIC,
            self.VERSION,
            self.dims,
            self.node_size,
            self.node_count,
            self.root_id,
            self.point_count,
        )
        file.seek(0)
        file.write(header)

    def _write_node(self, file, node_id: int, node: _StoredNode) -> None:
        # Узлы пишутся сразу в файл, без буфера на весь индекс.
        file.seek(self._node_offset(node_id))
        file.write(
            struct.pack(
                self._node_format,
                1 if node.left_id < 0 and node.right_id < 0 else 0,
                node.axis,
                node.point_id,
                node.left_id,
                node.right_id,
                *node.coords,
            )
        )
        self.write_count += 1

    def _build_subtree(self, points: list[tuple[int, Tuple[float, ...]]], depth: int, file) -> int:
        if not points:
            return -1

        axis = depth % self.dims
        # Разрез по медиане держит дерево ровным.
        ordered = sorted(points, key=lambda item: (item[1][axis], item[0]))
        middle = len(ordered) // 2
        point_id, coords = ordered[middle]

        left_id = self._build_subtree(ordered[:middle], depth + 1, file)
        right_id = self._build_subtree(ordered[middle + 1 :], depth + 1, file)

        node_id = self.node_count
        self.node_count += 1
        self._write_node(file, node_id, _StoredNode(point_id=point_id, coords=coords, axis=axis, left_id=left_id, right_id=right_id))
        return node_id

    def build(self, points: Iterable[Sequence[float]]):
        raw_points = list(points)
        if raw_points:
            first_dim = len(raw_points[0])
            if self.dims in (None, 0):
                self.dims = first_dim
                self.node_size = self._calc_node_size(self.dims)
            elif first_dim != self.dims:
                raise ValueError(f"Point dimension mismatch: expected {self.dims}, got {first_dim}")

        materialized = [self._normalize_point(point) for point in raw_points]
        self.point_count = len(materialized)

        if not materialized:
            if self.dims is None:
                self.dims = 0
                self.node_size = self._calc_node_size(self.dims)
            elif self.node_size is None:
                self.node_size = self._calc_node_size(self.dims)
            self.node_count = 0
            self.root_id = -1
            self.reset_stats()
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "wb+") as file:
                file.truncate(self.HEADER_SIZE)
                self._write_header(file)
            return self

        if self.dims is None:
            self.dims = len(materialized[0])
        if self.node_size is None:
            self.node_size = self._calc_node_size(self.dims)

        indexed_points = [(index, point) for index, point in enumerate(materialized)]
        self.node_count = 0
        self.root_id = -1
        self.reset_stats()
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "wb+") as file:
            file.truncate(self.HEADER_SIZE)
            self.root_id = self._build_subtree(indexed_points, 0, file)
            self._write_header(file)
        return self

    def _node_offset(self, node_id: int) -> int:
        if self.node_size is None:
            raise ValueError("Tree is not built")
        return self.HEADER_SIZE + node_id * self.node_size

    def _read_node(self, file, node_id: int) -> _StoredNode:
        if node_id < 0:
            raise ValueError("Negative node id")
        file.seek(self._node_offset(node_id))
        raw = file.read(self.node_size)
        if len(raw) != self.node_size:
            raise ValueError(f"Failed to read node {node_id}")
        self.read_count += 1
        unpacked = struct.unpack(self._node_format, raw)
        _, axis, point_id, left_id, right_id, *coords = unpacked
        return _StoredNode(point_id=point_id, coords=tuple(coords), axis=axis, left_id=left_id, right_id=right_id)

    @staticmethod
    def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
        return sum((a - b) * (a - b) for a, b in zip(left, right))

    @staticmethod
    def _better(candidate_sq: float, candidate_id: int, best_sq: float, best_id: int) -> bool:
        if candidate_sq < best_sq:
            return True
        if math.isclose(candidate_sq, best_sq, rel_tol=1e-12, abs_tol=1e-12):
            return candidate_id < best_id
        return False

    def _search(self, file, node_id: int, query: Tuple[float, ...], best):
        if node_id < 0:
            return best

        node = self._read_node(file, node_id)
        candidate_sq = self._squared_distance(query, node.coords)
        if best is None or self._better(candidate_sq, node.point_id, best[2], best[0]):
            best = (node.point_id, node.coords, candidate_sq)

        axis = node.axis
        delta = query[axis] - node.coords[axis]
        near_child, far_child = (node.left_id, node.right_id) if delta <= 0 else (node.right_id, node.left_id)

        # Сначала идем в ближнюю ветку, дальнюю режем по плоскости.
        best = self._search(file, near_child, query, best)
        if best is None or delta * delta <= best[2]:
            best = self._search(file, far_child, query, best)
        return best

    def nearest_neighbor(self, query_point: Sequence[float]):
        if self.root_id < 0 or self.node_count == 0:
            return None

        query = self._normalize_point(query_point)
        with open(self.path, "rb") as file:
            best = self._search(file, self.root_id, query, None)
        if best is None:
            return None
        return best[0], best[1], math.sqrt(best[2])

    def close(self):
        return None
