import math
import os
import random
import struct

import pytest

from hw7.kd_tree import DiskKDTree


def _naive_nn(points, query):
    best = None
    for point_id, point in enumerate(points):
        squared_distance = sum((a - b) * (a - b) for a, b in zip(point, query))
        candidate = (point_id, tuple(float(value) for value in point), squared_distance)
        if best is None:
            best = candidate
            continue
        if candidate[2] < best[2] or (math.isclose(candidate[2], best[2], rel_tol=1e-12, abs_tol=1e-12) and candidate[0] < best[0]):
            best = candidate
    if best is None:
        return None
    return best[0], best[1], math.sqrt(best[2])


def test_build_writes_fixed_size_binary_tree_to_disk_and_reopens(tmp_path):
    tree_path = tmp_path / "kd_tree.bin"
    points = [
        (2.0, 3.0),
        (5.0, 4.0),
        (9.0, 6.0),
        (4.0, 7.0),
        (8.0, 1.0),
        (7.0, 2.0),
    ]

    tree = DiskKDTree(str(tree_path), clear_on_init=True)
    tree.build(points)

    assert tree_path.exists()
    assert tree.node_count == len(points)
    assert tree.root_id == len(points) - 1
    assert os.path.getsize(tree_path) == tree.HEADER_SIZE + tree.node_count * tree.node_size

    reopened = DiskKDTree(str(tree_path))
    expected = _naive_nn(points, (9.0, 2.0))
    assert reopened.nearest_neighbor((9.0, 2.0)) == expected


def test_file_header_and_read_counter(tmp_path):
    tree_path = tmp_path / "kd_tree_header.bin"
    points = [
        (0.0, 0.0, 0.0),
        (1.0, 2.0, 3.0),
        (2.0, 4.0, 6.0),
        (3.0, 6.0, 9.0),
        (4.0, 8.0, 12.0),
        (5.0, 10.0, 15.0),
    ]

    tree = DiskKDTree(str(tree_path), clear_on_init=True)
    tree.build(points)

    with open(tree_path, "rb") as file:
        header = file.read(DiskKDTree.HEADER_SIZE)

    magic, version, dims, node_size, node_count, root_id, point_count = struct.unpack(DiskKDTree.HEADER_FORMAT, header)
    assert magic == DiskKDTree.MAGIC
    assert version == DiskKDTree.VERSION
    assert dims == 3
    assert node_size == tree.node_size
    assert node_count == len(points)
    assert root_id == tree.root_id
    assert point_count == len(points)

    reopened = DiskKDTree(str(tree_path))
    assert reopened.read_count == 0
    expected = _naive_nn(points, (4.0, 8.0, 11.0))
    assert reopened.nearest_neighbor((4.0, 8.0, 11.0)) == expected
    assert reopened.read_count > 0


def test_random_queries_match_naive_reference(tmp_path):
    rng = random.Random(42)
    dims = 6
    points = [tuple(rng.uniform(-50.0, 50.0) for _ in range(dims)) for _ in range(256)]
    tree_path = tmp_path / "kd_tree_random.bin"

    tree = DiskKDTree(str(tree_path), clear_on_init=True)
    tree.build(points)
    reopened = DiskKDTree(str(tree_path))

    for _ in range(80):
        query = tuple(rng.uniform(-60.0, 60.0) for _ in range(dims))
        assert reopened.nearest_neighbor(query) == _naive_nn(points, query)


def test_duplicate_points_tie_breaks_by_smallest_id(tmp_path):
    tree_path = tmp_path / "kd_tree_duplicates.bin"
    points = [(1.0, 1.0), (1.0, 1.0), (2.0, 2.0)]

    tree = DiskKDTree(str(tree_path), clear_on_init=True)
    tree.build(points)

    assert tree.nearest_neighbor((1.0, 1.0)) == (0, (1.0, 1.0), 0.0)


def test_empty_tree_returns_none(tmp_path):
    tree_path = tmp_path / "kd_tree_empty.bin"
    tree = DiskKDTree(str(tree_path), dims=3, clear_on_init=True)

    assert tree.nearest_neighbor((1.0, 2.0, 3.0)) is None


def test_dimension_mismatch_raises(tmp_path):
    tree_path = tmp_path / "kd_tree_dims.bin"
    tree = DiskKDTree(str(tree_path), clear_on_init=True)
    tree.build([(1.0, 2.0, 3.0)])

    with pytest.raises(ValueError):
        tree.nearest_neighbor((1.0, 2.0))
