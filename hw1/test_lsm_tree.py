import os
import shutil
import random
from lsm_v3 import LSMTree, SSTable



def setup_data(path="test_data"):
    if os.path.exists(path):
        shutil.rmtree(path)

def rand_key_int(max_val=1_000_000):
    return f"k{random.randint(0, max_val)}"


def rand_val(length=16):
    return bytes(random.getrandbits(8) for _ in range(length))

def test_lsm_put_get_basic():
    setup_data()

    tree = LSMTree("test_data", mem_limit=5)

    tree.put("a", b"1")
    tree.put("b", b"2")
    tree.put("c", b"3")

    tree.close()

    assert tree.get("a") == b"1"
    assert tree.get("b") == b"2"
    assert tree.get("c") == b"3"
    assert tree.get("x") is None


def test_lsm_range_basic():
    setup_data()

    tree = LSMTree("test_data", mem_limit=5)

    for i in range(20):
        tree.put(f"k{i}", str(i).encode())

    tree.close()

    result = tree.range("k5", "k15")
    keys = [k for k, _ in result]

    assert "k5" in keys
    assert "k15" in keys
    assert "k4" not in keys
    assert "k16" not in keys

def test_multi_level_flush_and_factor():
    setup_data()

    factor = 2
    tree = LSMTree("test_data", mem_limit=2, level_factor=factor)

    for i in range(20):
        tree.put(f"k{i}", str(i).encode())

    tree.close()
    assert len(tree.levels) > 1
    prev_keys = None

    for lvl in range(len(tree.levels)):

        curr_keys = sum(len(sst.index) for sst in tree.levels[lvl])

        if curr_keys == 0:
            continue

        if prev_keys is not None:
            assert curr_keys <= prev_keys * factor + factor

        prev_keys = curr_keys



def test_random_stress():
    setup_data()

    tree = LSMTree("test_data", mem_limit=50, level_factor=5)

    data = {}

    for _ in range(500):
        k = rand_key_int()
        v = rand_val()

        data[k] = v
        tree.put(k, v)

    tree.close()

    sample_keys = random.sample(list(data.keys()), 50)

    for k in sample_keys:
        val = tree.get(k)
        assert val == data[k], f"Mismatch for key {k}"

    sorted_keys = sorted(data.keys(), key=lambda k: int(k[1:]))

    for _ in range(20):

        start_idx = random.randint(0, len(sorted_keys) - 10)
        end_idx = min(start_idx + 10, len(sorted_keys) - 1)

        start_key = sorted_keys[start_idx]
        end_key = sorted_keys[end_idx]

        result = tree.range(start_key, end_key)

        result_dict = dict(result)

        for k in sorted_keys[start_idx:end_idx + 1]:
            assert k in result_dict
            assert result_dict[k] == data[k]

def test_restart_persistence():
    setup_data()

    tree = LSMTree("test_data", mem_limit=10)

    for i in range(50):
        tree.put(f"k{i}", str(i).encode())

    tree.close()
    tree2 = LSMTree("test_data")

    for i in range(50):
        val = tree2.get(f"k{i}")
        assert val == str(i).encode()

    shutil.rmtree("test_data", ignore_errors=True)





def test_update_deduplication_during_compaction():
    setup_data()
    tree = LSMTree("test_data", mem_limit=2, level_factor=2)
    tree.put("k1", b"v1_initial")
    tree.put("k2", b"v2")
    tree.put("k1", b"v1_updated")
    tree.put("k3", b"v3")
    tree.put("k4", b"v4")
    tree.put("k5", b"v5")
    
    assert tree.get("k1") == b"v1_updated", "Дедупликация при компакции не работает!"
    assert tree.get("k2") == b"v2"
    assert tree.get("k3") == b"v3"
    
    tree.close()
    shutil.rmtree("test_data", ignore_errors=True)
    print("✓ test_update_deduplication_during_compaction passed")


def test_delete_and_tombstone_propagation():
    setup_data()
    tree = LSMTree("test_data", mem_limit=5)
    for i in range(10):
        tree.put(f"k{i}", f"v{i}".encode())
    tree.delete("k3")
    tree.put("k5", b"v5_updated")
    tree.delete("k5")
    assert tree.get("k3") is None, "Удалённый ключ в memtable не скрыт"
    assert tree.get("k5") is None, "Удаление после обновления не сработало"
    assert tree.get("k7") == b"v7", "Неудалённый ключ пропал"
    tree.close()
    tree2 = LSMTree("test_data", mem_limit=5)
    assert tree2.get("k3") is None, "Tombstone не пережил перезапуск (удаление из memtable)"
    assert tree2.get("k5") is None, "Tombstone не пережил перезапуск (удаление после update)"
    assert tree2.get("k7") == b"v7", "Неудалённый ключ пропал после перезапуска"
    result = tree2.range("k0", "k9")
    keys = [k for k, _ in result]
    assert "k3" not in keys, "Удалённый ключ попал в range()"
    assert "k5" not in keys, "Удалённый ключ попал в range()"
    assert "k7" in keys, "Неудалённый ключ пропал из range()"
    
    tree2.close()
    shutil.rmtree("test_data", ignore_errors=True)
    print("✓ test_delete_and_tombstone_propagation passed")


def test_mixed_key_types_range_query():
    setup_data()
    tree = LSMTree("test_data", mem_limit=10)
    test_data = [
        ("k10", b"a"), ("k2", b"b"), ("k100", b"c"),
        ("user:5", b"d"), ("doc_abc", b"e"), ("z", b"f"),
    ]
    
    for k, v in test_data:
        tree.put(k, v)
    try:
        result = tree.range("k1", "z")
        result_keys = [k for k, _ in result]
        for k, _ in test_data:
            if "k1" <= k <= "z" or SSTable.key_sort("k1") <= SSTable.key_sort(k) <= SSTable.key_sort("z"):
                assert k in result_keys, f"Ключ {k} пропал из диапазона"
        for k, expected_v in test_data:
            assert tree.get(k) == expected_v, f"get({k}) вернул неверное значение"
            
    except TypeError as e:
        raise AssertionError(f"TypeError при range-запросе с разнотипными ключами: {e}")
    
    tree.close()
    shutil.rmtree("test_data", ignore_errors=True)
    print("✓ test_mixed_key_types_range_query passed")


def test_bloom_filter_persistence_across_runs():
    setup_data()
    tree = LSMTree("test_data", mem_limit=5)
    keys = [f"k{i}" for i in range(50)]
    for k in keys:
        tree.put(k, b"value")
    tree.close()
    tree2 = LSMTree("test_data", mem_limit=5)
    found = 0
    for k in keys:
        if tree2.get(k) == b"value":
            found += 1
    assert found >= len(keys) * 0.95, f"Потеряно {len(keys) - found} ключей после перезапуска (BloomFilter broken?)"
    tree2.report_stats()
    
    tree2.close()
    setup_data()
    print("✓ test_bloom_filter_persistence_across_runs passed")

if __name__ == "__main__":
    test_lsm_put_get_basic()
    test_lsm_range_basic()
    test_multi_level_flush_and_factor()
    test_random_stress()
    test_restart_persistence()
    test_mixed_key_types_range_query()
    test_update_deduplication_during_compaction()
    test_bloom_filter_persistence_across_runs()
    test_delete_and_tombstone_propagation()