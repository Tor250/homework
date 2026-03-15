import os
import struct
import bisect
import hashlib

TOMBSTONE = b'\x00'


class BloomFilter:
    def __init__(self, size=8192, num_hashes=4):
        self.size = size
        self.num_hashes = num_hashes
        self.bits = bytearray((size + 7) // 8)

    def _hash(self, key, seed):
        h = hashlib.md5(f"{key}{seed}".encode()).hexdigest()
        return int(h, 16) % self.size

    def add(self, key):
        for i in range(self.num_hashes):
            pos = self._hash(key, i)
            self.bits[pos // 8] |= 1 << (pos % 8)

    def might_contain(self, key):
        for i in range(self.num_hashes):
            pos = self._hash(key, i)
            if not (self.bits[pos // 8] & (1 << (pos % 8))):
                return False
        return True

    def serialize(self):
        return struct.pack('II', self.size, self.num_hashes) + bytes(self.bits)

    @classmethod
    def deserialize(cls, data):
        size, num_hashes = struct.unpack('II', data[:8])
        bits = data[8:]
        bf = cls(size=size, num_hashes=num_hashes)
        bf.bits = bytearray(bits)
        return bf


class SSTable:
    def __init__(self, filepath, data=None):
        self.bloom_checks = 0
        self.bloom_rejected = 0
        self.filepath = filepath
        self.index = []
        self.bloom = None
        self.sorted_keys = []
        if data:
            self._build(data)
        else:
            self._load()

    @staticmethod
    def key_sort(k):
        if isinstance(k, str) and k.startswith("k") and k[1:].isdigit():
            return (0, int(k[1:]))
        return (1, k)

    def _build(self, data):
        unique = {}
        for k, v in data:
            unique[k] = v
        items = list(unique.items())
        items_sorted = sorted(items, key=lambda kv: self.key_sort(kv[0]))
        self.bloom = BloomFilter()
        offsets = []
        with open(self.filepath, 'wb') as f:
            for k, v in items_sorted:
                self.bloom.add(k)
                key_b = k.encode() if isinstance(k, str) else k
                val_b = v
                offsets.append(f.tell())
                f.write(struct.pack('I', len(key_b)))
                f.write(key_b)
                f.write(struct.pack('I', len(val_b)))
                f.write(val_b)
        self.index = list(zip([k for k, _ in items_sorted], offsets))
        self.sorted_keys = [self.key_sort(k) for k, _ in self.index]
        index_data = struct.pack('I', len(self.index))
        for k, off in self.index:
            kb = k.encode()
            index_data += struct.pack('I', len(kb)) + kb + struct.pack('Q', off)

        bloom_bytes = self.bloom.serialize()

        with open(self.filepath, 'ab') as f:
            f.write(index_data)
            f.write(struct.pack('I', len(index_data)))
            f.write(bloom_bytes)
            f.write(struct.pack('I', len(bloom_bytes)))

    def get(self, key):
        if self.bloom:
            self.bloom_checks += 1
            if not self.bloom.might_contain(key):
                self.bloom_rejected += 1
                return (False, None)

        target = self.key_sort(key)
        pos = bisect.bisect_left(self.sorted_keys, target)
        if pos >= len(self.sorted_keys) or self.index[pos][0] != key:
            return (False, None)

        off = self.index[pos][1]
        with open(self.filepath, 'rb') as f:
            f.seek(off)
            klen = struct.unpack('I', f.read(4))[0]
            f.seek(klen, os.SEEK_CUR)
            vlen = struct.unpack('I', f.read(4))[0]
            val = f.read(vlen)
            if val == TOMBSTONE:
                return (True, None)
            return (True, val)

    def _load(self):
        if not os.path.exists(self.filepath):
            return
        with open(self.filepath, 'rb') as f:
            f.seek(-4, os.SEEK_END)
            bloom_size = struct.unpack('I', f.read(4))[0]
            f.seek(-4 - bloom_size, os.SEEK_END)
            bloom_data = f.read(bloom_size)
            self.bloom = BloomFilter.deserialize(bloom_data)
            f.seek(-4 - bloom_size - 4, os.SEEK_END)
            index_size = struct.unpack('I', f.read(4))[0]
            f.seek(-4 - bloom_size - 4 - index_size, os.SEEK_END)
            count = struct.unpack('I', f.read(4))[0]
            self.index = []
            for _ in range(count):
                klen = struct.unpack('I', f.read(4))[0]
                key = f.read(klen).decode()
                offset = struct.unpack('Q', f.read(8))[0]
                self.index.append((key, offset))
            self.sorted_keys = [self.key_sort(k) for k, _ in self.index]

    def range(self, start, end):
        start_val = self.key_sort(start)
        end_val = self.key_sort(end)
        left = bisect.bisect_left(self.sorted_keys, start_val)
        right = bisect.bisect_right(self.sorted_keys, end_val)
        res = []
        with open(self.filepath, 'rb') as f:
            for i in range(left, right):
                k, off = self.index[i]
                f.seek(off)
                klen = struct.unpack('I', f.read(4))[0]
                f.seek(klen, os.SEEK_CUR)
                vlen = struct.unpack('I', f.read(4))[0]
                val = f.read(vlen)
                if val == TOMBSTONE:
                    res.append((k, None))
                else:
                    res.append((k, val))
        return res

    def cleanup(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)


class LSMTree:
    def __init__(self, path='data', mem_limit=100, level_factor=10):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.memtable = {}
        self.mem_limit = mem_limit
        self.level_factor = level_factor
        self.levels = []
        self._load_existing()

    def _load_existing(self):
        files = [f for f in os.listdir(self.path) if f.endswith(".sst")]
        levels = {}
        for f in files:
            name = f.replace(".sst", "")
            level, idx = name.split("_")
            level = int(level.replace("level", ""))
            idx = int(idx)
            if level not in levels:
                levels[level] = []
            levels[level].append((idx, SSTable(os.path.join(self.path, f))))
        if not levels:
            return
        max_level = max(levels.keys())
        for i in range(max_level + 1):
            if i not in levels:
                self.levels.append([])
                continue
            sorted_tables = sorted(levels[i], key=lambda x: x[0])
            self.levels.append([sst for _, sst in sorted_tables])

    def _sst_filename(self, level, idx):
        return os.path.join(self.path, f"level{level}_{idx}.sst")

    def put(self, key, value):
        self.memtable[key] = value
        if len(self.memtable) >= self.mem_limit:
            self._flush()

    def delete(self, key):
        self.put(key, TOMBSTONE)

    def _flush(self):
        if not self.memtable:
            return
        if not self.levels:
            self.levels.append([])
        idx = len(self.levels[0])
        filepath = self._sst_filename(0, idx)
        sst = SSTable(filepath, list(self.memtable.items()))
        self.levels[0].append(sst)
        self.memtable.clear()
        self._compact_level(0)

    def get(self, key):
        if key in self.memtable:
            v = self.memtable[key]
            return None if v == TOMBSTONE else v
        for level in self.levels:
            for sst in reversed(level):
                found, val = sst.get(key)
                if found:
                    return val
        return None

    def range(self, start, end):
        result = {}
        deleted = set()

        start_val = SSTable.key_sort(start)
        end_val = SSTable.key_sort(end)

        for k, v in self.memtable.items():
            ks = SSTable.key_sort(k)
            try:
                in_range = start_val <= ks <= end_val
            except TypeError:
                in_range = str(start_val) <= str(ks) <= str(end_val)
            if not in_range:
                continue
            if v == TOMBSTONE:
                deleted.add(k)
            else:
                result[k] = v

        for level in self.levels:
            for sst in reversed(level):
                for k, v in sst.range(start, end):
                    if k in deleted:
                        continue
                    if v is None:
                        deleted.add(k)
                    else:
                        if k not in result:
                            result[k] = v
        return sorted(result.items(), key=lambda kv: SSTable.key_sort(kv[0]))

    def cleanup(self):
        for level in self.levels:
            for sst in level:
                sst.cleanup()
        self.levels.clear()
        for f in os.listdir(self.path):
            if f.endswith('.sst'):
                os.remove(os.path.join(self.path, f))

    def close(self):
        self._flush()

    def report_stats(self):
        for i, level in enumerate(self.levels):
            total_keys = sum(len(sst.index) for sst in level)
            total_size = sum(os.path.getsize(sst.filepath) for sst in level if os.path.exists(sst.filepath))
            bloom_checks = sum(sst.bloom_checks for sst in level)
            bloom_rejected = sum(sst.bloom_rejected for sst in level)
            print(f"Level {i}: SSTables={len(level)}, keys={total_keys}, size={total_size} bytes, bloom checks={bloom_checks}, bloom rejected={bloom_rejected}")

    def _compact_level(self, level):
        if level >= len(self.levels):
            return
        max_tables = self.level_factor ** level
        if len(self.levels[level]) <= max_tables:
            return
        merged_items = []
        for sst in self.levels[level]:
            with open(sst.filepath, 'rb') as f:
                for k, off in sst.index:
                    f.seek(off)
                    klen = struct.unpack('I', f.read(4))[0]
                    f.seek(klen, os.SEEK_CUR)
                    vlen = struct.unpack('I', f.read(4))[0]
                    val = f.read(vlen)
                    merged_items.append((k, val))

        dedup = {}
        for k, v in merged_items:
            dedup[k] = v

        merged_list = list(dedup.items())
        merged_list.sort(key=lambda kv: SSTable.key_sort(kv[0]))

        if level + 1 >= len(self.levels):
            self.levels.append([])

        idx = len(self.levels[level + 1])
        filepath = self._sst_filename(level + 1, idx)
        new_sst = SSTable(filepath, merged_list)
        self.levels[level + 1].append(new_sst)
        for sst in self.levels[level]:
            sst.cleanup()
        self.levels[level].clear()
        self._compact_level(level + 1)