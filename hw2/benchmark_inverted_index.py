import shutil
import random
import time
from pyroaring import BitMap
from hw2.inverted_index_v2 import LSMInvertedIndex

def cleanup_lsm():
    shutil.rmtree("lsm_index", ignore_errors=True)

def benchmark_lsm(N=5000, batch_size=100, mem_limit=500):
    cleanup_lsm()
    idx = LSMInvertedIndex(mem_limit=mem_limit)

    start = time.time()
    batch_docs = []
    for i in range(N):
        doc = f"doc{i} " + " ".join(f"word{random.randint(0,500)}" for _ in range(20))
        batch_docs.append(doc)
        if len(batch_docs) >= batch_size:
            for d in batch_docs:
                idx.add_document(d)
            idx._flush()
            batch_docs.clear()
    for d in batch_docs:
        idx.add_document(d)
    idx._flush()
    idx.close()
    print(f"Inserting {N} docs took {time.time() - start:.2f}s")
    start = time.time()
    idx = LSMInvertedIndex(mem_limit=mem_limit)
    idx._flush()
    for _ in range(5000):
        idx.query_or(f"doc{random.randint(0,N-1)}")
    print(f"5000 random OR queries took {time.time() - start:.2f}s")
    start = time.time()
    for _ in range(5000):
        idx.query_and(f"word{random.randint(0,500)}", f"word{random.randint(0,500)}")
    print(f"5000 random AND queries took {time.time() - start:.2f}s")

    cleanup_lsm()

def benchmark_document_size():
    cleanup_lsm()
    idx = LSMInvertedIndex(mem_limit=200, lsm_path="lsm_index")
    
    import time
    for size in [10, 50, 100]:
        start = time.time()
        for i in range(1000):
            doc = f"doc{i} " + " ".join(f"word{random.randint(0,5000)}" for _ in range(size))
            idx.add_document(doc)
        idx._flush()
        print(f"Inserting 1000 docs of size {size} words took {time.time()-start:.2f}s")
    
    cleanup_lsm()

def benchmark_selectivity(N=5000):
    cleanup_lsm()
    idx = LSMInvertedIndex(mem_limit=500, lsm_path="lsm_index")
    
    for i in range(N):
        tokens = ["common"]
        if i % 50 == 0:
            tokens.append("rare")
        tokens.append(f"unique_{i}")
        idx.add_document(" ".join(tokens))
    idx._flush()
    
    import time
    start = time.time()
    bm = idx.query_and("common")
    print(f"'common' in {len(bm)} docs, query time: {time.time()-start:.3f}s")
    start = time.time()
    bm = idx.query_and("rare")
    print(f"'rare' in {len(bm)} docs, query time: {time.time()-start:.3f}s")
    start = time.time()
    bm = idx.query_and("common", "rare")
    print(f"'common AND rare' in {len(bm)} docs, query time: {time.time()-start:.3f}s")
    
    cleanup_lsm()

def benchmark_complex_and(N=2000, terms_count=5):
    cleanup_lsm()
    idx = LSMInvertedIndex(mem_limit=200, lsm_path="lsm_index")
    
    for i in range(N):
        tokens = [f"term{j}" for j in range(i%terms_count, i%terms_count+terms_count)]
        idx.add_document(" ".join(tokens))
    idx._flush()
    
    import time
    start = time.time()
    for i in range(100):
        query_terms = [f"term{random.randint(0,terms_count-1)}" for _ in range(terms_count)]
        bm = idx.query_and(*query_terms)
    print(f"100 complex AND queries took {time.time()-start:.3f}s")
    
    cleanup_lsm()

def benchmark_interleaved(N=1000):
    cleanup_lsm()
    idx = LSMInvertedIndex(mem_limit=100, lsm_path="lsm_index")
    
    import time
    start = time.time()
    for i in range(N):
        idx.add_document(f"doc{i} common word{i%50}")
        if i % 10 == 0:
            bm = idx.query_or("common", f"word{i%50}")
    print(f"Interleaved insert+query {N} docs took {time.time()-start:.2f}s")
    
    cleanup_lsm()

def benchmark_scalability():
    for N in [1000, 5000, 10000]:
        cleanup_lsm()
        idx = LSMInvertedIndex(mem_limit=500)
        import time
        start = time.time()
        for i in range(N):
            idx.add_document(f"doc{i} " + " ".join(f"word{random.randint(0,5000)}" for _ in range(20)))
        idx._flush()
        insert_time = time.time() - start
        
        start = time.time()
        bm = idx.query_and("doc0")
        query_time = time.time() - start
        print(f"N={N}: Insert time={insert_time:.2f}s, query time={query_time:.4f}s, results={len(bm)}")
        cleanup_lsm()

if __name__ == "__main__":
    benchmark_lsm()
    benchmark_document_size()
    benchmark_selectivity()
    benchmark_complex_and()
    benchmark_interleaved()
    benchmark_scalability()