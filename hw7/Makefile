.PHONY: test benchmark clean

test:
	PYTHONPATH=.. python3 -m pytest -v test_kd_tree.py

benchmark:
	PYTHONPATH=.. python3 benchmark_kd_tree.py

clean:
	rm -rf __pycache__ .pytest_cache data/hw7_kdtree_*.bin data/hw7_kdtree_benchmark.csv data/hw7_kdtree_benchmark.png
