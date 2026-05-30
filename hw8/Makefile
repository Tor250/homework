.PHONY: test benchmark clean

test:
	PYTHONPATH=.. python3 -m pytest -v test_fm_index.py

benchmark:
	PYTHONPATH=.. python3 benchmark_fm_index.py

clean:
	rm -rf __pycache__ .pytest_cache data/hw8_fm_index_benchmark.csv data/hw8_fm_index_benchmark.png
