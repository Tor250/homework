.PHONY: all test benchmark run clean

all: test benchmark

test:
	python3 -m pytest -v test_lsm_tree.py

benchmark:
	python3 benchmark_lsm.py

run:
	python3 run_lsm.py

clean:
	rm -rf test_data bench_data *.pyc __pycache__