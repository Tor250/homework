.PHONY: install test benchmark clean

test:
	PYTHONPATH=.. python3 -m pytest -v test_inverted_index.py

benchmark:
	# табуляция
	PYTHONPATH=.. python3 benchmark_inverted_index.py

clean:
	# табуляция
	rm -rf lsm_index __pycache__