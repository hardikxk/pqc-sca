setup:
	python -m pip install -r requirements.txt

check:
	python scripts/00_check_environment.py

synthetic:
	python scripts/01_generate_synthetic_dataset.py

preprocess:
	python scripts/02_python_preprocess_baseline.py

mojo:
	python scripts/03_run_mojo_preprocess.py

train:
	python scripts/04_train_cnn.py

train-mps:
	python scripts/05_train_cnn_mps.py

evaluate:
	python scripts/06_evaluate_ge.py

benchmark-models:
	python scripts/07_benchmark_models.py

benchmark-preprocessing:
	python scripts/08_benchmark_preprocessing.py

export:
	python scripts/09_export_onnx.py

infer:
	python scripts/10_local_inference.py
