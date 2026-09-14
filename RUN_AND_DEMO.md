# MojoPQC-SCA: Run and Demo Guide

This document is the operational guide for running the MojoPQC-SCA side-channel analysis pipeline from the repository root:

```powershell
Set-Location C:\Users\xane\Desktop\NeuralSCA
```

## 1. What is currently runnable

The current implementation includes the CPU-first foundation:

1. Environment inspection.
2. Chunked synthetic PQC trace generation in HDF5.
3. Streaming Python preprocessing.
4. Optional Mojo dispatch with a safe Python fallback.
5. HDF5 and memory-mapped data-access helpers.
6. Lightweight CNN and CNN+MPS model definitions.
7. CPU-safe training scripts and guessing-entropy evaluation.
8. External HDF5 inspection and chunked conversion for common fixed/random ML-KEM datasets.
9. Streaming dataset validation and JSON quality reports.
10. Reproducibility manifests containing hashes, environment, Git state, validation results, and config consistency checks.

Native Mojo acceleration remains planned work. Model/preprocessing comparison, ONNX entry points, tests, notebooks, GPU-capable training selection, and the paper scaffold are now present.

GitHub Actions runs the same CPU smoke path on Windows and Ubuntu through `.github/workflows/ci.yml`. The workflow does not require CUDA, an NVIDIA GPU, or Mojo.

## 2. One-Command Full Demo (`uv run`)

Run the complete pipeline and launch the interactive visual dashboard in one single command using `uv`:

```powershell
uv run demo.py
```

This runs all stages end-to-end and starts the interactive dark-mode dashboard at `http://127.0.0.1:8000`.

## 3. Install dependencies (Manual)

Use Python 3.10 or newer. From PowerShell:

```powershell
python -m pip install -r requirements.txt
```

For only the currently runnable CPU pipeline, these packages are sufficient:

```powershell
python -m pip install numpy scipy h5py pyyaml psutil
```

PyTorch, ONNX Runtime, and Mojo are optional for this first stage. The environment checker reports missing optional tools without failing.

## 3. Check the environment

```powershell
python scripts/00_check_environment.py
```

Expected behavior:

- Python and installed scientific packages are reported.
- Missing CUDA is reported as CPU mode; this is expected on a CPU-only machine.
- Missing Mojo is reported as fallback mode; the pipeline continues using Python/NumPy/SciPy.

## 4. Fast local demo

This command creates a small dataset suitable for a quick demonstration. It does not allocate the full research dataset:

```powershell
python scripts/01_generate_synthetic_dataset.py `
  --profiling 32 `
  --attack 8 `
  --trace-length 20000
```

Then preprocess it:

```powershell
python scripts/02_python_preprocess_baseline.py `
  --input data/raw/synthetic_pqc.h5 `
  --output data/processed/python_processed.h5
```

Run the optional Mojo stage. If `mojo` is not on `PATH`, this automatically uses the reference Python fallback:

```powershell
python scripts/03_run_mojo_preprocess.py `
  --input data/raw/synthetic_pqc.h5 `
  --output data/processed/mojo_processed.h5
```

Inspect the generated HDF5 shapes:

```powershell
python -c "import h5py; f=h5py.File('data/processed/python_processed.h5'); print(f['traces/profiling'].shape); print(f['traces/attack'].shape)"
```

The processed trace shape is `(number_of_traces, 5000)` by default, which is the planned maximum neural-network input length.

## 5. Full synthetic dataset

The default configuration is in `config/default.yaml`:

- 10,000 profiling traces.
- 2,000 attack traces.
- 100,000 samples per raw trace.
- Chunk size 32.

Generate it with:

```powershell
python scripts/01_generate_synthetic_dataset.py
```

This dataset is intentionally large. Generation is chunked, but the resulting HDF5 file can still require several gigabytes of disk space and preprocessing time. Use the small demo configuration first.

For cloud training, use `config/colab.yaml`, which uses larger I/O chunks:

```powershell
python scripts/01_generate_synthetic_dataset.py --config config/colab.yaml
```

## 6. Test-data contract

The default pipeline input is:

```text
data/raw/synthetic_pqc.h5
```

To provide an external test dataset that the current scripts pick up automatically, place it at exactly that path and use this HDF5 layout:

```text
/traces/profiling    float32 array, shape (N, L)
/labels/profiling    uint8 array, shape (N,)
/traces/attack       float32 array, shape (M, L)
/labels/attack       uint8 array, shape (M,)
/metadata/config     optional JSON string
```

Requirements:

- Profiling and attack traces must have the same raw sample length `L`.
- Labels must be integers in `[0, 255]` for the default 256-class configuration.
- Trace arrays should be HDF5 datasets, not Python object arrays.
- Store raw test data only under `data/raw/`.

With that filename and structure, run the normal preprocessing command without `--input`:

```powershell
python scripts/02_python_preprocess_baseline.py
```

For another filename, pass it explicitly:

```powershell
python scripts/02_python_preprocess_baseline.py `
  --input data/raw/my_test_traces.h5 `
  --output data/processed/my_test_processed.h5
```

The Mojo/fallback stage follows the same convention. The default input is `data/raw/synthetic_pqc.h5`; its default output is `data/processed/mojo_processed.h5`.

## 7. Train and evaluate the models

Training consumes the processed profiling split. PyTorch is required for these commands. For a local smoke test, use one epoch and a small batch size:

```powershell
python scripts/04_train_cnn.py --epochs 1 --batch-size 8
python scripts/05_train_cnn_mps.py --epochs 1 --batch-size 8
```

The training scripts accept `--device auto|cpu|cuda`. `auto` selects CUDA when available and safely selects CPU otherwise; on this Windows machine, use `--device cpu` or leave the default `auto`.

The normal research configuration is 30 epochs and batch size 128:

```powershell
python scripts/04_train_cnn.py
python scripts/05_train_cnn_mps.py
```

Outputs:

- `results/models/cnn_baseline.pt`
- `results/models/cnn_mps.pt`
- `results/logs/cnn_training.csv`
- `results/logs/cnn_mps_training.csv`

Evaluate the CNN+MPS checkpoint against the attack split:

```powershell
python scripts/06_evaluate_ge.py --checkpoint results/models/cnn_mps.pt
```

This writes `results/benchmarks/ge_results.json` and `results/figures/ge_curve.png`. Each GE point is the rank after cumulative evidence from that many attack traces. It prints GE at 100, 500, and 1,000 traces when those attack-trace counts exist, plus GE for the complete attack split.

## 8. Generate comparison artifacts

After both checkpoints exist, benchmark the models:

```powershell
python scripts/07_benchmark_models.py
```

This writes:

- `results/benchmarks/model_comparison.json`
- `results/benchmarks/model_comparison.csv`
- `results/tables/model_comparison.tex`
- `results/figures/model_comparison.png`

Benchmark the reference and fallback preprocessing implementations:

```powershell
python scripts/08_benchmark_preprocessing.py
```

This writes the corresponding JSON, CSV, LaTeX table, and speedup figure under `results/`. The current fallback is a correctness/reference path; do not describe its measured speed as Mojo acceleration.

## 9. Export and run CPU inference

After training CNN+MPS:

```powershell
python scripts/09_export_onnx.py
python scripts/10_local_inference.py
python scripts/11_convert_external_hdf5.py --input path\to\ml-kem-512_masked.h5 --inspect
python scripts/12_validate_dataset.py --input data/processed/python_processed.h5 --strict
python scripts/13_create_run_manifest.py --dataset data/processed/python_processed.h5
```

The manifest records whether the dataset dimensions match the YAML configuration. For the deliberately small smoke run, count mismatches are expected because `--profiling`, `--attack`, and `--trace-length` override the default configuration; use a matching YAML file for formal experiments.

The export creates `results/models/cnn_mps.onnx` and `results/models/cnn_mps_int8.onnx`. Inference creates `results/benchmarks/local_inference.json` and prints milliseconds per trace and GE. ONNX Runtime is required for these commands.

## 10. Where results are written

The scripts create output directories automatically.

| Artifact | Location |
|---|---|
| Raw synthetic or supplied HDF5 | `data/raw/` |
| Python-processed HDF5 | `data/processed/python_processed.h5` |
| Mojo/fallback-processed HDF5 | `data/processed/mojo_processed.h5` |
| Python preprocessing benchmark | `results/benchmarks/python_preprocess.json` |
| Mojo/fallback benchmark | `results/benchmarks/mojo_preprocess.json` |
| Dataset validation report | `results/benchmarks/dataset_validation.json` |
| Reproducibility manifest | `results/benchmarks/run_manifest.json` |
| Metadata | `data/metadata/` |
| Future model checkpoints | `results/models/` |
| Future figures | `results/figures/` |
| Future CSV/LaTeX tables | `results/benchmarks/` and `results/tables/` |

Generated datasets and result artifacts should not be committed to source control.

## 11. Reproducibility

The random seed is configured in the YAML file. To reproduce synthetic data, keep the same config and seed. Command-line overrides such as `--profiling`, `--attack`, and `--trace-length` intentionally create a different dataset size while preserving deterministic generation for the same seed and parameters.

## 12. Troubleshooting

### `ModuleNotFoundError: h5py`

Install the CPU dependencies:

```powershell
python -m pip install numpy scipy h5py pyyaml psutil
```

### Mojo is unavailable

This is supported. `scripts/03_run_mojo_preprocess.py` prints a warning and writes the fallback result and benchmark. Install Mojo later only when benchmarking native kernels.

### The full run is too slow or uses too much disk

Use the fast demo arguments first. The raw plan uses 100,000 samples per trace and 12,000 total traces; reduce `--profiling`, `--attack`, and `--trace-length` for local development.

### Input HDF5 is rejected or produces incorrect output

Check dataset names, shapes, and dtypes against the test-data contract above. In particular, do not place labels under a flat name such as `/labels`; the loader expects `/labels/profiling` and `/labels/attack`.

## 13. Planned end-to-end command sequence

For the currently implemented Python/CPU path, the complete sequence is:

```powershell
python scripts/00_check_environment.py
python scripts/01_generate_synthetic_dataset.py
python scripts/02_python_preprocess_baseline.py
python scripts/03_run_mojo_preprocess.py
python scripts/04_train_cnn.py
python scripts/05_train_cnn_mps.py
python scripts/06_evaluate_ge.py
python scripts/07_benchmark_models.py
python scripts/08_benchmark_preprocessing.py
python scripts/09_export_onnx.py
python scripts/10_local_inference.py
python scripts/12_validate_dataset.py --input data/processed/python_processed.h5 --strict
python scripts/13_create_run_manifest.py --dataset data/processed/python_processed.h5
```

Training defaults to `--device auto`, selecting CUDA when available and CPU otherwise. Use Google Colab for larger training runs when local CPU execution is too slow; evaluation, benchmarking, and ONNX inference remain CPU-compatible.

## 14. Local validation

Run the same checks used by CI:

```powershell
python -m compileall -q mojopqc_sca scripts
python -m unittest discover -s tests -v
```

A successful run confirms software correctness for the tested inputs. It does not establish cryptographic security, hardware-trace realism, or statistically meaningful research performance.
