# PQC-SCA

PQC-SCA is a lightweight deep-learning side-channel analysis pipeline for masked ML-KEM/Kyber-style traces. It is designed to run locally on a CPU and can also select CUDA automatically when a compatible GPU is available for larger training runs. An NVIDIA GPU and CUDA are not required for the local path.

## Readiness confirmation

The implemented software path is runnable end-to-end on Windows CPU using the supplied synthetic data. The following have been executed successfully: environment checks, unit tests, synthetic HDF5 generation, streaming preprocessing, CNN and CNN+MPS one-epoch training, GE evaluation, benchmark artifact generation, ONNX export, int8 quantization, and ONNX Runtime CPU inference.

This is a validated software smoke test, not a completed hardware side-channel study. No real ML-KEM oscilloscope traces have been collected in this repository. The test data is synthetic and intentionally tiny; demo sizes are selected by the command line (the latest local smoke run used 16 profiling traces and 8 attack traces with 2,000 raw samples each). Therefore generated accuracy, GE, timing, and memory numbers are factual measurements of the local smoke configuration, but they are not statistically meaningful publication results and must not be presented as such.

## Project goals

- Generate reproducible synthetic traces that approximate NTT leakage.
- Preprocess traces in bounded-memory chunks.
- Compare a small 1-D CNN with a compact Matrix Product State (MPS) classifier head.
- Evaluate attack quality with guessing entropy.
- Provide an optional Mojo acceleration boundary with a safe Python fallback.
- Export lightweight models for local CPU inference in later phases.

## Architecture

```text
Raw HDF5 traces
       |
       v
Chunked normalization -> FIR smoothing -> alignment -> 5,000-sample reduction
       |
       +--> LightweightCNN --------------------+
       |                                       |
       +--> CNN feature extractor + MPS head --> logits -> rank/GE evaluation
       |
       +--> optional Mojo kernels
             (Python fallback when Mojo is unavailable)
```

### Data layer

`mojopqc_sca/datasets/` contains chunked HDF5 generation and loaders. The PyTorch dataset reads one trace at a time, so training does not load the full HDF5 file into RAM.

The external-data adapter can inspect common Zenodo-style `fixed/` and `random/` groups and convert them in chunks. It never invents a sensitive label silently: the caller must explicitly select an input byte with `--label-byte` and record that choice in provenance metadata.

### Preprocessing layer

`mojopqc_sca/preprocessing/` contains normalization, FIR filtering, cross-correlation alignment, downsampling, and the reference pipeline. The target neural input length is 5,000 samples.

### Model layer

The CNN uses two convolution/pooling blocks with 16 and 32 channels. `CNNWithMPS` replaces the final dense feature-to-class projection with a compact MPS head. The default MPS bond dimension is 8 and the model is asserted to remain below 200,000 trainable parameters.

### Evaluation layer

Guessing entropy accumulates per-trace log probabilities and reports the rank of the correct label after each attack-trace prefix. Repeated attack partitions can be aggregated externally to obtain a mean GE curve.

### Execution modes

| Mode | Required hardware | Status |
|---|---|---|
| CPU Python/SciPy preprocessing | CPU | Implemented and tested |
| CNN/CNN+MPS training | CPU | Implemented; CPU smoke-tested |
| ONNX Runtime inference | CPU | Implemented and tested |
| Mojo preprocessing | Mojo compiler | Interface exists; native kernels still pending |
| Real-trace analysis | ML-KEM implementation and acquisition hardware | Data collection not included |

## Repository layout

```text
config/                 YAML configurations for local and Colab runs
data/raw/               raw/synthetic HDF5 input
data/processed/         preprocessed HDF5 output
mojo_kernels/           optional Mojo kernel boundary
mojopqc_sca/datasets/   HDF5, memmap, and PyTorch data access
mojopqc_sca/models/     CNN and CNN+MPS models
mojopqc_sca/preprocessing/
mojopqc_sca/evaluation/
scripts/                command-line pipeline stages
results/                checkpoints, logs, figures, tables, benchmarks
```

## Quick One-Click Demo (`uv` runtime)

You can run the entire end-to-end pipeline and launch the interactive visual demo dashboard with a single command:

```powershell
uv run demo.py
```

This single command automatically resolves dependencies, verifies the environment, generates synthetic NTT traces, runs streaming FIR & alignment preprocessing, validates dataset contracts, trains the CNN and CNN+MPS models, calculates Guessing Entropy (GE), benchmarks model metrics, dynamically quantizes to Int8 ONNX, benchmarks local CPU inference, generates the reproducibility manifest, and starts the live interactive web dashboard at `http://127.0.0.1:8000`.

Options:
- `uv run demo.py --cli-only` (run full pipeline in terminal without opening browser)
- `uv run demo.py --full` (train with larger dataset)
- `uv run demo.py --port 8080` (custom web port)

## Manual Step-by-Step Installation and Demo

From the repository root:

```powershell
python -m pip install -r requirements.txt
python scripts/00_check_environment.py
python scripts/01_generate_synthetic_dataset.py --profiling 32 --attack 8 --trace-length 20000
python scripts/02_python_preprocess_baseline.py
python scripts/03_run_mojo_preprocess.py
```

For a model smoke test after preprocessing, run one epoch:

```powershell
python scripts/04_train_cnn.py --epochs 1 --batch-size 8
python scripts/05_train_cnn_mps.py --epochs 1 --batch-size 8
```

Evaluate the MPS model:

```powershell
python scripts/06_evaluate_ge.py
```

The complete operating guide, including external test-data placement and output paths, is in [RUN_AND_DEMO.md](RUN_AND_DEMO.md).

## Implemented phases

### Phase 0 — Environment and project setup

Implemented: `requirements.txt`, YAML configuration, Makefile targets, environment reporting, package metadata, and CPU/optional-tool detection.

### Phase 1 — Synthetic trace generation

Implemented: deterministic synthetic NTT-like leakage generation, Hamming-weight labels, chunked HDF5 writes, profiling/attack splits, and metadata storage.

### Phase 2 — Python preprocessing baseline

Implemented: per-trace normalization, moving-average FIR filtering, bounded cross-correlation alignment, downsampling to 5,000 samples, chunked HDF5 processing, and timing/peak-RSS reporting.

### Phase 3 — Mojo boundary and fallback

Implemented: Mojo kernel file layout, fallback dispatch, and benchmark entry point. Native Mojo SIMD kernels and a production HDF5-to-Mojo bridge are not yet implemented; when Mojo is absent the Python reference path runs safely.

### Phase 4 — Lightweight CNN

Implemented: two-block 1-D CNN with 16/32 channels, CPU training, deterministic train/validation split, checkpoint saving, CSV training logs, and parameter counting. The current model has 14,400 trainable parameters.

### Phase 5 — CNN+MPS

Implemented: compact bond-dimension-8 MPS classifier head, CPU training, checkpoint saving, and parameter-budget assertion. The current model has 9,226 trainable parameters.

### Phase 6 — Guessing entropy

Implemented: cumulative log-probability evidence, key-rank calculation, GE prefix curves, JSON output, and GE plotting.

### Phase 7 — Model benchmarking

Implemented: CNN versus CNN+MPS parameter, validation, inference, memory, and GE comparison with JSON, CSV, LaTeX, and PNG outputs.

### Phase 8 — Preprocessing benchmarking

Implemented: Python-reference versus portable fallback timing, throughput, speedup, memory reporting, and paper-oriented output files. Because both current paths use Python-side processing, this is not evidence of Mojo speedup.

### Phase 9 — ONNX deployment

Implemented: ONNX export, int8 dynamic quantization, graph sanitization for the current PyTorch exporter, and local ONNX Runtime CPU inference.

### Phase 10 — Colab and visualization

Implemented: Colab training notebook and results visualization notebook. The trainer supports `--device auto|cpu|cuda`, streams HDF5 samples on demand, and records the selected device and peak GPU memory when CUDA is used. The notebooks still require a complete dataset/checkpoint run on the target Colab environment.

### Phase 11 — Academic packaging

Implemented: unit tests, contribution guidance, citation metadata, paper scaffold, explicit limitations, and reproducibility documentation.

### Phase 12 — Research-engineering hardening and CI

Implemented: configuration validation, HDF5 schema validation, chunk-iterator tests, optional Numba normalization acceleration, cross-platform peak-memory monitoring, and GitHub Actions CPU smoke CI on Windows and Ubuntu. CI intentionally uses a small deterministic dataset and does not require CUDA or Mojo.

### Phase 13 — External ML-KEM HDF5 ingestion

Implemented: external HDF5 inspection, discovery of common `fixed/traces`, `random/traces`, `fixed/input(s)`, and `random/input(s)` layouts, chunked conversion into the project contract, explicit input-byte label selection, provenance metadata, and adapter unit tests. This enables the next real-data phase but does not by itself prove that an input byte is the correct sensitive intermediate for an attack.

### Phase 14 — Dataset validation and provenance checks

Implemented: streaming validation of the project HDF5 contract, finite-value checks, label-range checks, optional metadata JSON validation, JSON reporting, explicit source-path overrides for non-standard external files, and unit tests. The validator holds only one batch in memory and can be run before preprocessing or training.

### Phase 15 — Portable training device selection

Implemented: explicit `auto`, `cpu`, and `cuda` training modes, automatic CPU fallback when CUDA is unavailable, pinned host batches for CUDA transfers, and device/peak-GPU-memory metadata in checkpoints and training logs. This preserves the CPU-only local workflow while making the Colab training path use an available GPU.

### Phase 16 — Reproducibility manifests

Implemented: streaming SHA-256 hashing, Git revision/dirty-state capture, environment/package capture, embedded dataset validation, configuration capture, config-versus-dataset consistency checks, selected artifact hashes, and a JSON run-manifest command. This makes a demo or experiment auditable after the terminal session has ended.

## What the current results mean

The files under `results/` are generated artifacts from the small local validation run. They confirm that the code paths work and provide examples of the expected output formats. They do not establish that the models recover ML-KEM secrets, that MPS outperforms the CNN, or that Mojo accelerates preprocessing.

For factual research claims, rerun the pipeline with a sufficiently large synthetic or hardware dataset, multiple seeds/attack partitions, stable CPU/GPU settings, and a native Mojo implementation. Report confidence intervals or repeated-run variability and include the acquisition metadata for hardware traces.

## Continuous integration

The workflow at `.github/workflows/ci.yml` runs on pushes and pull requests. It installs the declared dependencies, compiles the package, executes the unit tests, trains one-epoch smoke checkpoints, generates GE and benchmark artifacts, exports ONNX, and runs CPU inference. A green CI run confirms software integration, not scientific validity of side-channel results.

Run the same core checks locally:

```powershell
python -m compileall -q mojopqc_sca scripts
python -m unittest discover -s tests -v
python scripts/12_validate_dataset.py --input data/processed/python_processed.h5 --strict
```

## External HDF5 conversion

Inspect a downloaded Zenodo-style file without loading trace payloads:

```powershell
python scripts/11_convert_external_hdf5.py --input path\to\ml-kem-512_masked.h5 --inspect
```

Convert it to the project layout using an explicitly chosen input byte as the temporary class label:

```powershell
python scripts/11_convert_external_hdf5.py `
  --input path\to\ml-kem-512_masked.h5 `
  --output data/raw/mlkem_converted.h5 `
  --label-byte 0 `
  --profiling-limit 1000 `
  --attack-limit 200
```

If automatic discovery does not find the source paths, provide all four paths explicitly with `--profiling-traces-path`, `--profiling-inputs-path`, `--attack-traces-path`, and `--attack-inputs-path`. Validate the converted file before preprocessing:

```powershell
python scripts/12_validate_dataset.py `
  --input data/raw/mlkem_converted.h5 `
  --output results/benchmarks/mlkem_converted_validation.json `
  --strict
```

The label byte is an experiment decision, not a universal default. Confirm from the implementation and dataset documentation that it corresponds to the sensitive intermediate being modeled. The resulting file can then be passed to preprocessing with `--input data/raw/mlkem_converted.h5`.

## Available Datasets and Training Guide

The repository includes both reproducible synthetic datasets and real ML-KEM-512 side-channel power/EM datasets, properly sorted into `data/raw/`, `data/processed/`, `data/metadata/`, and `data/external/`.

### 1. Dataset Catalog

| Dataset | Raw File (`data/raw/`) | Preprocessed File (`data/processed/`) | Traces & Samples | Description |
|---|---|---|---|---|
| **Synthetic PQC Baseline** | `synthetic_pqc.h5` | `python_processed.h5` | 32/8 (demo) up to 10k/2k, L=5,000 | Deterministic NTT leakage generator with Hamming-weight class labels for development and CI smoke tests. |
| **Real Masked ML-KEM-512** | `ml-kem-512_masked.h5` *(8.45 GB)* & `mlkem_masked_converted.h5` | `mlkem_masked_processed.h5` | 100,000 profiling / 100,000 attack, L=41,800 raw (downsampled to 5,000) | Full Zenodo fixed-vs-random dataset from real masked ML-KEM-512 power acquisition campaigns on ARM Cortex-M4. |
| **Real Unprotected ML-KEM-512** | `ml-kem-512_unprotected.h5` *(595 MB)* & `mlkem_unprotected_converted.h5` | `mlkem_unprotected_processed.h5` | 10,000 profiling / 10,000 attack, L=28,600 raw (downsampled to 5,000) | Unprotected ML-KEM-512 power traces for leakage comparison and vulnerability baselines. |
| **Simulation & Confusion Matrices** | `data/external/doi-10.48804-4qbdrp.zip` | Extracted in `data/metadata/confusion_matrices/` | N/A | KU Leuven Kyber case-study reference matrices and belief-propagation simulation artifacts. |
| **Oscilloscope Chosen-Ciphertext Archive** | `data/external/kyber-sca.zip` *(2.19 GB)* | Documentation in `data/metadata/kyber_sca_readme.txt` | Binary traces & key sheets | Real STM32F4 oscilloscope traces and secret key data sets. |

### 2. How to Train on Any Dataset

You can train on the synthetic baseline, the real masked ML-KEM dataset, or the unprotected dataset using either the one-click runner or step-by-step CLI commands.

#### Option A: One-Click Runner (`uv run demo.py`)

Run the complete pipeline (validation, training, GE evaluation, ONNX dynamic int8 export, and web dashboard) on the selected dataset:

```powershell
# Train on Real Masked ML-KEM-512 dataset:
uv run demo.py --dataset masked

# Train on Real Unprotected ML-KEM-512 dataset:
uv run demo.py --dataset unprotected

# Train on Synthetic Baseline (default):
uv run demo.py --dataset synthetic
```

#### Option B: Step-by-Step Modular Training Scripts

To train models individually on a specific preprocessed dataset:

```powershell
# 1. Train Lightweight 1-D CNN Baseline (e.g. on Masked ML-KEM-512)
python scripts/04_train_cnn.py `
  --input data/processed/mlkem_masked_processed.h5 `
  --epochs 10 `
  --batch-size 32 `
  --device auto

# 2. Train CNN + MPS Tensor-Network Classifier
python scripts/05_train_cnn_mps.py `
  --input data/processed/mlkem_masked_processed.h5 `
  --epochs 10 `
  --batch-size 32 `
  --bond-dim 8 `
  --device auto

# 3. Evaluate Guessing Entropy (Key Rank) on Attack Split
python scripts/06_evaluate_ge.py `
  --input data/processed/mlkem_masked_processed.h5 `
  --checkpoint results/models/cnn_mps.pt

# 4. Generate Comparative Model Benchmarks
python scripts/07_benchmark_models.py `
  --input data/processed/mlkem_masked_processed.h5

# 5. Export to Dynamic Int8 Quantized ONNX & Run Local CPU Inference
python scripts/09_export_onnx.py
python scripts/10_local_inference.py --input data/processed/mlkem_masked_processed.h5
```

## Test data

The repository’s included data is synthetic. It is suitable for validating software, but it is not evidence from a physical ML-KEM implementation. For a real-trace study, look for power or EM traces captured from a specific implementation, together with the labels and acquisition metadata needed to define a profiling split and an attack split.

### Most relevant public sources

| Source | What it provides | Practical use here |
|---|---|---|
| [SOLO ML-KEM/Kyber traces](https://zenodo.org/records/20573193) | A Zenodo record describing fixed-key and variable-key ML-KEM decapsulation power-trace subsets in HDF5 format. The record currently marks the files as restricted, so access must be checked before planning around it. | Best direct match if access is granted. |
| [ML-KEM Side Channel Traces](https://huggingface.co/datasets/ai-eldorado/ML-KEM-SideChannel-Traces) | ML-KEM-768 traces from an STM32 Cortex-M4/PQM4 setup, with FIXED and RANDOM groups and more than 158k samples per trace. The dataset page currently reports that the dataset is empty, so treat it as a lead and verify availability. | Good format and acquisition reference; not currently a guaranteed download. |
| [Kyber chosen-ciphertext trace dataset](https://zenodo.org/records/4726798) | Dataset associated with a published Kyber side-channel case study. | Relevant historical Kyber data; inspect its metadata and license before adapting it. |
| [KU Leuven Kyber case-study materials](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/4QBDRP) | Research materials for a CRYSTALS-Kyber side-channel case study, linked to source code and analysis artifacts. | Useful for reproducing a known Kyber experiment, subject to its download/access conditions. |
| [KyberCPA collection workflow](https://github.com/Chaman-veteran/KyberCPA) | Firmware/build and trace-collection scripts based on PQM4 and an ARM Cortex-M4 capture workflow. | Best route if you need to collect your own traces. |
| [PQM4 implementation repository](https://github.com/mupq/pqm4) | Embedded post-quantum implementations and build targets, not a ready-made trace dataset. | Use as the implementation source for a controlled acquisition campaign. |
| [ASCAD](https://github.com/ANSSI-FR/ASCAD) | Well-known labeled AES side-channel datasets and scripts. It is not ML-KEM data. | Use only to validate generic SCA tooling/adapters, not to claim ML-KEM results. |

### Important: do not confuse the ML-DSA and ML-KEM records

The [ML-DSA record](https://zenodo.org/records/18670586) you may encounter contains `ml-dsa-44_masked.h5` (45.1 GB) and `ml-dsa-44_unprotected.h5` (3.4 GB). Those files are for the signature algorithm ML-DSA/Dilithium, not for the ML-KEM decapsulation experiment this repository is currently designed to analyze. Do not download the 48.5 GB ML-DSA collection just to test the current ML-KEM pipeline.

The corresponding [ML-KEM and masked-components record](https://zenodo.org/records/18681117) is the relevant one. Its `ml-kem-512_masked.h5` file is listed as 8.9 GB, and its `ml-kem-512_unprotected.h5` file as 623.9 MB. The smaller component files in that record are useful for staged leakage experiments, but they do not represent the complete ML-KEM decapsulation path. Start with the unprotected file only if the immediate goal is format inspection and adapter development; use the masked ML-KEM file for the actual masked-implementation study, provided you have enough disk space for the download, temporary files, converted data, and results.

These Zenodo files use groups such as `fixed/traces`, `random/traces`, `fixed/input(s)`, `random/inputs`, and `settings`; they are not already in this repository's `/traces/profiling`, `/labels/profiling`, `/traces/attack`, `/labels/attack` contract. A conversion/labeling adapter is therefore required before the current training scripts can consume them. Downloading the HDF5 file alone will not make the pipeline automatically train on it.

Useful search queries are:

```text
ML-KEM power traces HDF5 dataset
Kyber side-channel power traces Cortex-M4
ML-KEM decapsulation side-channel traces dataset
PQM4 Kyber ChipWhisperer traces
SOLO side-channel observations lattice-based operations dataset
Kyber chosen ciphertext side-channel dataset
```

### What a usable real dataset must contain

At minimum, record the algorithm and parameter set (for example ML-KEM-768), implementation and commit, target device, capture modality (power or EM), sampling rate, trigger/window definition, number of samples per trace, key/ciphertext policy, masking/countermeasure configuration, and any desynchronization. For profiled deep-learning SCA, you also need a sensitive intermediate or equivalent per-trace class label. A file containing only unlabeled traces or only TVLA fixed/random groups is useful for leakage detection, but it is not automatically sufficient for supervised key-rank evaluation.

### Adapting external data to this repository

Do not copy an arbitrary CSV/HDF5 file into `data/raw/` and assume the pipeline can infer its meaning. Convert it into the repository contract:

```text
/traces/profiling    float32, shape (N, L)
/labels/profiling    uint8, shape (N,)
/traces/attack       float32, shape (M, L)
/labels/attack       uint8, shape (M,)
/metadata/config     JSON string containing source, parameter set, device, sampling rate, label definition, and split policy
```

Save the converted file as `data/raw/synthetic_pqc.h5` to have the default commands pick it up automatically, or pass a custom path with `--input`. Keep the original downloaded files outside the converted pipeline file so provenance remains auditable. If the source labels are not 0–255 class IDs, write an adapter/conversion script and document exactly how each class was derived.

For a serious result, use separate profiling and attack traces, multiple attack repetitions or fixed-key attack campaigns, and a held-out key/device condition where appropriate. Preserve the source license and cite the dataset paper or repository.

Before preprocessing or training, run the bounded-memory validator. It checks the project schema, scans for non-finite trace values, reports label coverage, and records whether optional provenance JSON is valid:

```powershell
python scripts/12_validate_dataset.py --input data/raw/synthetic_pqc.h5 --output results/benchmarks/dataset_validation.json --strict
python scripts/13_create_run_manifest.py --dataset data/processed/python_processed.h5
```

The manifest also reports whether the dataset dimensions match the YAML declarations. If you generated a deliberately small smoke dataset with command-line overrides, those count checks are expected to be false; use a matching configuration file for publication experiments.

## Current status

Implemented: environment check, synthetic HDF5 generation, streaming preprocessing, optional Mojo dispatch/fallback, data loaders, CNN, CNN+MPS, training scripts, guessing-entropy evaluation, external ML-KEM HDF5 conversion, and streaming dataset validation.

Implemented in this development slice: model comparison, preprocessing comparison, ONNX export/local inference entry points, unit tests, Colab notebooks, visualization notebook, paper scaffold, citation metadata, contribution guidance, external-data validation, and portable training-device selection.

Still planned: native SIMD Mojo implementations and a validated hardware-trace study. The ONNX and benchmark stages require their optional runtime dependencies and trained checkpoints.

## Research workflow

The recommended experiment sequence is:

1. Freeze a YAML configuration and record the environment report.
2. Generate or place raw HDF5 traces under `data/raw/`.
3. Run streaming preprocessing and retain its JSON benchmark.
4. Train both models using the same split and seed.
5. Evaluate guessing entropy on the held-out attack split.
6. Run model and preprocessing benchmarks.
7. Export the selected checkpoint to ONNX and measure CPU inference.
8. Inspect generated figures and tables before inserting them into `paper/main.tex`.

Never compare synthetic and hardware traces as if they were interchangeable experiments. Record the trace source, acquisition setup, preprocessing configuration, checkpoint hash, and software environment for every reported result.

## Research artifacts

- `tests/`: deterministic unit tests for preprocessing, GE calculations, HDF5 conversion, and validation.
- `mojopqc_sca/datasets/validation.py`: streaming schema and quality reporting.
- `scripts/12_validate_dataset.py`: command-line dataset validation report.
- `scripts/13_create_run_manifest.py`: reproducibility manifest with hashes and environment metadata.
- `notebooks/colab_train.ipynb`: cloud-oriented training workflow.
- `notebooks/results_visualization.ipynb`: inspection of JSON/CSV/PNG outputs.
- `paper/main.tex`: manuscript scaffold with explicit limitations.
- `CITATION.cff`: citation metadata template; replace the repository URL and author before release.
- `PITCH_NOTES.md`: concise personal review/demo pitch and project explanation.

## Reproducibility and constraints

The seed and data sizes are in `config/default.yaml`; Colab-oriented values are in `config/colab.yaml`. The default full dataset is intentionally large, so use the small CLI overrides for local development. Generated data and result artifacts belong under `data/` and `results/` and should not be committed.
