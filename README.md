# NeuralSCA / MojoPQC-SCA: Accelerated Neural Side-Channel Analysis of Masked ML-KEM

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.19+-005CED.svg)](https://onnxruntime.ai/)
[![NIST Standard](https://img.shields.io/badge/NIST_PQC-FIPS_203_(ML--KEM)-darkgreen.svg)](https://csrc.nist.gov/pubs/fips/203/final)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **MojoPQC-SCA (NeuralSCA)** is a lightweight, end-to-end deep learning side-channel analysis (DL-SCA) profiling pipeline engineered to evaluate the physical hardware security of **NIST FIPS 203 (ML-KEM / CRYSTALS-Kyber)** on embedded microcontrollers.
> 
> It eliminates the severe memory and parameter bottlenecks of post-quantum side-channel profiling by combining **bounded-memory streaming HDF5 ingestion** ($<112\text{ MB}$ peak RAM) with a **quantum-inspired Matrix Product State (MPS) tensor network** ($9,226$ parameters, $35.9\%$ smaller than baseline CNNs) and **dynamic symmetric Int8 ONNX deployment** ($0.14\text{ ms per trace}$ on a standard CPU).

---

## Table of Contents
- [Executive Summary](#executive-summary)
- [The Problem We Solve](#the-problem-we-solve)
- [Key Features & Architecture](#key-features--architecture)
- [Empirical Results & Benchmark Highlights](#empirical-results--benchmark-highlights)
- [Comparison with Existing Alternatives](#comparison-with-existing-alternatives)
- [Interactive Developer Console & Web Testbed](#interactive-developer-console--web-testbed)
- [Quick Start (One-Click Demo)](#quick-start-one-click-demo)
- [Step-by-Step Modular Pipeline](#step-by-step-modular-pipeline)
- [Available Datasets (Real & Synthetic)](#available-datasets-real--synthetic)
- [Hardware Cryptographic Audit & Reproducibility](#hardware-cryptographic-audit--reproducibility)
- [Research Paper & Explanation Guide](#research-paper--explanation-guide)
- [Repository Structure](#repository-structure)
- [Citation](#citation)

---

## Executive Summary

When post-quantum lattice cryptography primitives such as **ML-KEM-512** are executed on physical silicon (such as 32-bit ARM Cortex-M4 microcontrollers), the internal arithmetic loops of the **Number Theoretic Transform (NTT)** unintentionally leak sensitive information through instantaneous power consumption and electromagnetic (EM) emissions. 

While algorithmic **Boolean masking** ($x = x_1 \oplus x_2$) splits secrets into randomized shares to prevent first-order leakage, non-linear arithmetic conversions (A2B/B2A) introduce higher-order vulnerabilities that deep learning models can exploit. However, existing DL-SCA tools fail when applied to PQC targets:
1. **The Ingestion Wall:** A $100,000$-trace dataset of length $L = 41,800$ samples requires over **$16.7\text{ GB}$ of RAM** in standard NumPy/PyTorch arrays, immediately causing Out-Of-Memory (OOM) crashes on standard developer machines.
2. **The Parameter Explosion:** Mapping long PQC traces into standard dense classification heads creates over $400,000$ trainable parameters, causing severe overfitting on low-SNR physical noise.

**MojoPQC-SCA resolves both challenges:**
- **Streaming Pipeline:** Processes traces in contiguous chunks ($B=256$) with zero-phase digital FIR filtering, template cross-correlation alignment, and POI decimation to $5,000$ points, capping peak RAM at **$\le 111.8\text{ MB}$** ($>99.3\%$ memory reduction) with a sustained throughput of **$761.2\text{ traces/s}$**.
- **Quantum-Inspired MPS Head:** Replaces the dense classifier with a low-rank Matrix Product State tensor network ($\chi = 8$). It shrinks trainable parameters to **$9,226$** ($35.93\%$ smaller than the 1D-CNN baseline and utilizing only $4.6\%$ of the $200\text{k}$ budget ceiling), speeds up training by **$1.69\times$**, and drives **Guessing Entropy to Rank 1.0** on masked ML-KEM-512.
- **Quantized Edge Deployment:** Dynamically quantized to **Int8 ONNX** ($18.6\text{ KB}$ binary), executing single-threaded CPU inference in **$0.14\text{ ms per trace}$** ($7,142\text{ traces/second}$).

---

## System Architecture

```text
+-----------------------------------------------------------------------------------------+
|                               MojoPQC-SCA Core Dataflow                                 |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|  [Raw HDF5 Container] (100k traces, L=41,800 samples)                                   |
|            |                                                                            |
|            v                                                                            |
|  [Stage 1: Streaming Ingestion]  --> Lazy generator chunk iterator (B=256, Peak <112 MB) |
|            |                                                                            |
|            v                                                                            |
|  [Stage 2: Digital Filtering]    --> Zero-phase FIR low-pass filter (N=31, fc=0.25)     |
|            |                                                                            |
|            v                                                                            |
|  [Stage 3: Phase Alignment]      --> Template cross-correlation alignment (Δτ = ±100)   |
|            |                                                                            |
|            v                                                                            |
|  [Stage 4: POI Decimation]       --> Downsample to D=5,000 salient NTT butterfly features|
|            |                                                                            |
|            v                                                                            |
|  [Stage 5: Z-Score Scaling]      --> Zero-mean, unit-variance per-trace normalization   |
|            |                                                                            |
|            v                                                                            |
|  [Stage 6: Hybrid Profiler]      --> 1D-CNN Backbone + Quantum-Inspired MPS Head (χ=8)   |
|                                      Total Parameters: 9,226 (<5% of 200k ceiling)      |
|            |                                                                            |
|            v                                                                            |
|  [Stage 7: Edge Runtime & Audit] --> Dynamic Int8 ONNX Runtime (0.14 ms/tr on CPU)       |
|                                  --> Cryptographic SHA-256 Provenance Manifest          |
|                                                                                         |
+-----------------------------------------------------------------------------------------+
```

---

## Empirical Results & Benchmark Highlights

All benchmarks were recorded on standard consumer commodity hardware (x86_64 CPU, 16 GB RAM, zero dedicated GPU requirement):

### 1. Ingestion & Preprocessing Throughput
| Pipeline Implementation | Dataset Size | Raw Length | Peak Memory (RSS) | Throughput | Acceleration |
|---|---|---|---|---|---|
| **Monolithic Python/NumPy** | $100,000$ | $41,800$ | $>16.5\text{ GB}$ (OOM) | Failed (Crashed) | — |
| **Standard Chunked SciPy** | $100,000$ | $41,800$ | $251.5\text{ MB}$ | $85.4\text{ traces/s}$ | $1.00\times$ (Baseline) |
| **NumPy Optimized Chunked** | $100,000$ | $41,800$ | $100.9\text{ MB}$ | $557.7\text{ traces/s}$ | $6.53\times$ |
| **MojoPQC-SCA Streaming Engine** | $\mathbf{100,000}$ | $\mathbf{41,800}$ | $\mathbf{111.8\text{ MB}}$ ($>99.3\%$ drop) | $\mathbf{761.2\text{ traces/s}}$ | $\mathbf{8.91\times}$ |

### 2. Model Complexity & Parameter Footprint
| Model Architecture | Conv Feature Params | Head Classifier Params | Total Trainable Params | Epoch Time | Budget (<200k) |
|---|---|---|---|---|---|
| **1D-CNN Baseline (Dense Head)** | $5,984$ | $8,416$ | $14,400$ | $0.1065\text{ s}$ | PASSED (7.2%) |
| **CNN + MPS (Tensor Network, Ours)** | $\mathbf{5,984}$ | $\mathbf{3,242}$ | $\mathbf{9,226}$ ($\mathbf{35.93\%}$ compression) | $\mathbf{0.0630\text{ s}}$ ($\mathbf{1.69\times}$ faster) | **PASSED (4.61%)** |

### 3. Key Recovery via Guessing Entropy ($GE$)
| Target Implementation | Initial Rank ($m=1$) | GE @ $m=20$ | GE @ $m=50$ | GE @ $m=100$ | Final Key Rank |
|---|---|---|---|---|---|
| **Unprotected ML-KEM-512** | $128.0$ (Random) | $5.0$ | $2.1$ | $1.0$ | **1.0 (Unique Recovery)** |
| **Masked ML-KEM-512 (ARM Cortex-M4)** | $128.0$ (Random) | $38.2$ | $18.4$ | $4.2$ | **1.0 (Bypassed Masking)** |

### 4. Edge Quantization & Local CPU Inference Latency
| Deployment Format | Precision | Serialized Size | CPU Latency / Trace | Sustained Throughput |
|---|---|---|---|---|
| **PyTorch Checkpoint** | FP32 | $58.4\text{ KB}$ | $0.92\text{ ms}$ | $1,086\text{ traces/s}$ |
| **ONNX Runtime (Unquantized)** | FP32 | $42.1\text{ KB}$ | $0.38\text{ ms}$ | $2,631\text{ traces/s}$ |
| **ONNX Runtime (Dynamic Int8, Ours)** | **Int8** | $\mathbf{18.6\text{ KB}}$ ($68.2\%$ reduction) | $\mathbf{0.14\text{ ms}}$ | $\mathbf{7,142\text{ traces/s}}$ |

---

## Comparison with Existing Alternatives

The table below summarizes how **MojoPQC-SCA** compares against the prevailing side-channel profiling paradigms:

| Profiling Paradigm / Framework | Ingestion RAM ($100\text{k}$) | Trainable Parameters | Hardware Dependency | Masking Bypass ($d=2$) | CPU Latency / Trace | Audit Trail |
|---|---|---|---|---|---|---|
| **Classical 2nd-Order CPA (SCALib / Lascar)** | $>4.2\text{ GB}$ | N/A (Statistical) | CPU Only | Weak (Combinatorial failure under jitter) | $>12.40\text{ ms}$ (Pairwise) | Manual scripts |
| **ASCAD Baseline CNN (Dense Head)** | $>16.5\text{ GB}$ (OOM) | $420,000+$ (Head explosion) | High-VRAM GPU | Moderate (Prone to noise overfitting) | $1.25\text{ ms}$ (FP32) | Static scripts |
| **Heavyweight Deep ResNet-18 Profiler** | $>16.5\text{ GB}$ (OOM) | $>1,200,000$ | Dedicated GPU (CUDA) | High (Slow convergence) | $4.80\text{ ms}$ (FP32) | None |
| **Standard 1D-CNN Baseline (Dense Head)** | $100.9\text{ MB}$ | $14,400$ | Commodity CPU / GPU | Moderate ($GE \to 1.0$) | $0.38\text{ ms}$ (ONNX FP32) | SHA-256 Manifest |
| **MojoPQC-SCA (Ours: 1D-CNN + MPS)** | $\mathbf{111.8\text{ MB}}$ | $\mathbf{9,226}$ | **Commodity CPU Only** | **High ($GE \to 1.0$ in $\le 150$ traces)** | $\mathbf{0.14\text{ ms}}$ (Int8 ONNX) | **Automated SHA-256** |

### Why MojoPQC-SCA is Superior:
1. **No Out-of-Memory Failures:** Bounded streaming chunking ($B=256$) eliminates the $16.7\text{ GB}$ memory wall, running large datasets within $\le 111.8\text{ MB}$ of RAM.
2. **Anti-Overfitting Low-Rank Regularization:** Unlike massive ResNet/VGG models ($>1.2\text{M}$ params) or dense heads ($>420\text{k}$ params) that memorize oscilloscope noise, the MPS tensor network ($\chi = 8$) strictly bounds virtual entanglement entropy, filtering uncorrelated noise while extracting multi-share leakage.
3. **Automated Feature Extraction:** Overcomes the $\mathcal{O}(L^2)$ combinatorial bottleneck of 2nd-order CPA by extracting shift-invariant features via 1D-CNN convolutions.
4. **Edge Deployment Without GPUs:** Dynamic Int8 ONNX quantization runs at $0.14\text{ ms per trace}$ ($>7,100\text{ traces/s}$) on standard consumer laptop CPUs.

---

## Interactive Developer Console & Web Testbed

MojoPQC-SCA includes a browser-based, dark-mode developer console and live oscilloscope testbed built with a terminal aesthetic (`#000000` canvas, zinc typography, monochrome charts, zero layout jumping).

The console includes 6 functional tabs:
1. **Overview & Execution Summary:** Real-time pipeline status, phase completion indicators, and parameter budget compliance ($9,226 / 200,000$).
2. **Oscilloscope Waveforms:** Multi-trace interactive waveform visualizer showing raw traces, zero-phase FIR filtered waveforms, phase alignment translations, and decimated $5,000$-sample POI regions.
3. **Model Architecture:** Visual tensor dimension comparison between the 1D-CNN baseline ($14,400$ params) and the CNN+MPS tensor network ($9,226$ params).
4. **Attack Testbed & Key Recovery:** Real-time Guessing Entropy ($GE$) curve tracking convergence towards Rank $1.0$ across attack trace prefixes.
5. **Inference Testbed:** Live edge inference simulator executing the dynamic Int8 ONNX model with latency distribution and candidate byte posterior probabilities.
6. **Audit & Manifest:** Cryptographic SHA-256 fingerprint verification table auditing all datasets, models, and configuration files.

---

## Quick Start (One-Click Demo)

With the [`uv`](https://docs.astral.sh/uv/) runtime installed, launch the entire pipeline and developer testbed with a single command:

```powershell
# Run smoke demo, execute pipeline, and open the web dashboard at http://127.0.0.1:8000
uv run demo.py
```

### Command Options:
```powershell
# Run full-size training and benchmarking:
uv run demo.py --full

# Train on physical masked ML-KEM-512 power traces:
uv run demo.py --dataset masked

# Train on unprotected ML-KEM-512 traces:
uv run demo.py --dataset unprotected

# Terminal-only execution (runs full pipeline without starting web server):
uv run demo.py --cli-only

# Start dashboard immediately from existing artifacts without retraining:
uv run demo.py --skip-pipeline

# Run on a custom port without auto-launching browser:
uv run demo.py --port 8080 --no-browser
```

---

## Step-by-Step Modular Pipeline

You can also run every pipeline stage individually using the standalone scripts in `scripts/`:

```powershell
# 0. Check Environment & Tooling
python scripts/00_check_environment.py

# 1. Generate Deterministic Synthetic NTT Traces (or convert real data)
python scripts/01_generate_synthetic_dataset.py --profiling 32 --attack 8 --trace-length 20000

# 2. Run Streaming Preprocessing Baseline (FIR, Alignment, POI decimation)
python scripts/02_python_preprocess_baseline.py

# 3. Validate Dataset Contract (Non-finite checks, label ranges)
python scripts/12_validate_dataset.py --input data/processed/python_processed.h5 --strict

# 4. Train 1D-CNN Baseline
python scripts/04_train_cnn.py --epochs 10 --batch-size 16 --device auto

# 5. Train Quantum-Inspired CNN + MPS Model
python scripts/05_train_cnn_mps.py --epochs 10 --batch-size 16 --bond-dim 8 --device auto

# 6. Evaluate Guessing Entropy (Key Rank Convergence)
python scripts/06_evaluate_ge.py --input data/processed/python_processed.h5

# 7. Benchmark Model & Preprocessing Footprints
python scripts/07_benchmark_models.py
python scripts/08_benchmark_preprocessing.py

# 8. Export to Dynamic Int8 ONNX & Benchmark CPU Inference Latency
python scripts/09_export_onnx.py
python scripts/10_local_inference.py

# 9. Generate Cryptographic Provenance Manifest
python scripts/13_create_run_manifest.py --dataset data/processed/python_processed.h5
```

---

## Available Datasets (Real & Synthetic)

The repository provides both reproducible synthetic benchmarks and physical acquisition campaigns:

| Dataset Identifier | Location | Traces & Samples | Description |
|---|---|---|---|
| **Synthetic PQC Baseline** | `data/raw/synthetic_pqc.h5` | 32/8 up to 10k/2k, $L=5,000$ | Deterministic NTT leakage generator for rapid testing, CI, and smoke verification. |
| **Real Masked ML-KEM-512** | `data/raw/mlkem_masked_converted.h5` | 100,000 prof / 100,000 atk, $L=41,800$ | Physical power acquisitions from 32-bit ARM Cortex-M4 running 1st-order Boolean masked ML-KEM-512 (Zenodo). |
| **Real Unprotected ML-KEM-512** | `data/raw/mlkem_unprotected_converted.h5` | 10,000 prof / 10,000 atk, $L=28,600$ | Physical power traces from ARM Cortex-M4 without countermeasures for leakage comparison. |
| **Kyber Case-Study Matrices** | `data/metadata/confusion_matrices/` | N/A | KU Leuven reference matrices and belief-propagation artifacts. |

### Ingesting External Zenodo / PQM4 Datasets:
Inspect external HDF5 files without loading trace payloads:
```powershell
python scripts/11_convert_external_hdf5.py --input path/to/external_traces.h5 --inspect
```

Convert external traces into the project streaming contract:
```powershell
python scripts/11_convert_external_hdf5.py `
  --input path/to/external_traces.h5 `
  --output data/raw/mlkem_converted.h5 `
  --label-byte 0 `
  --profiling-limit 10000 `
  --attack-limit 2000
```

---

## Hardware Cryptographic Audit & Reproducibility

Every execution of the pipeline automatically records an immutable, verifiable manifest in `results/benchmarks/run_manifest.json`:
- **Git Commit & Dirty State:** Exact code snapshot.
- **Environment Metadata:** OS, CPU architecture, Python version, library versions.
- **SHA-256 Digests:** Cryptographic fingerprinting for raw HDF5 files, preprocessed containers, model weights (`.pt`), and ONNX runtime graphs (`.onnx`).
- **Dataset Contract Verification:** Enforces shape consistency, absence of `NaN`/`Inf` values, and valid class boundaries $[0, 255]$.

---

## Research Paper & Explanation Guide

This project includes a complete academic manuscript ready for submission to top-tier security and AI venues (AAAI / CHES / IEEE S&P):

- **Research Manuscript:** [`paper/main.tex`](paper/main.tex)
  - Standard AAAI 2-column format with unnumbered subheadings (`secnumdepth = 1`).
  - 58 embedded academic citations with clickable DOIs and URLs in [`paper/references.bib`](paper/references.bib) and [`paper/main.bbl`](paper/main.bbl).
  - Formal pseudocode in **Algorithm 1**, complete architectural specs in **Table 1–5**, and comprehensive comparative benchmarks in **Table 6**.
- **Explainer & Presentation Script:** [`EXPLAINER_AND_PRESENTATION_GUIDE.md`](EXPLAINER_AND_PRESENTATION_GUIDE.md)
  - Plain-English breakdown of all project concepts.
  - 60-second elevator pitch and 5-minute technical defense walkthrough script.
  - Key statistics reference table and prepared answers for examiner/reviewer questions.

---

## Repository Structure

```text
NeuralSCA/
├── config/                     # YAML pipeline configurations (default.yaml, colab.yaml)
├── data/
│   ├── raw/                    # Raw & converted HDF5 trace files
│   ├── processed/              # Filtered, aligned, and decimated HDF5 datasets
│   └── metadata/               # Acquisition metadata & confusion matrices
├── mojopqc_sca/
│   ├── datasets/               # Chunked HDF5 loaders, IterableDataset & contract validation
│   ├── models/                 # 1D-CNN baseline, MPS tensor layers, parameter counters
│   ├── preprocessing/          # Zero-phase FIR filtering, template alignment, POI decimation
│   └── evaluation/             # Cumulative log-likelihood, Guessing Entropy & rank metrics
├── scripts/                    # Numbered standalone pipeline execution scripts (00 to 13)
├── results/
│   ├── models/                 # Trained checkpoints (.pt) and Int8 ONNX graphs (.onnx)
│   ├── benchmarks/             # Throughput, memory, latency, and run_manifest.json
│   └── figures/                # Waveforms, parameter comparisons, and GE convergence plots
├── paper/                      # AAAI LaTeX research paper, references.bib & main.bbl
├── demo.py                     # Unified runner and interactive developer console web testbed
├── EXPLAINER_AND_PRESENTATION_GUIDE.md # Complete project explanation & defense script
├── pyproject.toml              # Modern Python packaging configuration
└── requirements.txt            # Declared dependencies
```

---

## Citation

If you use MojoPQC-SCA / NeuralSCA in your research, please cite our paper:

```bibtex
@article{kumar2026accelerated,
  author    = {Kumar, Hardik},
  title     = {{Accelerated Neural Side-Channel Analysis of Masked ML-KEM Using Quantum-Inspired Architectures}},
  journal   = {School of Computer Science and Engineering, Vellore Institute of Technology, Andhra Pradesh},
  year      = {2026},
  note      = {Artifact repository: \url{https://github.com/hardikxk/pqc-sca}}
}
```

---

## License
This project is licensed under the [MIT License](LICENSE).
