# MojoPQC-SCA — Final Unified Build Plan

This document is a complete, self-contained implementation plan for building MojoPQC-SCA, a high-performance deep learning side-channel analysis pipeline for masked ML-KEM/Kyber traces.

The plan is optimized for the following constraints:

- No local GPU is available.
- Heavy models cannot be trained locally.
- Training should be done using free cloud GPU resources such as Google Colab.
- Local execution should be limited to preprocessing, inference, benchmarking, and evaluation.
- Models must remain lightweight.
- The project should maximize performance efficiency and publication quality.

This plan can be handed directly to an AI coding agent or used as a personal development roadmap.

---

## 1. Project Summary

Project Name: MojoPQC-SCA

Goal: Build a high-performance deep learning side-channel analysis pipeline for masked ML-KEM.

Main Constraint: No local GPU. Lightweight models only.

Strategy:

- Use Mojo for high-speed preprocessing on CPU.
- Use a lightweight CNN + MPS tensor-network model.
- Train on Google Colab using free GPU.
- Export the trained model to ONNX for local CPU inference.
- Apply int8 quantization for efficient local inference.

---

## 2. Final Technology Stack

| Layer | Primary Choice | Fallback |
|---|---|---|
| Preprocessing | Mojo kernels | Numba JIT |
| Deep Learning | PyTorch | TensorFlow/Keras |
| Tensor Network | TorchMPS or custom MPS layer | Low-rank SVD layer |
| Training Compute | Google Colab free T4 GPU | Kaggle Notebooks |
| Local Inference | ONNX Runtime | PyTorch CPU |
| Model Compression | int8 dynamic quantization | Pruning |
| Dataset Phase 1 | Synthetic PQC traces | ASCAD public dataset |
| Dataset Phase 2 | Real ML-KEM traces if hardware is available | Continue with synthetic traces |
| Data Format | HDF5 and memory-mapped binary | NumPy .npy |

---

## 3. Repository Structure

The project must use the following structure:

    MojoPQC-SCA/
    │
    ├── README.md
    ├── requirements.txt
    ├── pyproject.toml
    ├── Makefile
    │
    ├── config/
    │   ├── default.yaml
    │   └── colab.yaml
    │
    ├── data/
    │   ├── raw/
    │   ├── processed/
    │   └── metadata/
    │
    ├── scripts/
    │   ├── 00_check_environment.py
    │   ├── 01_generate_synthetic_dataset.py
    │   ├── 02_python_preprocess_baseline.py
    │   ├── 03_run_mojo_preprocess.py
    │   ├── 04_train_cnn.py
    │   ├── 05_train_cnn_mps.py
    │   ├── 06_evaluate_ge.py
    │   ├── 07_benchmark_models.py
    │   ├── 08_benchmark_preprocessing.py
    │   ├── 09_export_onnx.py
    │   └── 10_local_inference.py
    │
    ├── mojopqc_sca/
    │   ├── __init__.py
    │   ├── datasets/
    │   │   ├── synthetic_pqc.py
    │   │   ├── hdf5_loader.py
    │   │   └── memmap_loader.py
    │   ├── preprocessing/
    │   │   ├── python_baseline.py
    │   │   ├── normalize.py
    │   │   ├── filtering.py
    │   │   └── alignment.py
    │   ├── models/
    │   │   ├── cnn.py
    │   │   ├── cnn_mps.py
    │   │   └── mps_layer.py
    │   ├── evaluation/
    │   │   ├── guessing_entropy.py
    │   │   └── metrics.py
    │   └── utils/
    │       ├── config.py
    │       └── benchmark.py
    │
    ├── mojo_kernels/
    │   ├── normalize.mojo
    │   ├── fir_filter.mojo
    │   ├── align_xcorr.mojo
    │   └── preprocess.mojo
    │
    ├── notebooks/
    │   ├── colab_train.ipynb
    │   └── results_visualization.ipynb
    │
    ├── results/
    │   ├── benchmarks/
    │   ├── figures/
    │   ├── tables/
    │   ├── models/
    │   └── logs/
    │
    └── paper/
        └── main.tex

---

## 4. Performance Design Rules

The following rules must be followed in every module:

1. Never load the full dataset into RAM. Use HDF5 chunked reads or memory-mapped files.

2. All preprocessing must operate in-place or stream data in chunks.

3. The final model must have fewer than 200,000 trainable parameters.

4. Trace input to the neural network must be reduced to at most 5,000 samples using downsampling or region selection.

5. Training must run on Google Colab. Local scripts must only perform preprocessing, inference, evaluation, and benchmarking.

6. All benchmark results must be saved as JSON and CSV.

7. Every script must print execution time and peak memory usage.

---

## 5. Phased Execution Plan

---

## Phase 0 — Environment Setup

### Goal

Verify that all dependencies are installed and working.

### Tasks

1. Create requirements.txt with the following dependencies:

    numpy
    scipy
    h5py
    matplotlib
    seaborn
    pandas
    scikit-learn
    torch
    tqdm
    pyyaml
    psutil
    memory-profiler
    onnx
    onnxruntime

2. Create scripts/00_check_environment.py.

The script must check:

- Python version
- NumPy
- SciPy
- h5py
- PyTorch
- CUDA availability
- ONNX Runtime
- Mojo availability

If CUDA or Mojo is unavailable, the script should print a warning but must not fail.

3. Create a Makefile with common commands:

    setup:
        pip install -r requirements.txt

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

    benchmark:
        python scripts/08_benchmark_preprocessing.py

### Acceptance Criteria

The following command must run without crashing:

    python scripts/00_check_environment.py

---

## Phase 1 — Synthetic Dataset Generation

### Goal

Create a realistic PQC-like trace dataset without requiring hardware.

### Tasks

1. Create:

    mojopqc_sca/datasets/synthetic_pqc.py

2. Generate synthetic traces that simulate ML-KEM NTT leakage.

Suggested generation logic:

    For each trace:
        length = 100,000 samples
        label = random integer from 0 to 255
        hw = hamming_weight(label)

        trace = gaussian_noise(length)

        ntt_start = random integer from 20000 to 30000
        jitter = random integer from -50 to 50

        for butterfly in range(8):
            pos = ntt_start + butterfly * 5000 + jitter
            trace[pos : pos + 200] += hw * leakage_amplitude

        trace += random_dc_offset

3. Save dataset as HDF5:

    data/raw/synthetic_pqc.h5

Required HDF5 structure:

    /traces/profiling    shape: (10000, 100000)
    /labels/profiling    shape: (10000,)
    /traces/attack       shape: (2000, 100000)
    /labels/attack       shape: (2000,)
    /metadata/config     JSON string

4. Create script:

    scripts/01_generate_synthetic_dataset.py

### Acceptance Criteria

The following command must create the dataset:

    python scripts/01_generate_synthetic_dataset.py

The script must print dataset shapes and file location.

---

## Phase 2 — Python Preprocessing Baseline

### Goal

Build the slow but correct Python preprocessing baseline.

This baseline will be used to prove that Mojo or Numba acceleration is necessary.

### Tasks

1. Create:

    mojopqc_sca/preprocessing/python_baseline.py

2. Implement the following functions:

    def normalize_traces(traces):
        mean = traces.mean(axis=1, keepdims=True)
        std = traces.std(axis=1, keepdims=True)
        return (traces - mean) / (std + 1e-8)

    def fir_filter(traces, kernel_size=11):
        kernel = np.ones(kernel_size) / kernel_size
        return np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode="same"),
            axis=1,
            arr=traces
        )

    def align_xcorr(traces, reference, max_shift=500):
        # Cross-correlation-based trace alignment.
        ...

    def downsample_traces(traces, target_len=5000):
        # Reduce trace length for lightweight model input.
        ...

    def preprocess_pipeline(input_path, output_path, config):
        # Load HDF5 in chunks.
        # Normalize.
        # Filter.
        # Align.
        # Downsample to 5000 samples.
        # Save processed HDF5.
        # Record execution time and peak memory.

3. Create script:

    scripts/02_python_preprocess_baseline.py

4. Save benchmark results to:

    results/benchmarks/python_preprocess.json

### Acceptance Criteria

The following command must complete successfully:

    python scripts/02_python_preprocess_baseline.py

It must print execution time and peak memory.

It must create a processed HDF5 file and a benchmark JSON file.

---

## Phase 3 — Mojo Preprocessing Kernels

### Goal

Replace the Python preprocessing bottleneck with Mojo kernels.

### Tasks

1. Create Mojo kernels:

    mojo_kernels/normalize.mojo
    mojo_kernels/fir_filter.mojo
    mojo_kernels/align_xcorr.mojo
    mojo_kernels/preprocess.mojo

2. The main Mojo entry point must be:

    mojo_kernels/preprocess.mojo

Its logic should be:

    Read input traces from binary or HDF5-compatible raw file.

    For each trace chunk:
        normalize trace
        apply FIR filter using SIMD
        align trace using cross-correlation with SIMD
        downsample trace

    Write output as memory-mapped binary or NumPy-compatible file.

    Print total execution time.

3. Create:

    scripts/03_run_mojo_preprocess.py

This script must:

- Call the Mojo kernel using subprocess.
- Pass input and output paths.
- Wait for completion.
- Validate the output file.
- Save benchmark results.

4. Create a mandatory fallback:

    mojopqc_sca/preprocessing/numba_fallback.py

If Mojo is not installed, the system must automatically use the Numba fallback and print a warning.

### Acceptance Criteria

The following command must run:

    python scripts/03_run_mojo_preprocess.py

It must produce:

    data/processed/mojo_processed.bin
    results/benchmarks/mojo_preprocess.json

If Mojo is unavailable, it must fall back to Numba without crashing.

---

## Phase 4 — Lightweight CNN Baseline Model

### Goal

Build a lightweight 1D-CNN baseline for trace classification.

### Tasks

1. Create:

    mojopqc_sca/models/cnn.py

2. Use the following lightweight architecture:

    class LightweightCNN(nn.Module):
        def __init__(self, input_len=5000, num_classes=256):
            super().__init__()

            self.features = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.AvgPool1d(10),

                nn.Conv1d(16, 32, kernel_size=11, stride=1, padding=5),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AvgPool1d(10),
            )

            self.pool = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Linear(32, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x).squeeze(-1)
            return self.classifier(x)

This model must remain extremely lightweight.

3. Create:

    scripts/04_train_cnn.py

Training configuration:

    optimizer: Adam
    learning_rate: 0.001
    epochs: 30
    batch_size: 128
    loss: CrossEntropyLoss

4. Save outputs:

    results/models/cnn_baseline.pt
    results/logs/cnn_training.csv

### Acceptance Criteria

The following command must complete successfully:

    python scripts/04_train_cnn.py

It must print validation accuracy and save the model checkpoint.

---

## Phase 5 — CNN + MPS Tensor Network Model

### Goal

Implement the quantum-inspired tensor-network compression layer.

### Tasks

1. Create:

    mojopqc_sca/models/mps_layer.py

2. Implement an MPS layer conceptually as follows:

    class MPSLayer(nn.Module):
        """
        Matrix Product State classifier layer.

        Replaces a dense classification layer with a low-rank
        tensor contraction.
        """

        def __init__(self, input_dim, num_classes, bond_dim=8):
            super().__init__()

            self.num_classes = num_classes
            self.bond_dim = bond_dim

            self.num_sites = int(math.ceil(math.log2(input_dim)))

            self.cores = nn.ParameterList([
                nn.Parameter(torch.randn(bond_dim, 2, bond_dim) * 0.01)
                for _ in range(self.num_sites)
            ])

            self.output = nn.Linear(bond_dim, num_classes)

        def forward(self, x):
            # Normalize features.
            # Reshape into MPS-compatible site structure.
            # Contract input with MPS cores.
            # Project final bond dimension to class logits.
            ...

3. Create:

    mojopqc_sca/models/cnn_mps.py

Architecture:

    class CNNWithMPS(nn.Module):
        def __init__(self, input_len=5000, num_classes=256, bond_dim=8):
            super().__init__()

            self.features = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.AvgPool1d(10),

                nn.Conv1d(16, 32, kernel_size=11, stride=1, padding=5),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AvgPool1d(10),
            )

            self.pool = nn.AdaptiveAvgPool1d(1)
            self.mps = MPSLayer(32, num_classes, bond_dim)

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x).squeeze(-1)
            return self.mps(x)

4. Create:

    scripts/05_train_cnn_mps.py

### Acceptance Criteria

The following command must complete successfully:

    python scripts/05_train_cnn_mps.py

The model must print its parameter count.

The parameter count must be below:

    200,000 trainable parameters

---

## Phase 6 — Guessing Entropy Evaluation

### Goal

Measure side-channel attack performance using Guessing Entropy.

### Tasks

1. Create:

    mojopqc_sca/evaluation/guessing_entropy.py

2. Implement the following functions:

    def compute_key_rank(log_probs, true_label):
        """
        Compute the rank of the true key candidate
        among all predicted candidates.
        """
        ...

    def compute_guessing_entropy(log_probs, true_labels):
        """
        Compute average rank of the correct key.
        """
        ...

    def compute_ge_curve(log_probs, true_labels, step=100):
        """
        Compute GE as a function of number of attack traces.
        """
        ...

3. Create:

    scripts/06_evaluate_ge.py

4. Save outputs:

    results/figures/ge_curve.png
    results/benchmarks/ge_results.json

### Acceptance Criteria

The following command must generate a GE curve:

    python scripts/06_evaluate_ge.py

It must print:

    GE @ 100 traces
    GE @ 500 traces
    GE @ 1000 traces
    GE @ all traces

---

## Phase 7 — Model Benchmarking

### Goal

Compare CNN baseline against CNN + MPS.

### Tasks

1. Create:

    scripts/07_benchmark_models.py

2. Compare the following metrics:

    Model name
    Total parameters
    Training time per epoch
    Peak memory usage
    Validation accuracy
    GE at 100 traces
    GE at 500 traces
    GE at 1000 traces

3. Save outputs:

    results/benchmarks/model_comparison.csv
    results/tables/model_comparison.tex
    results/figures/model_comparison.png

### Acceptance Criteria

The script must generate a paper-ready comparison table.

---

## Phase 8 — Preprocessing Benchmarking

### Goal

Prove the performance benefit of Mojo preprocessing.

### Tasks

1. Create:

    scripts/08_benchmark_preprocessing.py

2. Compare:

    Python preprocessing time
    Mojo preprocessing time
    Numba fallback time, if Mojo is unavailable
    Peak memory usage
    Throughput in traces per second
    Speedup factor

3. Save outputs:

    results/benchmarks/preprocessing_comparison.csv
    results/tables/preprocessing_comparison.tex
    results/figures/preprocessing_speedup.png

### Acceptance Criteria

The script must generate a paper-ready preprocessing comparison table.

---

## Phase 9 — ONNX Export and Local Inference

### Goal

Enable fast local CPU inference without requiring PyTorch or GPU.

### Tasks

1. Create:

    scripts/09_export_onnx.py

2. The script must:

    Load trained CNN + MPS model
    Export model to ONNX
    Apply int8 dynamic quantization
    Save ONNX model

3. Save outputs:

    results/models/cnn_mps.onnx
    results/models/cnn_mps_int8.onnx

4. Create:

    scripts/10_local_inference.py

5. The local inference script must:

    Load ONNX model
    Run inference on attack traces
    Compute Guessing Entropy
    Print inference time per trace

### Acceptance Criteria

Local inference must work on CPU using ONNX Runtime.

---

## Phase 10 — Google Colab Training Notebook

### Goal

Allow GPU training without requiring local GPU resources.

### Tasks

1. Create:

    notebooks/colab_train.ipynb

2. The notebook must contain:

    Cell 1: Install dependencies
    Cell 2: Mount Google Drive
    Cell 3: Load processed dataset
    Cell 4: Define CNN and CNN + MPS models
    Cell 5: Train both models
    Cell 6: Save model weights to Google Drive
    Cell 7: Provide download instructions

3. Add Colab instructions to README.md.

### Acceptance Criteria

The notebook must run end-to-end on Google Colab free tier.

---

## Phase 11 — Final Results and Paper Integration

### Goal

Generate all paper-ready figures and tables.

### Tasks

Run the full pipeline in order:

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

### Final Outputs

The following outputs must exist:

    results/tables/preprocessing_comparison.tex
    results/tables/model_comparison.tex
    results/figures/ge_curve.png
    results/figures/preprocessing_speedup.png
    results/figures/model_comparison.png
    results/benchmarks/preprocessing_comparison.csv
    results/benchmarks/model_comparison.csv
    results/benchmarks/ge_results.json

---

## 6. AI Agent Instruction Prompt

Copy and paste the following block into an AI coding agent.

    You are building a research prototype called MojoPQC-SCA.

    PROJECT GOAL:
    Build a high-performance deep learning side-channel analysis
    pipeline for masked ML-KEM/Kyber traces. The pipeline must
    work without a local GPU and use lightweight models only.

    HARD CONSTRAINTS:
    1. No local GPU is available. Training must happen on Google Colab.
    2. Models must have fewer than 200,000 trainable parameters.
    3. Trace input to the model must be at most 5,000 samples.
    4. All local operations must run on CPU.
    5. Never load the full dataset into RAM.
       Use chunked or memory-mapped I/O.

    TECHNOLOGY STACK:
    - Preprocessing: Mojo kernels with Numba JIT fallback
    - Deep Learning: PyTorch
    - Tensor Network: Custom MPS layer in PyTorch
    - Training: Google Colab notebook
    - Local Inference: ONNX Runtime with int8 quantization
    - Data Format: HDF5 and memory-mapped binary
    - Benchmarking: psutil, memory-profiler, time module

    REPOSITORIES TO REFERENCE:
    - https://github.com/ANSSI-FR/ASCAD
    - https://github.com/mupq/pqm4
    - https://github.com/uclcrypto/pqm4_masked
    - https://github.com/jemisjoky/TorchMPS
    - https://github.com/google/scaaml

    PROJECT STRUCTURE:
    Use the exact repository structure defined in the project plan.

    IMPLEMENT THESE SCRIPTS IN ORDER:
    scripts/00_check_environment.py
    scripts/01_generate_synthetic_dataset.py
    scripts/02_python_preprocess_baseline.py
    scripts/03_run_mojo_preprocess.py
    scripts/04_train_cnn.py
    scripts/05_train_cnn_mps.py
    scripts/06_evaluate_ge.py
    scripts/07_benchmark_models.py
    scripts/08_benchmark_preprocessing.py
    scripts/09_export_onnx.py
    scripts/10_local_inference.py

    IMPLEMENT THESE MODULES:
    mojopqc_sca/__init__.py
    mojopqc_sca/datasets/synthetic_pqc.py
    mojopqc_sca/datasets/hdf5_loader.py
    mojopqc_sca/datasets/memmap_loader.py
    mojopqc_sca/preprocessing/python_baseline.py
    mojopqc_sca/preprocessing/normalize.py
    mojopqc_sca/preprocessing/filtering.py
    mojopqc_sca/preprocessing/alignment.py
    mojopqc_sca/preprocessing/numba_fallback.py
    mojopqc_sca/models/cnn.py
    mojopqc_sca/models/mps_layer.py
    mojopqc_sca/models/cnn_mps.py
    mojopqc_sca/evaluation/guessing_entropy.py
    mojopqc_sca/evaluation/metrics.py
    mojopqc_sca/utils/config.py
    mojopqc_sca/utils/benchmark.py

    IMPLEMENT THESE MOJO KERNELS:
    mojo_kernels/normalize.mojo
    mojo_kernels/fir_filter.mojo
    mojo_kernels/align_xcorr.mojo
    mojo_kernels/preprocess.mojo

    IMPLEMENT THESE NOTEBOOKS:
    notebooks/colab_train.ipynb
    notebooks/results_visualization.ipynb

    PERFORMANCE RULES:
    1. All preprocessing must stream in chunks.
       Never load the entire dataset into RAM.
    2. Mojo kernels must use SIMD vectorization where possible.
    3. If Mojo is not installed, automatically fall back to
       Numba JIT and print a warning.
    4. Every script must print execution time and peak memory.
    5. All benchmark results must be saved as JSON and CSV.
    6. The CNN must use at most 16 and 32 filters in its two blocks.
    7. MPS bond dimension must default to 8.

    OUTPUT REQUIREMENTS:
    Every benchmark script must produce:
    - A JSON file in results/benchmarks/
    - A CSV file in results/benchmarks/
    - A PNG figure in results/figures/
    - A LaTeX table in results/tables/

    ACCEPTANCE CRITERIA:
    1. python scripts/00_check_environment.py runs without error.
    2. Synthetic dataset generation works.
    3. Python preprocessing completes and saves benchmark results.
    4. Mojo preprocessing completes or Numba fallback activates.
    5. CNN training completes and saves a checkpoint.
    6. CNN + MPS training completes and saves a checkpoint.
    7. CNN + MPS has fewer than 200,000 parameters.
    8. Guessing Entropy curve plot is generated.
    9. Preprocessing speedup table is generated.
    10. Model comparison table is generated.
    11. ONNX export succeeds.
    12. Local CPU inference works via ONNX Runtime.
    13. Google Colab notebook runs end-to-end on free tier.
    14. README explains how to reproduce all results.

    IMPORTANT:
    Do not assume Mojo can compile arbitrary Python code.
    Use Mojo only for performance-critical kernels.
    Use Python and PyTorch for all training logic.

---

## 7. Validation Checklist

After the agent completes the implementation, verify the following:

    [ ] Project structure matches the specification.
    [ ] requirements.txt installs cleanly.
    [ ] scripts/00_check_environment.py passes.
    [ ] Synthetic HDF5 dataset is created.
    [ ] Python preprocessing benchmark is saved.
    [ ] Mojo preprocessing works or Numba fallback works.
    [ ] CNN model has fewer than 10,000 parameters.
    [ ] CNN + MPS model has fewer than 200,000 parameters.
    [ ] CNN training completes.
    [ ] CNN + MPS training completes.
    [ ] Guessing Entropy curve plot is generated.
    [ ] Preprocessing comparison table is generated.
    [ ] Model comparison table is generated.
    [ ] ONNX export succeeds.
    [ ] Local ONNX inference works on CPU.
    [ ] Google Colab notebook is provided.
    [ ] README explains full reproduction steps.

---

## 8. Expected Timeline

    Day 1:     Phase 0 + Phase 1
               Environment setup and synthetic dataset generation.

    Day 2:     Phase 2
               Python preprocessing baseline.

    Day 3-4:   Phase 3
               Mojo preprocessing kernels and Numba fallback.

    Day 5:     Phase 4
               Lightweight CNN baseline.

    Day 6-7:   Phase 5
               CNN + MPS tensor-network model.

    Day 8:     Phase 6
               Guessing Entropy evaluation.

    Day 9:     Phase 7 + Phase 8
               Model and preprocessing benchmarking.

    Day 10:    Phase 9 + Phase 10
               ONNX export and Colab training notebook.

    Day 11-12: Phase 11
               Final results, figures, tables, and paper integration.

---

## 9. Final Deliverables

At the end of the project, the repository must contain:

1. A working synthetic PQC trace generation pipeline.
2. A Python preprocessing baseline.
3. Mojo preprocessing kernels with Numba fallback.
4. A lightweight CNN baseline model.
5. A lightweight CNN + MPS tensor-network model.
6. Guessing Entropy evaluation scripts.
7. Benchmarking scripts for preprocessing and models.
8. ONNX export and local CPU inference scripts.
9. A Google Colab notebook for GPU training.
10. Paper-ready tables and figures.
11. A README explaining how to reproduce all results.

---

## 10. Recommended First Action

The first implementation step should be:

    mkdir MojoPQC-SCA
    cd MojoPQC-SCA

Then create the repository structure and start with:

    scripts/00_check_environment.py
    scripts/01_generate_synthetic_dataset.py

Do not begin with hardware acquisition.

Build the entire software pipeline using synthetic traces first. Once the pipeline works end-to-end, real traces can be swapped in without changing the rest of the system.
