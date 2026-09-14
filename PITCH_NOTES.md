# MojoPQC-SCA — Personal Pitch Notes

## One-sentence pitch

MojoPQC-SCA is a CPU-first, reproducible side-channel analysis pipeline for masked ML-KEM/Kyber traces that combines streaming trace preprocessing, lightweight CNN and MPS-based models, guessing-entropy evaluation, and ONNX CPU inference.

## 30-second pitch

Side-channel analysis projects often assume access to expensive GPUs and large specialized tooling. MojoPQC-SCA explores a more deployable workflow: traces are processed in bounded-memory chunks, reduced to a lightweight 5,000-sample representation, classified with models under 200,000 parameters, evaluated using guessing entropy, and exported for CPU inference. The current repository proves the software workflow end to end using deterministic synthetic traces, with native Mojo acceleration and real hardware-trace validation planned as the next research steps.

## Problem

Masked post-quantum implementations such as ML-KEM need practical side-channel evaluation. Raw traces are large, noisy, and expensive to process. A useful research workflow should be reproducible, memory-conscious, and capable of running without a local GPU.

## What this project contributes

1. A deterministic synthetic ML-KEM-style trace generator for development.
2. Streaming HDF5 preprocessing with normalization, filtering, alignment, and downsampling.
3. A very small 1-D CNN baseline.
4. A compact Matrix Product State classifier head as a tensor-network alternative.
5. Guessing-entropy/key-rank evaluation based on cumulative trace evidence.
6. JSON, CSV, LaTeX, and PNG experiment artifacts.
7. ONNX, int8, and CPU inference support.
8. Unit tests and GitHub Actions CPU smoke CI on Windows and Ubuntu.
9. External HDF5 inspection/conversion and bounded-memory dataset validation reports.
10. Reproducibility manifests with configuration, environment, Git state, and artifact hashes.

## Architecture explanation

```text
Raw HDF5 traces
        |
        v
Chunked preprocessing
  normalize -> FIR filter -> cross-correlation alignment -> 5,000 samples
        |
        +--> Lightweight CNN -------> class logits
        |
        +--> CNN feature extractor
                 -> MPS tensor-network head -> class logits
                                      |
                                      v
                    rank / guessing entropy / ONNX CPU inference
```

Mojo is an optional acceleration boundary for the preprocessing kernels. The current implementation safely uses the Python reference/fallback path when Mojo is unavailable.

## What to demonstrate

From the repository root:

```powershell
python scripts/00_check_environment.py
python -m unittest discover -s tests -v
python scripts/01_generate_synthetic_dataset.py --profiling 16 --attack 8 --trace-length 2000
python scripts/02_python_preprocess_baseline.py
python scripts/12_validate_dataset.py --input data/processed/python_processed.h5 --strict
python scripts/13_create_run_manifest.py --dataset data/processed/python_processed.h5
python scripts/04_train_cnn.py --epochs 1 --batch-size 4
python scripts/05_train_cnn_mps.py --epochs 1 --batch-size 4
python scripts/06_evaluate_ge.py --step 1
python scripts/09_export_onnx.py
python scripts/10_local_inference.py --batch-size 4
```

Show the generated files in `results/`: the dataset validation report, model checkpoints, GE curve, benchmark tables, ONNX models, and inference JSON.

## Honest status statement

Say this explicitly:

> The current release is an end-to-end software prototype validated on deterministic synthetic traces. It demonstrates the data path, model path, evaluation path, deployment path, tests, and CI. Real ML-KEM hardware traces have not yet been collected or integrated, so the current metrics are engineering smoke-test measurements rather than final security claims.

## Strengths

- Runs without NVIDIA hardware or CUDA.
- Does not load the entire trace dataset into RAM.
- Small model sizes: approximately 14,400 CNN parameters and 9,226 CNN+MPS parameters.
- Reproducible configuration and deterministic smoke data.
- Clear separation between preprocessing, models, evaluation, and deployment.
- Automated Windows/Ubuntu CI checks.

## Current limitations

- Synthetic traces are not a substitute for a physical acquisition campaign.
- The current native Mojo kernels are placeholders; measured preprocessing speedups are not Mojo speedups.
- Training defaults safely to CPU and supports explicit `auto`, `cpu`, or `cuda` selection for Colab/GPU runs.
- The tiny demo dataset is too small for meaningful accuracy or guessing-entropy conclusions.
- External ML-KEM datasets require format conversion, validation, a reproducibility manifest, and a defensible sensitive-label definition.

## Likely review questions

### Why no CUDA?

The design target is a CPU-first development and inference environment. Training is intentionally lightweight, defaults to automatic device selection, and can use CUDA in Colab or another compatible environment. CUDA is an optimization, not a correctness requirement.

### Why use an MPS layer?

The MPS head is a compact tensor-network alternative to a dense classifier head. It tests whether structured low-rank contraction can maintain useful classification capacity with a small parameter budget.

### Are the current results final?

No. They confirm that the software pipeline works. Final research conclusions require larger datasets, repeated runs, native Mojo measurements, and real ML-KEM hardware traces.

### What is the next milestone?

Convert a real ML-KEM HDF5 dataset into the project schema, define a defensible sensitive intermediate/labeling strategy, run multiple seeds and attack partitions, and then implement/benchmark the native Mojo preprocessing kernels.

## Closing sentence

MojoPQC-SCA is currently a reproducible and deployable research foundation: the engineering pipeline is complete enough to review, while real-trace validation and native Mojo acceleration define the next experimental phase.
