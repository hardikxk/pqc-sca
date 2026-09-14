# NeuralSCA / MojoPQC-SCA: The Complete Project Guide & Explanation Cheat-Sheet

> **Purpose of this Document:**  
> Read this file first so **you** understand every piece of your project from high-level concepts down to the code and numbers.  
> Then, use the second half as your **exact presentation/defense script** to confidently explain it to professors, reviewers, examiners, or interviewers.

---

## PART 1: Understand It Yourself First (In Plain English)

### 1. The Core Problem: Why does this project exist?

1. **The Quantum Threat & NIST's New Standards:**
   - Quantum computers will soon break existing encryption (RSA, ECC, Diffie-Hellman) using Shor's Algorithm.
   - NIST (National Institute of Standards and Technology) standardized new **Post-Quantum Cryptography (PQC)** algorithms in August 2024:
     - **ML-KEM** (FIPS 203, formerly **CRYSTALS-Kyber**) for key exchange.
     - **ML-DSA** (FIPS 204, formerly Dilithium) for digital signatures.
2. **Mathematical Security vs. Physical Security:**
   - Mathematically, ML-KEM is unbreakable by quantum computers. Its hardness is based on lattice math called **M-LWE** (Module Learning With Errors).
   - **However**, when ML-KEM runs on physical silicon (like an ARM Cortex-M4 chip in a smart card, car key, medical device, or IoT sensor), electricity flows through transistors.
   - That physical execution causes **Side-Channel Leakage**:
     - Tiny fluctuations in **power consumption** (measured with an oscilloscope probe).
     - Subtle **electromagnetic (EM) radiation** leaked into the air.
3. **The Target Inside ML-KEM:**
   - To multiply huge polynomials quickly, ML-KEM uses the **Number Theoretic Transform (NTT)** (like an FFT for finite fields).
   - The NTT runs loops of arithmetic operations called **butterfly operations**.
   - As secret polynomial coefficients pass through the CPU registers and data buses, the power draw changes according to the **Hamming Weight** (number of `1` bits).
   - Even when developers apply **Boolean Masking** (splitting the secret into random shares $x = x_1 \oplus x_2$ so no single share reveals the secret), non-linear conversions (Arithmetic-to-Boolean, A2B) create subtle higher-order leakage.

---

### 2. The Big Bottleneck: Why existing tools choke on PQC

In classic AES attacks (e.g., attacking the AES S-Box):
- A single trace is only **1,000 to 3,000 samples long**.
- Datasets easily fit in 1–2 GB of laptop RAM.

In Post-Quantum ML-KEM:
- A single decapsulation trace is **20,000 to 45,000+ samples long** because of the long NTT loop operations!
- If you have 100,000 traces, each with 41,800 sample points as 32-bit floats:
  $$\text{Memory} = 100,000 \times 41,800 \times 4\text{ bytes} \approx \mathbf{16.72\text{ GB of RAM!}}$$
- Monolithic Python scripts (NumPy/SciPy/PyTorch) attempt to load this all into RAM at once, causing **Out-Of-Memory (OOM) crashes** and severe freezing.
- Furthermore, feeding 40,000 points into a standard neural network causes **parameter explosion** in the dense layers, leading to massive overfitting on noisy traces.

---

### 3. What We Built: The 3 Pillars of NeuralSCA / MojoPQC-SCA

```text
+-------------------------------------------------------------------------------+
|                             MojoPQC-SCA Architecture                         |
+-------------------------------------------------------------------------------+
|  1. STREAMING PIPELINE (Bounded-Memory System)                                |
|     Raw HDF5 (100k traces) --> Memory Chunking (B=256) --> Peak RAM < 112 MB  |
|     --> Digital FIR Filter --> Cross-Correlation Alignment --> POI Decimation |
|     (41,800 samples reduced to 5,000 salient NTT features)                    |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|  2. QUANTUM-INSPIRED NEURAL PROFILER                                          |
|     Input (5,000 samples) --> 1D-CNN Feature Extractor (16 & 32 filters)      |
|     --> Matrix Product State (MPS) Tensor Network Head (Bond chi=8)           |
|     Total Parameters: 9,226 (35.9% smaller than 14,400 CNN baseline)          |
|     Utilizes only 4.6% of the 200,000 parameter budget ceiling                |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|  3. EVALUATION, EDGE QUANTIZATION & AUDIT                                     |
|     --> Guessing Entropy (GE) calculation: converges to Rank 1.0 (key recovered)
|     --> ONNX Int8 Dynamic Quantization: 18.6 KB model size                    |
|     --> CPU Inference Latency: 0.14 ms/trace (7,142 traces/sec throughput)    |
|     --> SHA-256 Provenance Manifest: End-to-end reproducible audit trail      |
+-------------------------------------------------------------------------------+
```

Let's understand each of the 3 pillars clearly:

#### Pillar 1: The 6-Stage Streaming Preprocessing Pipeline
1. **Streaming Ingestion:** Instead of loading 16.7 GB into RAM, we load traces in small chunks of $B=256$ traces directly from HDF5 containers. RAM usage stays strictly under **112 MB** forever, even for 1,000,000 traces!
2. **Digital Zero-Phase FIR Low-Pass Filtering:** Removes high-frequency electrical switching noise from the oscilloscope without shifting the phase or distorting peaks.
3. **Cross-Correlation Phase Alignment:** Embedded microcontrollers have clock jitter and instruction variations. We compute cross-correlation against a reference butterfly window ($\Delta \tau = \pm 100$ samples) and circularly slide traces to align them perfectly.
4. **Point-of-Interest (POI) Decimation:** Compresses 41,800 raw points down to **5,000 informative POI samples**, discarding empty idle cycles and focusing on the leaky NTT operations.
5. **Z-Score Normalization:** Scales each trace to zero mean and unit variance ($\frac{t - \mu}{\sigma}$).
6. **Dataset Contract Validation:** Asserts no `NaN` or `Inf` values exist and class labels reside in $[0, 255]$.

#### Pillar 2: The Quantum-Inspired Neural Profiler (CNN + MPS)
- **Why a 1D-CNN?**  
  Convolutional filters scan across the time domain. They are shift-invariant, so even if traces have small remaining clock jitters, the convolutional kernels catch the leakage peaks.
- **Why Matrix Product States (MPS)?**  
  In a standard neural network, the classification head is a dense layer with thousands of weights ($1600 \times 256 = 409,600$ weights if uncompressed!).  
  In physics, **Matrix Product States (Tensor Networks)** are used to represent complex quantum systems with many interacting particles.  
  In machine learning, MPS factorizes a massive weight tensor into a small chain of low-rank 3-way tensor cores with a **bond dimension** $\chi = 8$.
  - CNN Baseline: **14,400 parameters**
  - Our CNN + MPS: **9,226 parameters** (**35.93% parameter reduction**, or $1.56\times$ more compact!).
  - Training speed: **$1.69\times$ faster** per epoch ($0.1065\text{ s} \to 0.0630\text{ s}$).
  - Strictly respects the project's **200,000 parameter budget** (consuming only 4.6% of it!).
  - Because it has fewer weights, it **cannot memorize noise**, which is crucial for side-channel analysis where Signal-to-Noise Ratio (SNR) is low.

#### Pillar 3: Evaluation, Edge Deployment & Reproducibility
- **What is Guessing Entropy (GE)?**  
  It is the standard academic metric (Standaert et al., EUROCRYPT 2009). For each attack trace, the neural network outputs probability scores for all 256 possible key byte candidates. We sum the log-probabilities over $m$ attack traces:
  $$\mathbf{S}_m(k) = \sum_{j=1}^m \log P(Y = k \mid \mathbf{t}_j)$$
  We sort all 256 candidates by score. The **Rank** of the true secret key is its position in this sorted list.
  - At the start ($m=1$): Rank is random ($\approx 128$).
  - As more attack traces are accumulated ($m=50, 100, 150$): True key score pulls ahead.
  - At convergence ($m \ge 150$): Rank reaches **1.0** (meaning the true key candidate is ranked #1 with 100% confidence).
- **Int8 Quantization for Edge Deployment:**
  - We export the PyTorch model to **ONNX** and quantize 32-bit floats to **8-bit integers (Int8)**.
  - Model file size drops from **58.4 KB $\to$ 18.6 KB**.
  - Inference latency drops to **0.14 ms per trace** on a regular single-threaded CPU!
  - That equals **7,142 traces per second**, proving this attack profiler can run in real-time on commodity laptops or Raspberry Pi without an expensive GPU.
- **Cryptographic Provenance Manifest (`run_manifest.json`):**
  - Hashes raw data, preprocessed files, trained models, and ONNX files using **SHA-256**, ensuring 100% scientific reproducibility.

---

## PART 2: How to Explain the Project (Your Presentation Script)

Use this section whenever someone asks you: *"What is your project about?", "Walk me through what you did",* or during your final paper review.

### Option A: The 60-Second Elevator Pitch

> *"My project is called **NeuralSCA (or MojoPQC-SCA)**. We developed an accelerated deep-learning side-channel analysis pipeline designed to test the physical security of the new Post-Quantum Cryptography standard, **ML-KEM (NIST FIPS 203)**.*  
>  
> *Even though ML-KEM is mathematically secure against quantum computers, physical microcontrollers leak secret information through power consumption and electromagnetic waves during the Number Theoretic Transform (NTT).*  
>  
> *The big problem is that PQC traces are enormous—over 40,000 sample points—which crashes standard Python tools with out-of-memory errors on large datasets. To solve this, we did three things:*  
> *1. Built a 6-stage streaming ingestion pipeline that bounds RAM usage to under 112 MB, achieving an 8.9x speedup.*  
> *2. Replaced heavy dense classification layers with a quantum-inspired Matrix Product State (MPS) tensor network, cutting model parameters by 36% down to just 9,226 parameters while accelerating training by 1.7x.*  
> *3. Dynamically quantized the model to 8-bit integers via ONNX Runtime, enabling real-time CPU evaluation at 0.14 milliseconds per trace—over 7,100 traces per second—without needing a GPU."*

---

### Option B: The Detailed 5-Minute Technical Walkthrough

#### Slide / Point 1: Motivation & The Quantum Shift
- "With NIST standardizing ML-KEM in FIPS 203, the world is migrating to lattice-based cryptography. But mathematical security does not imply physical security."
- "When running on IoT chips and smart cards (like ARM Cortex-M4), operations like polynomial butterfly loops in the NTT unintentionally leak secret key bytes through power dissipation."

#### Slide / Point 2: The Two Core Bottlenecks in Prior Work
- **Bottleneck 1 (Memory & I/O):** "Existing side-channel tools written in Python/NumPy try to load the entire dataset into memory. In PQC, 100,000 traces of 41,800 samples equals 16.7 GB of floats, which instantly crashes with Out-Of-Memory (OOM) errors."
- **Bottleneck 2 (Neural Overfitting):** "High-dimensional traces cause standard neural networks to explode in parameter count, causing the model to memorize high-frequency noise rather than true leakage."

#### Slide / Point 3: Our Solution — The Streaming Preprocessing Pipeline
- "We designed a 6-stage streaming pipeline:
  1. Chunked HDF5 reading (batch size 256) keeping peak memory below 112 MB.
  2. A zero-phase digital FIR low-pass filter (order 31) that removes high-frequency noise without phase delay.
  3. Template cross-correlation alignment ($\pm 100$ samples) to neutralize clock jitter.
  4. Point-of-Interest (POI) decimation to 5,000 salient samples covering the NTT loops.
  5. Per-trace Z-score normalization.
  6. Strict contract and sanity validation."

#### Slide / Point 4: Quantum-Inspired Tensor Network Architecture (CNN + MPS)
- "For the profiling model, we combined a 1D-CNN feature extractor with a quantum-inspired Matrix Product State (MPS) classifier head with bond dimension $\chi = 8$."
- "Rather than connecting all features with a massive weight matrix, MPS factorizes the weight tensor into a chain of contracted 3-way cores."
- "Results:
  - CNN baseline: 14,400 parameters.
  - CNN + MPS: **9,226 parameters** (a **35.93% reduction**).
  - Strictly complies with our **200,000-parameter budget ceiling** by utilizing only 4.6% of it.
  - Epoch training time sped up from 0.106s to 0.063s ($1.69\times$ faster)."

#### Slide / Point 5: Key Recovery & Guessing Entropy ($GE$)
- "We evaluated the attack using Guessing Entropy (the expected rank of the true key).
- In both unprotected and first-order masked ML-KEM-512 traces, the Guessing Entropy decays monotonically as attack traces accumulate, reaching **Rank 1.0** (complete secret key recovery within 150 traces)."

#### Slide / Point 6: Quantized Edge Deployment & Reproducibility
- "To prove edge practicality, we exported the model to ONNX and applied dynamic symmetric Int8 quantization.
- The model shrinks to **18.6 KB**.
- On a single-threaded commodity CPU, inference latency is **0.14 ms per trace**, achieving **7,142 traces/second**.
- Finally, we generate an automated `run_manifest.json` recording SHA-256 hashes of all datasets and model weights for 100% scientific reproducibility."

---

## PART 3: Quick Numbers Reference Table (Memorize These Key Stats!)

| Metric / Parameter | Value to Mention | Why it Matters |
|---|---|---|
| **Target Algorithm** | ML-KEM-512 (Kyber) & NTT | NIST FIPS 203 primary post-quantum standard |
| **Raw Trace Length** | 41,800 sample points | Captures prolonged NTT butterfly loops |
| **Decimated POI Length** | 5,000 sample points | Extracts the high-SNR leaky butterfly window |
| **Monolithic RAM Usage** | $>16.5\text{ GB}$ (OOM Crash) | Why existing Python tools fail |
| **Our Peak RAM Usage** | $\mathbf{\le 111.8\text{ MB}}$ ($>99.3\%$ reduction) | Runs on any consumer laptop |
| **Preprocessing Throughput** | $\mathbf{761.2\text{ traces/s}}$ | $8.9\times$ faster than standard chunked SciPy |
| **1D-CNN Baseline Params** | 14,400 parameters | Standard dense classifier head |
| **CNN + MPS (Ours) Params** | $\mathbf{9,226\text{ parameters}}$ | **35.93% parameter reduction** ($1.56\times$ more compact) |
| **Parameter Budget Ceiling** | $<200,000$ parameters | We use only **4.61%** of budget |
| **Training Epoch Time** | $\mathbf{0.063\text{ s}}$ vs $0.106\text{ s}$ | $1.69\times$ faster training |
| **MPS Bond Dimension** | $\chi = 8$ | Optimal virtual entanglement capacity |
| **Guessing Entropy ($GE$)** | Converges to **Rank 1.0** | Secret key uniquely identified |
| **Int8 Model Size** | $\mathbf{18.6\text{ KB}}$ (down from 58.4 KB) | 68.2% storage reduction |
| **CPU Inference Latency** | $\mathbf{0.14\text{ ms / trace}}$ | Real-time edge evaluation on laptop CPU |
| **Inference Throughput** | $\mathbf{7,142\text{ traces/second}}$ | No GPU required |
| **Provenance Method** | Cryptographic SHA-256 digests | Fully auditable and reproducible |

---

## PART 4: Tough Examiner / Interview Questions & How to Answer Them

#### Q1: "Why use a Matrix Product State (MPS) instead of just adding Dropout or L2 regularization?"
> **Answer:** *"Dropout and L2 regularization suppress overfitting, but they do not reduce the parameter footprint of dense layers, nor do they reduce memory and compute latency. MPS fundamentally decomposes the weight tensor into a low-rank core chain with a small bond dimension ($\chi=8$). This cuts parameters from 14,400 to 9,226, accelerates training by 1.69x, and explicitly bounds the virtual entanglement between features, preventing the network from fitting high-order noise while preserving multi-share leakage correlations."*

#### Q2: "Why is Guessing Entropy (GE) preferred over simple validation accuracy?"
> **Answer:** *"In side-channel analysis, individual traces are extremely noisy (very low SNR). A model might have only 10% to 20% single-trace top-1 accuracy, yet across 100 attack traces, the true secret key's accumulated log-likelihood probability will consistently rank above all other 255 candidates. Guessing Entropy measures the average rank of the correct key across attack traces. When GE reaches 1.0, the key is 100% recovered, even if single-trace accuracy is low."*

#### Q3: "What is the difference between your synthetic data and physical ARM Cortex-M4 traces?"
> **Answer:** *"In our project, we support both. We created deterministic synthetic NTT leakage benchmarks with Hamming-weight models and noise for rapid CI testing and smoke verification. For empirical validation, our adapter ingests real-world Zenodo power datasets recorded from physical 32-bit ARM Cortex-M4 microcontrollers running both unprotected and first-order masked ML-KEM-512 decapsulation campaigns."*

#### Q4: "Why did you use Int8 quantization instead of floating point?"
> **Answer:** *"Side-channel security auditing is often performed in the field on edge devices, portable oscilloscopes, or embedded testbeds without discrete GPUs. Int8 dynamic quantization compresses our model to 18.6 KB and allows integer SIMD instructions on modern CPUs, yielding an inference latency of 0.14 ms per trace (7,142 traces/sec) with zero loss in key-recovery rank."*

#### Q5: "How do you guarantee reproducibility?"
> **Answer:** *"Every run executes strict dataset contract checks (asserting tensor dimensions, finite values, and label ranges) and generates an automated `run_manifest.json`. This manifest cryptographically hashes raw data, preprocessed HDF5 containers, model checkpoints, and ONNX files using SHA-256 alongside the exact Git commit and environment configuration."*
