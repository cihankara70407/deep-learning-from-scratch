# Deep Learning from Scratch: Two-Hidden-Layer Multilayer Perceptron

**Engineering Laboratory Report:** Lab 2 — Chapter 3 DL Primer  
**Student:** Cihan Karayigit  
**Student ID:** 70407  
**GitHub Organization/User:** `cihankara70407`  
**Repository:** `cihankara70407/deep-learning-from-scratch`  

---

## 1. Project Overview & Architectural Design

This project provides a vectorized, functional implementation of a two-hidden-layer feedforward artificial neural network using purely **NumPy**. The network is designed to perform classification on handwritten numerical digits from the Scikit-Learn `load_digits` dataset ($8 \times 8$ grayscale images mapped to 10 classes).

### Architectural Topology

```
Input Features [X ∈ ℝ^(N × 64)]
               │
               ▼
   [Affine Transformation 1]  ──>  Z1 = X · W1 + b1
               │
               ▼
     [ReLU Subgradient Unit]  ──>  A1 = max(0, Z1)
               │
               ▼
   [Affine Transformation 2]  ──>  Z2 = A1 · W2 + b2
               │
               ▼
     [ReLU Subgradient Unit]  ──>  A2 = max(0, Z2)
               │
               ▼
   [Affine Transformation 3]  ──>  Z3 = A2 · W3 + b3
               │
               ▼
 [Stable Softmax Normalizer]  ──>  P_ik = exp(Z3_ik - max_j Z3_ij) / Σ exp(...)
               │
               ▼
[Categorical Cross-Entropy]  ──>  ℒ = - (1/N) Σ Σ Y_ik · ln(P_ik)
```

### Parameter Tensor Specifications
- **Weights Initialization:** Sampled from normal distribution $\mathcal{N}(\mu=0, \sigma^2=0.01^2)$
  - $W_1 \in \mathbb{R}^{64 \times W}$
  - $W_2 \in \mathbb{R}^{W \times W}$
  - $W_3 \in \mathbb{R}^{W \times 10}$
- **Bias Vectors:** Zero-initialized
  - $b_1 \in \mathbb{R}^W$
  - $b_2 \in \mathbb{R}^W$
  - $b_3 \in \mathbb{R}^{10}$
- **Evaluated Hidden Widths:** $W \in \{16, 64\}$

---

## 2. Experimental Verification & Graded Evidence

### Check A: Untrained Loss Sanity Validation
For an untrained model with weights near zero, the network outputs an approximately uniform distribution over the 10 classes ($P_{ik} \approx 0.1$). The expected initial cross-entropy loss is:
$$\mathcal{L}_0 \approx -\ln(0.1) = \ln(10) \approx 2.302585$$

- **Empirically Observed Loss:** `2.302557`
- **Deviation from Theory:** `0.000028` ($\Delta < 0.001$) $\implies$ **PASSED**

---

### Check B: Numerical vs. Analytical Gradient Checking (Finite Differences)
To ensure the analytical backpropagation implementation is exact, central difference perturbations were applied to 10 randomly selected weights:
$$g_{\text{num}} = \frac{\mathcal{L}(w + \epsilon) - \mathcal{L}(w - \epsilon)}{2\epsilon}, \quad \epsilon = 10^{-5}$$

The relative error metric used is:
$$\text{Relative Error} = \frac{|g_{\text{analytic}} - g_{\text{num}}|}{\max(|g_{\text{analytic}}|, |g_{\text{num}}|, 10^{-12})}$$

Specification: At least 9 out of 10 sampled parameters must exhibit relative error $< 10^{-6}$.

#### Detailed Gradient Verification Output

| Test # | Target Parameter | Analytical Gradient | Numerical Gradient | Relative Error | Tolerance Check |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | $W_1[3, 4]$ | `-6.9121128e-05` | `-6.9121131e-05` | `4.41e-08` | **PASS (< 1e-6)** |
| 02 | $W_2[1, 9]$ | `-3.1959683e-05` | `-3.1959679e-05` | `1.27e-07` | **PASS (< 1e-6)** |
| 03 | $W_3[5, 8]$ | `+3.2157666e-05` | `+3.2157654e-05` | `3.78e-07` | **PASS (< 1e-6)** |
| 04 | $W_1[19, 8]$ | `-4.6650917e-05` | `-4.6650950e-05` | `7.12e-07` | **PASS (< 1e-6)** |
| 05 | $W_2[15, 0]$ | `-3.0510840e-05` | `-3.0510838e-05` | `6.02e-08` | **PASS (< 1e-6)** |
| 06 | $W_3[13, 5]$ | `+3.5399586e-05` | `+3.5399572e-05` | `3.92e-07` | **PASS (< 1e-6)** |
| 07 | $W_1[60, 1]$ | `-5.6165270e-05` | `-5.6165295e-05` | `4.46e-07` | **PASS (< 1e-6)** |
| 08 | $W_2[3, 10]$ | `+0.0000000e+00` | `+0.0000000e+00` | `0.00e+00` | **PASS (< 1e-6)** |
| 09 | $W_3[11, 9]$ | `+6.8870935e-05` | `+6.8870953e-05` | `2.72e-07` | **PASS (< 1e-6)** |
| 10 | $W_1[61, 15]$| `-2.2617805e-05` | `-2.2617819e-05` | `5.90e-07` | **PASS (< 1e-6)** |

**Outcome:** **10 / 10 parameters passed** with relative error below $10^{-6}$. Derivations via differential forms and traces are cataloged in [MATH_NOTES.md](MATH_NOTES.md).

---

### Check C: Empirical Evaluation Across 6 Hyperparameter Configurations

All combinations of learning rates $\eta \in \{0.001, 0.01, 0.1\}$ and hidden widths $W \in \{16, 64\}$ were trained for 50 epochs with mini-batch size 32.

#### Final Experimental Metrics

| Configuration ID | Hidden Dim ($W$) | Step Size ($\eta$) | Final Train Loss | Final Test Loss | Test Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Width_16__LR_0.001` | 16 | 0.001 | 2.3025 | 2.3025 | 10.19% |
| `Width_16__LR_0.01`  | 16 | 0.010 | 2.3025 | 2.3025 | 10.19% |
| `Width_16__LR_0.1`   | 16 | 0.100 | 1.4324 | 1.4743 | 47.22% |
| `Width_64__LR_0.001` | 64 | 0.001 | 2.3025 | 2.3025 | 10.19% |
| `Width_64__LR_0.01`  | 64 | 0.010 | 2.3023 | 2.3023 | 10.19% |
| **`Width_64__LR_0.1`**  | **64** | **0.100** | **0.4266** | **0.5029** | **86.11%** |

#### Trajectory Plots

![Loss Curves](results/loss_curves.png)

---

### Check D: Forward vs. Backward Latency Benchmark

Measured over 100 consecutive runs using `time.perf_counter()` on mini-batch size 32:
- **Median Forward Pass Latency:** `0.0711 ms`
- **Median Backward Pass Latency:** `0.0423 ms`

#### Algorithmic Breakdown: Why Backward Pass Dominates in Deep Networks
In general neural network architectures, backpropagation requires calculating two distinct categories of derivatives:
1. **Derivatives with respect to parameters ($\nabla_W \mathcal{L}$):** Required to update the weight matrices.
2. **Derivatives with respect to inputs/activations ($\nabla_X \mathcal{L}$):** Required to pass error signals backward through the chain rule to preceding layers.

As detailed in [MATH_NOTES.md](MATH_NOTES.md), this causes backpropagation to require **5 dense matrix multiplications (GEMM operations)**, compared to only **3 GEMM operations** in the forward pass.

---

## 3. Repository Organization

```
dl_scratch_70407_cihan/
├── MATH_NOTES.md              # Rigorous matrix calculus derivations
├── README.md                  # Comprehensive technical laboratory report
├── main.py                    # Complete evaluation pipeline and visualization
├── engine/
│   ├── __init__.py            # Export interface
│   ├── computation_graph.py   # Vectorized graph, forward, loss, backward, updates
│   ├── gradient_verifier.py   # Finite difference slope checker
│   └── latency_profiler.py    # High-resolution phase benchmark
└── results/
    ├── loss_curves.png        # Dual-panel loss curve figures
    └── metrics_report.json    # Machine-readable output summary
```

---

## 4. Replication Steps

```bash
# Clone the repository
git clone https://github.com/cihankara70407/deep-learning-from-scratch.git
cd deep-learning-from-scratch

# Setup dependencies
pip install numpy scikit-learn matplotlib

# Execute pipeline
python main.py
```
