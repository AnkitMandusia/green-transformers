# A Four-Stage Framework for Sustainable and Efficient Large Language Models

This repository contains the official source code and experimental pipelines for the research paper, **"From Silicon to System: A Four-Stage Framework for Sustainable and Efficient Large Language Models"**.

## 1. Abstract

Large Language Models (LLMs) like BERT and DistilBERT offer remarkable performance but incur significant energy and carbon costs, hindering their deployment on resource-constrained or sustainable systems. This paper proposes and evaluates a holistic four-stage framework designed to enhance the sustainability of LLM inference. We systematically analyze four independent optimization strategies: (1) a **Hybrid Neuromorphic** architecture integrating Spiking Neural Networks (SNNs); (2) a **Sparse Neuromorphic** model combining structured sparsity with efficient attention mechanisms; (3) an **Adaptive Quantization** scheduler that dynamically adjusts numerical precision; and (4) a **Carbon-Aware Scheduler** that optimizes inference routing across edge data centers to minimize carbon intensity. Our findings demonstrate a viable path toward neuromorphic-compatible, sustainable LLM inference by showing significant gains in energy efficiency with competitive accuracy on standard NLP benchmarks.

---

## 2. The Four Stages of Optimization

This research is divided into four distinct stages, each implemented in its own Jupyter Notebook. Each stage represents an independent optimization strategy that can be evaluated in isolation.

### Stage 1: Hybrid Neuromorphic Model (`hybrid_neuromorphic.ipynb`)

* **Objective:** To reduce the energy consumption of the core self-attention mechanism by replacing it with a bio-inspired Spiking Self-Attention (SSA) module.
* **Methodology:** This stage implements a Hybrid Neuromorphic Transformer (HNT) where dense self-attention layers are replaced with our custom `SpikingSelfAttention` module. The implementation involves tensor-to-spike conversion using rate encoding and the use of surrogate gradients for stable fine-tuning.

### Stage 2: Sparse Neuromorphic Model (`sparse_neuromorphic.ipynb`)

* **Objective:** To reduce computational density by replacing the dense $O(N^2)$ attention operation with a highly efficient sparse equivalent using Top-K sparsification.
* **Methodology:** This stage implements a `NeuromorphicFusedAttention` module that calculates the full attention score matrix but retains only the Top-K most significant scores for each token. This drastically reduces the number of floating-point operations required for each forward pass.

### Stage 3: Adaptive Quantization (`adaptive_quantization.ipynb`)

* **Objective:** To introduce a software-level optimization that dynamically adjusts numerical precision (INT4/INT8/FP32) based on the difficulty of the input query.
* **Methodology:** This framework uses a `ComplexityEstimator` to calculate a "complexity score" for each input based on metrics like sequence length and token entropy. An `AdaptiveScheduler` then maps this score to an optimal precision level, balancing performance and energy on a query-by-query basis.

### Stage 4: Carbon-Aware Scheduling (`carbon_aware_scheduling.ipynb`)

* **Objective:** To minimize the real-world carbon footprint of the entire inference system by making intelligent, geography-aware deployment decisions.
* **Methodology:** This system-level optimization uses a `CarbonAwareScheduler` that integrates with real-time grid data APIs (e.g., Electricity Maps). It uses a Temporal Convolutional Network (TCN) to forecast carbon intensity and solves a multi-objective optimization problem to route inference requests to the data center with the best balance of low latency and minimal carbon emissions.

---

## 3. Setup and Installation

To reproduce the experiments, please follow these steps.

**1. Clone the repository:**
```bash
git clone [https://github.com/AnkitMandusia/green-transformers.git](https://github.com/AnkitMandusia/green-transformers.git)
cd green-transformers
```
### 2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
### 3. Install dependencies:

A requirements.txt file is provided. Install all necessary packages using pip:
```bash
pip install -r requirements.txt
```
### 4. How to Run the Experiments
Each of the four research stages is contained in its own Jupyter Notebook (.ipynb) file. Each notebook is a self-contained experiment.

#### 1. Launch Jupyter Notebook or JupyterLab:
```bash
jupyter notebook
```
#### 2. Open and Run a Notebook:
Navigate to the desired notebook file (e.g., hybrid_neuromorphic.ipynb) in the Jupyter interface. You can run the cells sequentially by clicking "Run" or using the shortcut Shift + Enter.

## 4. License
This project is licensed under the MIT License. See the LICENSE file for details.
