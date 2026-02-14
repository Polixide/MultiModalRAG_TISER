# Grounding Temporal Reasoning in Retrieval-Augmented Generation through Multi-Granular Timeline Representations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-orange)](https://huggingface.co/datasets/Dancat/MultiModal_TISER_train-dataset)
[![Model: Qwen3-VL](https://img.shields.io/badge/Model-Qwen3--VL-blueviolet)](https://huggingface.co/Qwen)

---

## 📌 Introduction

This project introduces a **multimodal preprocessing and reasoning framework** designed to enhance temporal reasoning in Retrieval-Augmented Generation (RAG) systems. Large Language Models (LLMs) often struggle with complex chronological dependencies and long-horizon narratives.

To address this, we developed a pipeline that:

1. **Parses** natural language temporal descriptions into structured representations.
2. **Visualizes** these structures as multi-granular timeline charts (**Gantt**, **Scatter**, **Line**).
3. **Retrieves** relevant charts using a late-interaction multimodal retriever (**ColQwen2**).
4. **Reasons** over these visual artifacts using a fine-tuned Vision-Language Model (**Qwen3-VL-8B**).

The system significantly improves performance on complex temporal reasoning benchmarks by explicitly grounding the reasoning process in visual chronological data.

---

## 🚀 Key Features

* **Multimodal Temporal Pipeline:** Automated extraction of temporal entities and dynamic generation of visual charts to represent duration, interval sequences, and narrative flows.
* **Late-Interaction Retrieval:** Implements a **ColPali-inspired** retriever (ColQwen2) that treats timeline visualizations as micro-documents for fine-grained cross-modal matching.
* **Visual-CoT Reasoning:** A structured Chain-of-Thought protocol that enforces sequential visual reasoning, interval extraction, and self-reflection.
* **Fine-Tuned VLM:** A Qwen3-VL-8B-Instruct model fine-tuned via **QLoRA** to interpret structured temporal charts.

---

## 🏗️ Architecture

The framework consists of three main stages:

### 1. Contextual Temporal Extraction & Visualization

The preprocessing pipeline transforms textual prompts into aligned question-image-answer triples.

* **Extraction:** Rule-based patterns isolate temporal segments and normalize dates to a year-level scale (preserving month precision where available).
* **Visualization:** Contexts are rendered into:

  * **Gantt Charts** (duration-centric),
  * **Scatter Plots** (event intervals),
  * **Line Charts** (temporal sequences).

Visual styles are randomized to prevent overfitting.

---

### 2. Multimodal Retrieval (RAG)

We separate retrieval from generation to ensure robust context grounding.

* **Indexer:** Timeline charts are indexed using a ColQwen2-based retriever.
* **Search:** Text-to-Image retrieval locates the chart corresponding to the temporal context of the user's query.

---

### 3. Vision-Language Inference

The top-ranked timeline chart is provided to the fine-tuned VLM. The model follows a strict reasoning format:

```xml
<reasoning>
...
</reasoning>
<timeline>
...
</timeline>
<reflection>
...
</reflection>
<answer>
...
</answer>
```

---

## 📂 Project Structure

The repository is organized as follows:

```text
MultiModalRAG_TISER/
├── __init__.py
├── config.py                     # Central configuration file (paths, model parameters, training hyperparameters)

├── data/
│   ├── charts_generator.py       # Timeline visualization and chart generation (Gantt, Scatter, Line)
│   └── dataset_tiser.py          # Dataset preprocessing and multimodal sample construction

├── modeling/
│   ├── collator_qwen.py          # Custom data collator for Qwen2-VL training
│   ├── lora_qwen.py              # LoRA and QLoRA model setup and parameter-efficient tuning
│   └── sft_trainer.py            # Supervised fine-tuning (SFT) training loop

├── rag/
│   ├── index_byaldi.py           # Multimodal indexing pipeline (ColQwen2 / ColPali-inspired)
│   ├── recall_metrics.py         # Retrieval evaluation and Recall@k computation
│   └── rag_vlm_eval.py           # End-to-end RAG + VLM inference and evaluation

├── eval/
│   └── text_metrics.py           # Text-based evaluation metrics (EM, F1, etc.)

scripts/
├── generate_charts_and_json.py   # Full preprocessing pipeline (text → structured timelines → charts)
├── train_qwen_sft.py             # QLoRA fine-tuning script for Qwen2-VL
├── build_rag_index.py            # Build multimodal index for retrieval
├── compute_rag_recall.py         # Evaluate retrieval performance
└── eval_rag_vlm.py               # End-to-end RAG + VLM evaluation

README.md
```

---

## ⚙️ Configuration

Project parameters are managed in `config.py`. The key settings include:

### Dataset Paths
- `TISER_TRAIN_JSON`, `TISER_TEST_JSON` — textual datasets
- `MM_TISER_TRAIN_JSON`, `MM_TISER_TEST_JSON` — multimodal datasets
- `IMAGES_DIR` — directory for timeline charts

### Model
- Base: `Qwen/Qwen3-VL-8B-Instruct`
- Fine-tuned: `Dancat/MM_Tiser_Qwen3_VL_FT_v2`

### Fine-Tuning (SFT)
- Epochs: 1
- Batch size: 2
- Gradient accumulation: 4
- Learning rate: 1e-4
- Scheduler: cosine

### LoRA
- Rank: 16
- Alpha: 32
- Target modules: attention and MLP projections

### RAG
- Retriever: `vidore/colqwen2-v1.0`
- Index name: `tiser_charts_index`
- Top-K: [1, 3, 5]

> All scripts automatically use these settings from `config.py`.

---

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU (A100 80GB recommended for full fine-tuning)

### Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Polixide/MultiModalRAG_TISER.git
cd MultiModalRAG_TISER

# It is recommended to use a virtual environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

> **Note:** Key dependencies include `transformers`, `peft`, `bitsandbytes`, `torch`, `matplotlib` (for chart generation), and `colpali-engine`.

---

## 🧪 Usage

Once `config.py` is configured, you can run the pipeline stages directly.

### 1. Data Generation (Pipeline)

Generate visual timelines from textual data and create the multimodal dataset:

```bash
python scripts/generate_charts_and_json.py
```

---

### 2. Fine-tuning Qwen2-VL

Launch the fine-tuning process using QLoRA:

```bash
python scripts/train_qwen_sft.py
```

---

### 3. Retrieval & Inference

Run the full RAG pipeline:

```bash
python scripts/build_rag_index.py
python scripts/compute_rag_recall.py
python scripts/eval_rag_vlm.py
```

---

## 📊 Results

### Retrieval Performance (Recall@k)

Evaluated on 300 held-out test instances.

| Chart Type  | Recall@1  | Recall@3  | Recall@5  |
| ----------- | --------- | --------- | --------- |
| **Gantt**   | 54.9%     | 59.8%     | 71.9%
| **Scatter** | 66.4%     | 68.6%     | 72.1% 
| **Line**    | 67.9%     | 82.1%     | 83.3%
| **Global**  | **63.7%** | **69.7%** | **75.0%**

---

### End-to-End Reasoning (Exact Match & F1)

Comparison between Baseline (Qwen3-VL) and Fine-Tuned Model.

| Dataset             | EM (Base) | F1 (Base) | EM (FT)   | F1 (FT)   |
| ------------------- | --------- |-----------|---------- |---------- |
| **TimeQA (Hard)**   | 53.33     | 0.571     | 60.00     | 0.623     |
| **TimeQA (Easy)**   | 76.67     | 0.771     | 76.67     | 0.772     |
| **TempReason (L3)** | 45.00     | 0.521     | 50.00     | 0.558     |
| **TempReason (L2)** | 41.67     | 0.497     | 43.33     | 0.514     |
| **TGQA**            | 41.67     | 0.602     | 46.67     | 0.758     |
| **Macro Avg.**      | **51.67** | **0.592** | **55.33** | **0.645** | 

---

## 📂 Datasets

The project utilizes a custom multimodal dataset derived from **TISER**, encompassing TimeQA, TempReason, and TGQA.

| Dataset Split     | HuggingFace Link                                                          |
| ----------------- | ------------------------------------------------------------------------- |
| **Training Data** | https://huggingface.co/datasets/Dancat/MultiModal_TISER_train-dataset     |
| **Test Data**     | https://huggingface.co/datasets/Dancat/MultiModal_TISER_test_only-dataset |

---

## 👥 Authors

## Authors

- Daniele Catalano (@Polixide) - Politecnico di Torino , Data Science & Engineering
- Francesco Dal Cero (@Dalceeee) - Politecnico di Torino , Data Science & Engineering
- Ramadan Mehmetaj (@Danki02) - Politecnico di Torino , Data Science & Engineering
- Samuele Caruso (@Knightmare2002) - Politecnico di Torino , Data Science & Engineering

*Politecnico di Torino — DNLP Project 2025-2026*
