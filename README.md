# Amharic-IR-Benchmarks

[![ACL 2025 Findings](https://img.shields.io/badge/Paper-ACL%202025%20Findings-b31b1b)](https://aclanthology.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![Models on HF](https://img.shields.io/badge/Models-HuggingFace-blueviolet)](https://huggingface.co/collections/rasyosef/amharic-text-embedding-models-679cb55eae1d498e3ac5bdc5)
[![Dataset on HF](https://img.shields.io/badge/Dataset-HuggingFace-ff69b4)](https://huggingface.co/datasets/rasyosef/amharic-passage-retrieval-dataset)

This repository accompanies our ACL 2025 Findings paper:
**"Optimized Text Embedding Models and Benchmarks for Amharic Passage Retrieval"**

We provide a reproducible benchmark suite for Amharic information retrieval, including:

* BM25 sparse baseline
* Dense embedding models (RoBERTa / BERT for Amharic)
* ColBERT-AM (late interaction retriever)

---

## 📁 Repository Structure

```
amharic-ir-benchmarks/
├── baselines/             # BM25, ColBERT, and dense Amharic retrievers
│   ├── bm25_retriever/
│   ├── ColBERT_AM/
│   └── embedding_models/
├── configs/               # Config files (training/evaluation)
├── data/                  # Scripts to download, preprocess, and prepare datasets
├── scripts/               # Shell scripts for training, indexing, evaluation
├── utils/                 # Utility functions
├── amharic_environment.yml  # Conda environment
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Conda (Recommended)

```bash
conda env create -f amharic_environment.yml
conda activate amharic_ir
```

Or using pip:

```bash
pip install -r requirements.txt
```

---

## 📚 Datasets

We use two publicly available Amharic datasets:

| Dataset          | Description                         | Link                                                                                                                |
| ---------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **2AIRTC**       | Ad-hoc IR test collection           | [IRIT Website](https://www.irit.fr/AmharicResources/airtc-the-amharic-adhoc-information-retrieval-test-collection/) |
| **Amharic News** | Headline–body classification corpus | [Hugging Face](https://huggingface.co/datasets/rasyosef/amharic-news-category-classification)                       |

Scripts for downloading and preprocessing can be found in the `data/` folder.

---

##  Usage

### Train ColBERT on Amharic News:

```bash
bash scripts/train_colbert.sh
```

### Evaluate BM25:

```bash
python baselines/bm25_retriever/run_bm25.ipynb
```

### Index and Retrieve (ColBERT):

```bash
bash scripts/index_colbert.sh
bash scripts/retrieve_colbert.sh
```

---

## 📊 Benchmark Results

| Model                     | MRR\@10 | Recall\@10 | Recall\@100 |
| ------------------------- | ------- | ---------- | ----------- |
| BM25                      | 0.657   | 0.774      | 0.871       |
| ColBERT-AM                | 0.754   | 0.858      | 0.931       |
| RoBERTa-Base-Amharic-Embd | 0.755   | 0.897      | 0.971       |
| RoBERTa-Medium-Amharic    | 0.707   | 0.861      | 0.963       |

Our Amharic-specific dense retrievers consistently outperform multilingual baselines from the MTEB leaderboard.

---

## 📄 Citation

If you use this repository, please cite our ACL 2025 Findings paper:

```bibtex
@inproceedings{mekonnen2025amharic,
  title={Optimized Text Embedding Models and Benchmarks for Amharic Passage Retrieval},
  author={Mekonnen, Kidist Amde and Yates, Andrew and de Rijke, Maarten},
  booktitle={Findings of ACL},
  year={2025}
}
```

---

## 📩 Contact

Please [open an issue](https://github.com/kidist-amde/amharic-ir-benchmarks/issues) for questions, feedback, or suggestions.

---

## 📜 License

This project is licensed under the [Apache 2.0 License](LICENSE).
