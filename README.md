# Amharic-IR-Benchmarks

[![ACL 2025 Findings](https://img.shields.io/badge/Paper-ACL%202025%20Findings-b31b1b)](https://aclanthology.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![Models on HF](https://img.shields.io/badge/Models-HuggingFace-blueviolet)](https://huggingface.co/collections/rasyosef/amharic-text-embedding-models-679cb55eae1d498e3ac5bdc5)
[![Dataset on HF](https://img.shields.io/badge/Dataset-HuggingFace-ff69b4)](https://huggingface.co/datasets/rasyosef/amharic-news-retrieval-dataset)

This repository accompanies our ACL 2025 Findings paper:
**"Optimized Text Embedding Models and Benchmarks for Amharic Passage Retrieval"**

✨  We provide a reproducible benchmark suite for Amharic information retrieval, including:

* BM25 sparse baseline
* Dense embedding models (RoBERTa / BERT variants fine-tuned for Amharic)

* ColBERT-AM (late interaction retriever)

---
## 👐 Features
-  **Pretrained Amharic Retrieval Models** Includes ( RoBERTa-Base-Amharic-Embd, RoBERTa-Medium-Amharic-Embd, BERT-Medium-Amharic-Embd, and ColBERT-AM for dense retrieval.)
-  **Hugging Face model & dataset links for easy access**
-  **Training, evaluation, and inference scripts for reproducibility**
-  **Benchmarks BM25 (sparse retrieval), bi-encoder dense retrieval, and ColBERT (late interaction retrieval) for Amharic.**
-  **MS MARCO-style dataset conversion script & direct dataset links**
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

## 📊 Amharic News: Dense Retrieval Benchmark

This table presents the performance of **Amharic-optimized** vs **multilingual** dense retrieval models on the **Amharic News dataset**, using a **bi-encoder** architecture. We report **MRR\@10**, **NDCG\@10**, and **Recall\@10/50/100**. Best scores are in **bold**, and † indicates statistically significant improvements *(p < 0.05)* over the strongest multilingual baseline.

## 📊 Benchmark Results

| Model                                 | Params | MRR@10    | NDCG@10   | Recall@10 | Recall@50 | Recall@100 |
|---------------------------------------|--------|-----------|-----------|-----------|-----------|------------|
| *Multilingual models*                 |        |           |           |           |           |            |
| gte-modernbert-base                   | 149M   | 0.019     | 0.023     | 0.033     | 0.051     | 0.067      |
| gte-multilingual-base                 | 305M   | 0.600     | 0.638     | 0.760     | 0.851     | 0.882      |
| multilingual-e5-large-instruct        | 560M   | 0.672     | 0.709     | 0.825     | 0.911     | 0.931      |
| snowflake-arctic-embed-l-v2.0         | 568M   | 0.659     | 0.701     | 0.831     | 0.922     | 0.942      |
| *Ours (Amharic-optimized models)*     |        |           |           |           |           |            |
| BERT-Medium-Amharic-embed             | 40M    | 0.682     | 0.720     | 0.843     | 0.931     | 0.954      |
| RoBERTa-Medium-Amharic-embed          | 42M    | 0.735     | 0.771     | 0.884     | 0.955     | 0.971      |
| **RoBERTa-Base-Amharic-embed**        | 110M   | **0.775↑** | **0.808↑** | **0.913↑** | **0.964↑** | **0.979↑** |


---

> 📖 **For further details on the baselines, see:**
> **Yu et al., 2024** — *Multilingual-E5*
> **Wang et al., 2024** — *Snowflake Arctic Embed*

---

## 📊  Amharic News: Sparse vs Dense Retrieval Comparison

The following table compares **sparse** and **dense** retrieval models trained on the Amharic News dataset. ColBERT-AM uses RoBERTa-Medium-Amharic as its backbone. Metrics reported include **MRR\@10**, **NDCG\@10**, and **Recall\@10/50/100**. Best results are shown in **bold**, and † marks statistically significant improvements *(p < 0.05)*.

## 📊 Summary Benchmark Results

| Type             | Model                          | MRR@10    | NDCG@10   | Recall@10 | Recall@50 | Recall@100 |
|------------------|--------------------------------|-----------|-----------|-----------|-----------|------------|
| Sparse retrieval | BM25-AM                        | 0.657     | 0.682     | 0.774     | 0.847     | 0.871      |
| Dense retrieval  | ColBERT-AM                     | 0.754     | 0.777     | 0.858     | 0.917     | 0.931      |
| Dense retrieval  | **RoBERTa-Base-Amharic-embed** | **0.775↑** | **0.808↑** | **0.913↑** | **0.964↑** | **0.979↑** |


---

> 📌 **Note:**
> * RoBERTa-Base-Amharic-embed achieves statistically significant improvements over ColBERT-AM across all metrics. Evaluation was performed using a paired t-test.
> * **For experiments on the 2AIRTC dataset**, please refer to the **Appendix section** of our [paper](#).
---

## 📄 Citation

If you use this repository, please cite our ACL 2025 Findings paper:

```bibtex
@inproceedings{mekonnen2025amharic,
  title={Optimized Text Embedding Models and Benchmarks for Amharic Passage Retrieval},
  author={Kidist Amde Mekonnen, Yosef Worku Alemneh, Maarten de Rijke },
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

## 🙏 Acknowledgments

This project builds on the [ColBERT repository](https://github.com/stanford-futuredata/ColBERT) by Stanford FutureData Lab. We sincerely thank the authors for open-sourcing their work, which served as a strong foundation for our Amharic ColBERT implementation and experiments.

