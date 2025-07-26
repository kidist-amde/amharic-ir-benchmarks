# Amharic-IR-Benchmarks

[![ACL 2025 Findings](https://img.shields.io/badge/Paper-ACL%202025%20Findings-b31b1b)](https://aclanthology.org/2025.findings-acl.543/)
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
│   ├── colbert-amharic-pylate/
│   └── embedding_models/
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
| **2AIRTC**       | Ad-hoc IR test collection           | [2AIRTC Website](https://www.irit.fr/AmharicResources/airtc-the-amharic-adhoc-information-retrieval-test-collection/) |
| **Amharic News** | Headline–body classification corpus | [Hugging Face](https://huggingface.co/datasets/rasyosef/amharic-news-category-classification)                       |

Scripts for downloading and preprocessing can be found in the `data/` folder.

---
## Models

Our Amharic text embedding and ColBERT models can be found in the following Hugging Face [Collection](https://huggingface.co/collections/rasyosef/amharic-text-embedding-models-679cb55eae1d498e3ac5bdc5)

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

## 📊 Amharic News: Sparse vs. Dense Retrieval Comparison

The table below presents a comparison between **sparse** and **dense** retrieval approaches on the **Amharic passage reterival** dataset. While BM25 represents traditional lexical matching, RoBERTa and ColBERT models leverage semantic embeddings optimized for Amharic. All models were trained and evaluated under the same data splits.

**ColBERT-RoBERTa-Base-Amharic**, leveraging late interaction with a RoBERTa backbone, delivers the highest retrieval quality across most metrics. Statistically significant gains are marked with ↑ (p < 0.05).

| Type             | Model                            | MRR@10    | NDCG@10   | Recall@10 | Recall@50 | Recall@100 |
|------------------|----------------------------------|-----------|-----------|-----------|-----------|------------|
| Sparse retrieval | BM25-AM                          | 0.657     | 0.682     | 0.774     | 0.847     | 0.871      |
| Dense retrieval  | RoBERTa-Base-Amharic-embed       | 0.755     | 0.808     | 0.913     | 0.964     | 0.979      |
| Dense retrieval  | **ColBERT-RoBERTa-Base-Amharic** | **0.843↑** | **0.866↑** | **0.939↑** | **0.973↑** | 0.979      |

---

> 📌 **Note**  
> - ColBERT-RoBERTa-Base-Amharic significantly outperforms RoBERTa-Base-Amharic on **all ranking metrics**, except Recall@100, where both models converge. Significance assessed using a paired t-test.  
> - For **additional experiments on the 2AIRTC dataset**, refer to the Appendix section of our [ACL 2025 Findings paper](https://arxiv.org/abs/2505.19356).

---

## 📄 Citation

If you use this repository, please cite our ACL 2025 Findings paper:

```bibtex
@inproceedings{mekonnen-etal-2025-optimized,
    title = "Optimized Text Embedding Models and Benchmarks for {A}mharic Passage Retrieval",
    author = "Mekonnen, Kidist Amde  and
      Alemneh, Yosef Worku  and
      de Rijke, Maarten",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.543/",
    pages = "10428--10445",
    ISBN = "979-8-89176-256-5",
    abstract = "Neural retrieval methods using transformer-based pre-trained language models have advanced multilingual and cross-lingual retrieval. However, their effectiveness for low-resource, morphologically rich languages such as Amharic remains underexplored due to data scarcity and suboptimal tokenization. We address this gap by introducing Amharic-specific dense retrieval models based on pre-trained Amharic BERT and RoBERTa backbones. Our proposed RoBERTa-Base-Amharic-Embed model (110M parameters) achieves a 17.6{\%} relative improvement in MRR@10 and a 9.86{\%} gain in Recall@10 over the strongest multilingual baseline, Arctic Embed 2.0 (568M parameters). More compact variants, such as RoBERTa-Medium-Amharic-Embed (42M), remain competitive while being over 13$\times$ smaller. Additionally, we train a ColBERT-based late interaction retrieval model that achieves the highest MRR@10 score (0.843) among all evaluated models. We benchmark our proposed models against both sparse and dense retrieval baselines to systematically assess retrieval effectiveness in Amharic. Our analysis highlights key challenges in low-resource settings and underscores the importance of language-specific adaptation. To foster future research in low-resource IR, we publicly release our dataset, codebase, and trained models at https://github.com/kidist-amde/amharic-ir-benchmarks."
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

