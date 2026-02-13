# Legal Contract Clause Classification

**Multi-class classification of legal contract clauses using TF-IDF + Feedforward Neural Networks and LSTM**

Final project for **BUAN 5312 – Advanced Machine Learning** (Fall 2025). This repository implements and compares two deep-learning approaches for labeling contract clauses using the [Contract Understanding Atticus Dataset (CUAD)](https://github.com/TheAtticusProject/cuad).

---

## Authors & Acknowledgments

### Project team

- **Khaja Moinuddin Mohammed**
- **Venkat Saketh Kommi**
- **Sowmya Polagoni**

### Data & resources

- **CUAD (Contract Understanding Atticus Dataset)** — [The Atticus Project](https://www.atticusprojectai.org/) / [GitHub](https://github.com/TheAtticusProject/cuad). Expert-annotated NLP dataset for legal contract review; used under its published terms.
- **NLTK** — Bird, S., Klein, E., & Loper, E. *Natural Language Processing with Python*. Used for tokenization, stopwords, and lemmatization.
- **TensorFlow/Keras** — Used for feedforward and LSTM model implementation.
- **scikit-learn** — Used for TF-IDF, train/test splitting, label encoding, and evaluation metrics.

---

## Overview

The pipeline (1) downloads and parses CUAD, (2) explores categories and text statistics, (3) cleans and encodes the data and produces TF-IDF and sequence features, and (4) trains and evaluates two classifiers:

| Model | Description |
|-------|-------------|
| **Feedforward MLP (TF-IDF)** | Dense layers on TF-IDF features; sparse categorical cross-entropy. |
| **LSTM** | Embedding → LSTM → Dense; same loss and label set. |

**Reported results (see final report):** Feedforward achieves higher macro F1 (0.9296) than the LSTM (0.9205) on the held-out test set, indicating strong performance from lexical features for this clause taxonomy.

---

## Repository structure

All paths are relative to the **project root** (directory containing `Notebooks/` and `scripts/`). The code infers the project root whether you run from the root or from `Notebooks/`.

```
project_root/
├── scripts/
│   ├── 00_download_cuad.py      # (0) Download and extract CUAD → cuad/data/
│   ├── 01_load_cuad_clauses.py  # (1) Parse CUAD JSON → clause DataFrame
│   ├── download_cuad.py         # Wrapper (preserves imports)
│   └── load_cuad_clauses.py     # Wrapper (preserves imports)
├── Notebooks/
│   ├── 02_eda.ipynb             # (2) Category descriptions and EDA
│   ├── 03_data_preprocessing.ipynb  # (3) Cleaning, encoding, TF-IDF, tokenization → data/processed/
│   └── 04_legal_clause_classification.ipynb  # (4) Train and evaluate both models
├── cuad/                        # CUAD data (created by 00_download_cuad.py)
│   └── data/
├── data/processed/              # Prepared arrays and artifacts (from 03)
├── Models/                      # Saved Keras models (.h5)
├── Visualizations/              # Saved figures (.png)
├── PROJECT_DOCUMENTATION.md     # Code walkthrough and rationale
├── Group Project-2 Final Report.pdf
└── README.md
```

---

## Run order

Execute in this order (script 0 optional if you run notebook 2 first; notebook 2 will trigger the download):

| Step | Item | Description |
|------|------|-------------|
| **0** | `python scripts/00_download_cuad.py` | Download and extract CUAD (or run 02_eda and it will call this). |
| **1** | `01_load_cuad_clauses.py` | Used by notebooks 02 and 03 (no standalone run needed). |
| **2** | `Notebooks/02_eda.ipynb` | Category descriptions and exploratory data analysis. |
| **3** | `Notebooks/03_data_preprocessing.ipynb` | Clean data, encode labels, build TF-IDF and sequences; writes `data/processed/`. |
| **4** | `Notebooks/04_legal_clause_classification.ipynb` | Load prepared data, train and evaluate both models; writes `Models/` and `Visualizations/`. |

---

## Quick start

1. **Clone the repository** and use the project root as your working directory.

2. **Create an environment and install dependencies:**

   ```bash
   pip install pandas numpy scikit-learn tensorflow nltk matplotlib seaborn joblib
   ```

3. **Download NLTK data** (once, in Python):

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

4. **Run the pipeline** in order: script **0** (optional if running notebook 2), then notebooks **02** → **03** → **04**. Outputs go to `Visualizations/` and `Models/` at project root.

---

## Results summary

| Model | Test accuracy | Macro F1 |
|-------|----------------|----------|
| TF-IDF + Feedforward NN | 0.9319 | **0.9296** |
| LSTM-based classifier   | 0.9217 | 0.9205   |

The feedforward model slightly outperforms the LSTM on this setup. For methodology, interpretation, and discussion, see **Group Project-2 Final Report.pdf** and **PROJECT_DOCUMENTATION.md**.

---

## Documentation

- **PROJECT_DOCUMENTATION.md** — Section-by-section code walkthrough and rationale, aligned with the final report.
- **Group Project-2 Final Report.pdf** — Problem statement, methods, results, and conclusions.

---

## References

1. Hendrycks, D., Burns, C., Chen, A., & Ball, S. (2021). CUAD: An expert-annotated NLP dataset for legal contract review. *arXiv:2103.06268*. https://arxiv.org/abs/2103.06268  
2. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O’Reilly Media. (NLTK.)  
3. Hassan, F., Le, T., & Lv, X. (2021). Addressing legal and contractual matters in construction using natural language processing: A critical review. *Journal of Construction Engineering and Management*, 147(6), 04021049.  
4. Aejas, B., Belhi, A., & Bouras, A. (2024). Contract clause extraction using question-answering task. In *International Conference on Web Information Systems and Technologies* (pp. 345–365). Springer.  
5. Mohite, A., Sheik, R., & Nirmala, S. J. (2025). Improving legal text classification through data augmentation using deep learning models. In *Recent Advances in Computing* (pp. 345–358). CRC Press.  
6. Aejas, B., Belhi, A., & Bouras, A. (2025). Using AI to ensure reliable supply chains: Legal relation extraction for sustainable and transparent contract automation. *Sustainability*, 17(9), 4215.

---

## License and data use

- **Code** in this repository is for course and portfolio use.
- **CUAD** is used under the terms of [The Atticus Project / CUAD](https://github.com/TheAtticusProject/cuad). Please cite the CUAD paper (Hendrycks et al., 2021) and refer to their repository for licensing details.
