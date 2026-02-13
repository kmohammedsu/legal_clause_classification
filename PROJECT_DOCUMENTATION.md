# Legal Contract Clause Classification – Code & Rationale

This document explains what each major code block in the project does, why it is needed, and the kinds of questions a professor (or reviewer) might ask about it. It aligns with the **Group Project-2 Final Report** and the implementation in `legal_clause_classification.ipynb`. The project is maintained for GitHub (see root `README.md` for setup and push instructions).

---

## Project Overview (from Final Report)

**Team:** Sowmya Polagoni, Venkat Saketh Kommi, Khaja Moinuddin Mohammed

**Goal:** Build and compare two supervised models for multi-class classification of legal contract clauses using the CUAD dataset: (1) a feedforward neural network on TF-IDF features, and (2) an LSTM-based classifier on tokenized sequences. The nine most frequent clause categories are modeled explicitly; the rest are grouped into an “Other” class (10 classes total).

**Data:** Contract Understanding Atticus Dataset (CUAD). After cleaning: **11,047** training clauses (CUADv1.json), **2,172** test clauses (test.json). Original 41 categories reduced to **10** (9 top + Other). Median clause length ~17 words; 95th percentile ~75 words.

**Main result:** The TF-IDF + feedforward model outperformed the LSTM on the held-out test set (Accuracy 0.9319, Macro F1 0.9296 vs. 0.9217 and 0.9205), indicating that keyword/lexical features are highly effective for this legal clause task.

**Notebook mapping:** §1–3 ≈ Notebook 1–3 (libraries, dataset, categories); §3–6 ≈ Notebook 4–5 (EDA, clause extraction, grouping, train/test split); §7–9 ≈ Notebook 5 (preprocessing, TF-IDF, LSTM prep); §8–11 ≈ Notebook 6 (model build, training, evaluation).

---

## 1. Libraries, Environment, and Reproducibility

### What the code does
- Imports core libraries: `pandas`, `numpy`, `json`, `os`, `pathlib`.
- Imports NLP tools: `re`, `nltk` (tokenizers, stopwords, lemmatizer).
- Imports ML tools: `sklearn` (TF‑IDF, train/test split, metrics, class weights).
- Imports deep learning tools: `tensorflow`, `keras` (Sequential, Dense, Dropout, Embedding, LSTM, callbacks).
- Imports visualization libraries: `matplotlib`, `seaborn`.
- Downloads necessary NLTK resources (punkt, stopwords, wordnet) if missing.
- Sets random seeds in NumPy and TensorFlow for reproducibility.
- Prints versions of TensorFlow/Keras and confirms successful setup.

### Why this is done
- Ensures that text preprocessing, modeling, and evaluation can be run end‑to‑end in a single environment.
- NLTK resources are required for tokenization, stopword removal, and lemmatization.
- Setting random seeds makes training runs more reproducible, which is critical when reporting performance.

### Questions you might be asked
- Why is reproducibility important in ML experiments?
- What randomness remains even after setting seeds (e.g., GPU nondeterminism, parallelism)?
- Why did you choose NLTK (vs. spaCy, etc.) for preprocessing?

---

## 2. Downloading and Inspecting the CUAD Dataset

### What the code does
- Checks whether the local `cuad/` directory exists.
- If not present, clones the CUAD GitHub repository.
- If a `data.zip` file exists and `data/` is missing, extracts it.
- Prints the contents of the `data/` directory (e.g., `CUADv1.json`, `test.json`, `train_separate_questions.json`).
- Loads and displays the category description CSV (if present), including:
  - Category names.
  - Descriptions and answer formats.
  - Group IDs (1–4, or “-”).
- Cleans category names by removing `"Category: "` prefixes.
- Stores a list of category names for later use.

### Why this is done
- Ensures the project is self-contained: if the dataset is missing, it fetches it automatically.
- The category description file documents the semantics of each clause type, which is important when interpreting model predictions.
- Knowing the number of categories (originally 41) and their groups helps motivate later decisions about grouping into a smaller number of classes.

### Questions you might be asked
- How many original clause categories exist in CUAD and how are they grouped?
- Why is it important to understand the legal meaning of each category before modeling?
- Did you verify licensing/usage rights for the CUAD dataset?

---

## 3. Clause Extraction and Initial DataFrame

### What the code does
- Defines a helper function to parse `CUADv1.json` and extract:
  - Contract ID and title.
  - Clause text.
  - Clause category.
- Builds a `DataFrame` (`clauses_df`) with columns like:
  - `contract_id`, `document_name`, `category`, `clause_text`.
- Drops empty or whitespace-only clauses.
- Drops duplicate clause texts.
- Computes frequency counts of each category and prints:
  - Top 10 most common categories.
  - Bottom 10 rarest categories.
  - Imbalance ratio (max count / min count).

### Why this is done
- Transforms the nested JSON structure into a flat table suitable for ML.
- Removing duplicates and empty clauses avoids data leakage and noise.
- The category frequency analysis quantifies class imbalance, which heavily influences model design and metrics.

### Questions you might be asked
- How did you define a “clause” programmatically?
- Why is it problematic if many duplicate clauses remain in both train and test sets?
- How severe is the class imbalance and how might it affect a naïve classifier?

---

## 4. Text Length Analysis

### What the code does
- Adds `char_count` and `word_count` columns to `clauses_df`.
- Plots:
  - Histograms of character counts and word counts.
  - Boxplots of word counts for the top‑10 categories.
  - Cumulative distribution of text lengths with key percentiles (50, 75, 90, 95, 99).
  - Log-scale histogram for word counts.

### Why this is done
- Text length statistics guide architectural decisions:
  - Maximum sequence length for the LSTM (padding/truncation).
  - Whether TF‑IDF features will be sparse but manageable.
- Understanding the distribution helps interpret what “typical” clauses look like.

### Questions you might be asked
- How did the length distribution influence your chosen max sequence length for LSTM?
- What trade-offs exist between truncating long clauses and increasing max length?
- Did you consider splitting very long clauses or merging very short ones?

---

## 5. Category Grouping and Label Encoding

### What the code does
- Computes category counts and selects the top N categories (e.g., 9) by frequency.
- All other categories are grouped into a single “Other” class.
- Constructs a mapping from original categories to grouped categories.
- Applies the mapping to produce a `category_grouped` column.
- Uses `LabelEncoder` to convert grouped categories into integer labels 0–9.
- Prints the label mapping and sample counts per label.

### Why this is done
- The original 41 categories contain many very rare labels (some with fewer than 30 clauses), which are difficult for a supervised classifier to learn.
- Grouping into 9 focused clause types plus `Other` makes the problem more tractable while still preserving important clause distinctions.
- Integer labels are required for Keras’ sparse categorical cross‑entropy loss.

### Questions you might be asked
- Why did you choose 9 top categories instead of a different number?
- What information is lost by grouping rare categories into `Other`?
- Could grouping introduce bias if the “Other” class is too heterogeneous?

---

## 6. Train / Test Split Strategy

### What the code does
- Uses all processed clauses from `CUADv1.json` as the training source.
- Extracts and preprocesses clauses from `test.json` in the same way:
  - Cleaning, minimum word count filter, duplicate removal.
  - Category grouping to the same 10 classes.
  - Label encoding using the same encoder fitted on training data.
- If `test.json` is available and yields valid clauses, uses it as a dedicated hold‑out test set.
- Otherwise, falls back to a stratified train–test split from `CUADv1.json`.
- Calculates class weights on `y_train` using `compute_class_weight`.
- Prints distribution of labels in the test set and class weights for training.

### Why this is done
- Using `test.json` as an external hold‑out set avoids information leakage and mimics a real deployment scenario.
- Stratified splitting or external test ensures all classes are represented in train and test.
- Class weights compensate for severe class imbalance by penalizing mistakes on rare classes more heavily.

### Questions you might be asked
- Why is it better to use `test.json` as a separate hold‑out than an internal split?
- How are class weights computed and how do they change the loss function?
- Did you ensure there is no contract‑level leakage between train and test sets?

---

## 7. Text Preprocessing for TF‑IDF

### What the code does
- Implements `preprocess_text_for_tfidf`:
  - Lowercases text.
  - Removes non‑alphanumeric characters and extra whitespace.
  - Tokenizes words.
  - Removes English stopwords.
  - Lemmatizes tokens.
- Applies preprocessing to all clauses to create `text_processed`.
- Initializes a `TfidfVectorizer` with unigrams and bigrams (e.g. `max_features=5000`) and fits it on the training text only; transforms train and test to dense TF-IDF matrices.

### Why this is done
- Consistent preprocessing avoids data leakage and ensures test data is never used to build the vocabulary or IDF statistics.
- Lemmatization and stopword removal reduce noise while preserving legally meaningful terms; TF-IDF emphasizes discriminative keywords (e.g. “governing law,” “limitation of liability”).

### Questions you might be asked
- Why fit the vectorizer only on training data?
- What is the effect of using both unigrams and bigrams for legal text?
- How does `max_features` affect model size and performance?

---

## 8. TF-IDF Vectorization and Feedforward Model Architecture

### What the code does
- Builds TF-IDF feature matrices: training (11,047 × 5,000), test (2,172 × 5,000) using the same vectorizer.
- Defines a feedforward (MLP) model: Input(5000) → Dense(512, ReLU) → Dropout(0.3) → Dense(256, ReLU) → Dropout(0.3) → Dense(10, Softmax). ~2.7M parameters.
- Compiles with Adam optimizer and sparse categorical cross-entropy loss.

### Why this is done
- TF-IDF provides a strong baseline for legal text where clause types are often signaled by distinctive terminology. The MLP learns nonlinear decision boundaries over these features.
- As noted in the final report, this architecture is computationally efficient and well-suited to standardized legal language.

### Questions you might be asked
- Why 5,000 features? How would you choose this in practice?
- What is the role of dropout in this model?
- Why is sparse categorical cross-entropy used instead of one-hot + categorical cross-entropy?

---

## 9. LSTM Tokenization, Padding, and Sequence Model

### What the code does
- Computes max sequence length from training word-count percentiles (e.g. 95th + buffer; often ~138–150) and sets vocabulary size (e.g. 10,000).
- Uses Keras `Tokenizer` (fit on training only) with OOV token; converts texts to sequences and pads/truncates to `max_sequence_length` (post padding, post truncation).
- Builds LSTM model: Input(seq_len) → Embedding(vocab_size, 128) → LSTM(128, dropout=0.3, recurrent_dropout=0.2) → Dropout(0.3) → Dense(10, Softmax). ~1.4M parameters. Embeddings are learned (no pretrained).

### Why this is done
- Sequence modeling can capture phrases like “subject to,” “shall remain in effect unless terminated,” which bag-of-words cannot. Legal clauses often have conditional or obligation structure.
- Padding ensures fixed-length input for batching; truncation keeps memory manageable while covering most clauses (per report: 95th percentile ~75 words).

### Questions you might be asked
- Why use the same tokenizer and max length for train and test?
- What are the trade-offs of random vs. pretrained embeddings for legal text?
- How does recurrent dropout differ from standard dropout in LSTMs?

---

## 10. Training Strategy: Class Weights and Early Stopping

### What the code does
- Computes class weights with `sklearn.utils.class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)` and passes them to `model.fit(..., class_weight=class_weights)` for both models.
- Uses `EarlyStopping` (e.g. monitor=`val_loss`, patience=5, restore_best_weights=True) and optional `ModelCheckpoint`. Validation split typically 10%. Feedforward batch size 64; LSTM batch size 32.

### Why this is done
- Class imbalance (e.g. “Parties,” “Other” vs. rarer categories) would otherwise bias the model toward frequent classes. Balanced class weights upweight errors on rare classes in the loss.
- Early stopping limits overfitting and reduces training time; restoring best weights gives the best validation performance.

### Questions you might be asked
- How do class weights change the effective loss function?
- Why might the LSTM need a smaller batch size than the feedforward model?
- What are the risks of stopping too early (underfitting) vs. too late (overfitting)?

---

## 11. Evaluation and Metrics

### What the code does
- Predicts on the held-out test set; converts logits to class indices. Computes accuracy, precision, recall, and F1 using `sklearn.metrics` with **macro** averaging (and optionally weighted).
- Prints `classification_report` and builds confusion matrices; may plot confusion matrix heatmaps and training/validation loss and accuracy curves.

### Why this is done
- Macro-averaged precision, recall, and F1 give equal weight to each class, so performance on rare clause types is not overshadowed by dominant classes (as emphasized in the final report).
- Confusion matrices and learning curves support interpretation (e.g. which categories are confused, whether the model overfits).

### Questions you might be asked
- Why is macro F1 preferred over accuracy or weighted F1 for this dataset?
- How would you use the confusion matrix to prioritize model improvements?
- What does the gap between training and validation accuracy suggest for the LSTM vs. the feedforward model?

---

## 12. Results Summary (from Final Report)

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) |
|-------|----------|-------------------|----------------|------------|
| TF-IDF + Feedforward NN | 0.9319 | 0.9484 | 0.9319 | **0.9296** |
| LSTM-based Model       | 0.9217 | 0.9338 | 0.9217 | 0.9205 |

- The feedforward model achieves higher macro F1 and more balanced performance across classes. Clause types with clear wording (Agreement Date, Parties, Governing Law, Audit Rights, Insurance) get very high precision and recall; **License Grant** and **Other** are harder (lower precision or recall).
- The LSTM converges more slowly and shows more validation fluctuation; without pretrained embeddings or larger data, it does not surpass the TF-IDF baseline. The report concludes that keyword patterns and lexical features are highly effective for this legal clause classification task.

---

## 13. Discussion and Conclusion

- **Interpretation:** Legal clauses are often distinguished by standardized terminology rather than complex syntax; TF-IDF captures this well. Sequence modeling adds complexity without clear gain in this setup.
- **Practical takeaway:** The TF-IDF-based feedforward model is easier to interpret, faster to train, and cheaper to deploy, making it suitable for contract review and compliance workflows.
- **Future work (from report):** Data augmentation for rare categories; pretrained embeddings (e.g. BERT) or BiLSTM/attention for the LSTM; hyperparameter tuning; cross-validation and qualitative error analysis; multi-label or ensemble extensions.

---

## 14. References

1. Hendrycks, D., Burns, C., Chen, A., & Ball, S. (2021). CUAD: An expert-annotated NLP dataset for legal contract review. *arXiv:2103.06268*. https://arxiv.org/abs/2103.06268  
2. Hassan, F., Le, T., & Lv, X. (2021). Addressing legal and contractual matters in construction using natural language processing: A critical review. *Journal of Construction Engineering and Management*, 147(6), 04021049.  
3. Aejas, B., Belhi, A., & Bouras, A. (2024). Contract clause extraction using question-answering task. In *International Conference on Web Information Systems and Technologies* (pp. 345–365). Springer.  
4. Mohite, A., Sheik, R., & Nirmala, S. J. (2025). Improving legal text classification through data augmentation using deep learning models. In *Recent Advances in Computing* (pp. 345–358). CRC Press.  
5. Aejas, B., Belhi, A., & Bouras, A. (2025). Using AI to ensure reliable supply chains: Legal relation extraction for sustainable and transparent contract automation. *Sustainability*, 17(9), 4215.
