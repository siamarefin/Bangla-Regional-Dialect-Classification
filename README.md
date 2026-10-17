# Bengali Regional Dialect Classification with Character-Level GRU

**Natural Language Processing · Low-Resource Languages · Bengali Dialects · Deep Learning · Text Classification**

## Project overview

This project investigates automatic identification of a text sample's **district label** from regional Bengali language. The notebook explores regional variation in a six-class dataset, builds a character-level preprocessing pipeline, and trains a stacked gated recurrent unit (GRU) neural network to classify text by district.

The project is an **experimental baseline**, not a deployment-ready dialect identifier. The saved run reaches **22.29% validation accuracy** after five epochs; that result is approximately the frequency of the largest class and calls for stronger baselines and further model development.

## Research questions

- Can a character-level neural network capture distinctions among Bengali regional varieties?
- What does the label distribution and text-length profile reveal about the dataset?
- How well does a stacked GRU classify six district labels under the notebook's current split?
- What additional baselines and safeguards are needed for a credible district classification evaluation?

## Data

The notebook reads two CSV files from a Kaggle input directory:

- `train_regipa.csv`
- `test_regipa.csv`

It selects `Contents` (Bengali text) and `District` (target class), then **concatenates the two files** for exploratory analysis and subsequent modeling.

### Dataset summary

| Characteristic | Recorded value |
|---|---:|
| Total samples after concatenation | 39,252 |
| Columns selected | `Contents`, `District` |
| Missing values in selected columns | 0 |
| District classes | 6 |
| Mean text length | 31.59 characters |
| Median text length | 26 characters |
| Minimum / maximum text length | 1 / 306 characters |

### Class distribution

| District | Samples |
|---|---:|
| Kishoreganj | 8,751 |
| Narail | 7,829 |
| Tangail | 6,793 |
| Narsingdi | 5,862 |
| Rangpur | 5,255 |
| Chittagong | 4,762 |
| **Total** | **39,252** |

The classes are imbalanced: Kishoreganj is the largest at approximately **22.29%**, and Chittagong is the smallest. The notebook visualizes the class distribution and character-length distribution.

## Methodology

### 1. Data ingestion and exploratory analysis

1. Load the two REGIPA CSV files using pandas.
2. Select the text and district columns.
3. Concatenate both dataframes into a combined dataset.
4. Check non-null counts and class frequencies.
5. Measure the number of characters per text sample and visualize its distribution.

### 2. Text preprocessing

- Remove characters in Python's `string.punctuation` set and trim leading/trailing whitespace.
- Produce a version with space-separated individual characters for exploratory representation.
- Create an encoded integer label for each district with `LabelEncoder`.
- Export intermediary CSVs: `Dataset.csv`, `cleaned_dataset.csv`, `char_split_dataset.csv`, and `encoded_dataset.csv`.

**Implementation detail:** The notebook's final model applies `Tokenizer(char_level=True)` to the already character-spaced text. This should be examined carefully, since inserted spaces become part of the tokenized representation. The earlier `char_count` column is also calculated before cleaning and splitting, so it describes the original text rather than the later transformed version.

### 3. Tokenization and data split

- Lowercase the text.
- Fit a Keras character-level tokenizer and transform texts into integer sequences.
- Pad sequences to the maximum observed sequence length, using post-padding.
- One-hot encode the six district labels.
- Split the combined data into 80% training and 20% validation using `train_test_split(test_size=0.2, random_state=42, stratify=y)`.

### 4. Stacked GRU classifier

| Layer | Configuration |
|---|---|
| Embedding | 64-dimensional embeddings |
| GRU 1 | 128 units; returns sequences |
| GRU 2 | 64 units |
| Dropout | 0.3 |
| Dense | 32 units; ReLU |
| Dropout | 0.2 |
| Output | 6 units; Softmax |

Training uses **Adam**, **categorical cross-entropy**, **5 epochs**, and **batch size 32**. The notebook saves the model as `gru_text_classification_model.h5`.

## Recorded experimental results

The notebook output reports the following epoch-level training and validation metrics:

| Epoch | Training accuracy | Training loss | Validation accuracy | Validation loss |
|---|---:|---:|---:|---:|
| 1 | 21.68% | 1.7742 | 22.29% | 1.7699 |
| 2 | 21.95% | 1.7706 | 22.29% | 1.7689 |
| 3 | 22.48% | 1.7696 | 22.29% | 1.7690 |
| 4 | 22.36% | 1.7695 | 22.29% | 1.7690 |
| 5 | 22.16% | 1.7710 | **22.29%** | **1.7689** |

### Interpretation

- Validation accuracy remains flat across all five epochs.
- The achieved **22.29%** is approximately the proportion of the largest category (Kishoreganj: 8,751 / 39,252). Thus the notebook **does not demonstrate a gain above a majority-class baseline**.
- The recorded run does **not** include a confusion matrix, per-class precision/recall, macro F1, repeated-seed confidence intervals, or a final independent holdout evaluation.
- This is a useful starting point for investigating low-resource Bengali dialect classification, but it needs further controlled experiments before any strong performance claims are warranted.

## Evaluation limitations and reproducibility notes

1. **Original test-set reuse:** The code merges the originally supplied `train_regipa.csv` and `test_regipa.csv` *before* splitting again. Therefore its displayed validation results must **not** be presented as performance on the original untouched test set. A stronger evaluation should preserve the original test file for final assessment.
2. **Vocabulary fitted before the new split:** The notebook fits the tokenizer on the combined dataset before splitting. For stricter evaluation, fit it on the training partition only.
3. **Preprocessing choice:** Character-spacing the text before applying `char_level=True` may be redundant and changes the sequence representation. Compare this pipeline against direct Unicode character tokenization.
4. **Class imbalance:** Accuracy by itself can hide poor predictions on smaller district classes; report macro F1, per-class recall, and a confusion matrix.
5. **No comparison models:** Include majority-class, TF-IDF + linear classifier, and other neural baselines before attributing any improvement to GRU.
6. **District labels are not definitive speaker identity:** The dataset assigns linguistic samples to district categories, but these do not necessarily establish a speaker's birthplace or identity.

## Tech stack

- Python, Jupyter Notebook / Kaggle
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn (`LabelEncoder`, `train_test_split`)
- TensorFlow / Keras (Tokenizer, embeddings, GRU, Dense, Dropout)

## Running the notebook

1. Install the libraries in a Python environment:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow notebook
   ```

2. Obtain the appropriately licensed REGIPA data files and place them in an accessible directory.
3. Open `Bangla_District_Classification_GRU.ipynb` and replace the Kaggle-specific file paths with paths on your system.
4. Run cells in order to reproduce exploratory analysis, preprocessing, and the GRU training workflow.

   ```bash
   jupyter notebook Bangla_District_Classification_GRU.ipynb
   ```

5. Inspect the class counts, generated CSVs, and training log. Exact values can vary across machines and training runs.

**Note:** The raw REGIPA CSV files and the saved `.h5` model are not included in the source notebook attachment. If distributing those files, respect their licenses and verify their contents first.

## Files

```text
Bangla-Regional-Dialect-Classification/
├── README.md
└── Bangla_District_Classification_GRU.ipynb
```

The notebook currently writes intermediate artifacts into its working directory, including cleaned datasets and `gru_text_classification_model.h5`. Those are runtime outputs rather than files verified to be present in this repository.

## Future work

- Preserve the provided training/test partition and create a validation split **only** from training data.
- Build majority-class and TF-IDF + logistic regression / SVM benchmarks.
- Rework character and grapheme-level representations for Bengali script.
- Tune padding lengths, learning rate, network size, regularization, and training epochs.
- Evaluate bidirectional recurrent networks and pretrained Bengali-language encoders where appropriate.
- Analyze class imbalance using per-class precision, recall, macro F1, and confusion matrices.
- Repeat experiments over several seeds; inspect misclassified examples and dialect-specific lexical patterns.

## Research relevance

This work explores the challenges of **regional Bengali text classification** in a low-resource NLP setting, including class imbalance, character-level representations, and recurrent sequence modeling. The current results are best viewed as an experimental baseline and a foundation for improved evaluation and language-aware modeling.

## Keywords

`Bengali NLP` · `Dialect Classification` · `District Classification` · `Character-Level Modeling` · `GRU` · `Low-Resource Languages` · `Text Classification`
