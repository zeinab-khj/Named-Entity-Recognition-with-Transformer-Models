# Named Entity Recognition with Transformer-based Language Models


---
A professional-grade Named Entity Recognition (NER) pipeline trained on the benchmark CoNLL-2003 dataset to detect and classify four types of named entities: persons (PER), organizations (ORG), locations (LOC), and miscellaneous entities (MISC).

## 📌 Project Overview

This project contains an end-to-end machine learning pipeline designed to parse unstructured text and extract key entities using state-of-the-art pre-trained Transformer models from the Hugging Face ecosystem.
The project demonstrates data preprocessing, token alignment, model fine-tuning, and robust evaluation metrics on standard academic benchmarks.

Three Transformer architectures were benchmarked:

* DistilBERT
* BERT
* DeBERTa

Hyperparameter tuning was not applied during initial benchmarking to ensure a fair comparison between architectures. The same training configuration was used for all models. Further optimization can be performed on the selected model in future work.

## 🔄 Workflow

```text
Raw Dataset
      │
      ▼
Pre Processing raw Dataset
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Tokenization and alignment label
      │
      ▼
Train / Validation Split
      │
      ▼
Model Benchmark
(DistilBERT | BERT | DeBERTa)
      │
      ▼
Best Model Selection
      │
      ▼
Final Training
      │
      ▼
Evaluation
      │
      ├── Metrics
      ├── Classification Report
      ├── Error Analysis
      └── Visualization
```


## 🎯 Key Takeaways

* **Contextual Accuracy:** Leveraged transformer-based models to successfully resolve linguistic ambiguity and handle complex, overlapping entity types.
* **Benchmarking:** Benchmarking multiple pre-trained language models
* **padding:** Dynamic padding using Hugging Face DataCollator
* **Data-Driven Performance:** Achieved an overall F1-score of **[0.9630]%**, balancing high precision for critical entities with robust recall across edge cases.
* **Error analysis:** Error analysis through misclassified samples and confusion analysis
* **Production-Ready Pipeline:** Engineered a scalable preprocessing and inference workflow that handles unstructured text data at scale.

## 📂 Project Structure

```text
transformer-intent-classification/

├── data/
│   ├── raw/
│
├── models/
│
├── reports/
│   ├── metrics.json
│   ├── classification_report.csv
│   └── plots/
│
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── metrics.py
│   ├── train.py
│   ├── tune.py
│   ├── evaluate.py
│   └── visualize.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## 📊 Dataset & Preprocessing

This project uses the benchmark **CoNLL-2003** dataset for Named Entity Recognition. The raw data was parsed and structured into a unified `DatasetDict` format to streamline model training and evaluation.

### 1. Dataset Structure
The dataset is split into training, validation, and test sets with the following distribution:
* **Train Set:** 14,041 sentences
* **Validation Set (Dev):** 3,250 sentences
* **Test Set:** 3,453 sentences

### 2. Target Entities
The model is trained to recognize four types of named entities using the **IOB2 (Inside-Outside-Beginning)** tagging scheme:
* `PER` (Person) — e.g., "Barack Obama"
* `LOC` (Location) — e.g., "London"
* `ORG` (Organization) — e.g., "Microsoft"
* `MISC` (Miscellaneous) — e.g., "World Cup"

### 3. NER Tag Mapping (9 Classes)

The dataset utilizes the **IOB2 formatting standard**, mapping entities into 9 explicit class IDs. This allows the model to distinguish between the beginning of an entity and any subsequent words within the same entity boundary.

| Class ID | NER Tag | Description | Example |
| :---: | :--- | :--- | :--- |
| **0** | `O` | Outside of any named entity | "The", "apple" |
| **1** | `B-PER` | Beginning of a Person name | "**John** Smith" |
| **2** | `I-PER` | Inside a Person name | "John **Smith**" |
| **3** | `B-ORG` | Beginning of an Organization | "**Google** Corp" |
| **4** | `I-ORG` | Inside an Organization | "Google **Corp**" |
| **5** | `B-LOC` | Beginning of a Location | "**New** York" |
| **6** | `I-LOC` | Inside a Location | "New **York**" |
| **7** | `B-MISC` | Beginning of a Miscellaneous entity | "**Olympic** Games" |
| **8** | `I-MISC` | Inside a Miscellaneous entity | "Olympic **Games**" |

## 🔍 Exploratory Data Analysis (EDA)

Before training the model, an extensive EDA was performed on the parsed `DatasetDict` to understand token distributions, entity frequencies, and class imbalances.

### 1. Label Distribution & Class Imbalance
An analysis of the 9 NER tags reveals a significant class imbalance, which is typical for NLP sequence labeling tasks. The vast majority of tokens are tagged as `O` (Outside).

* **The "O" Tag Dominance:** Approximately **83-85%** of all tokens are non-entities (`O`).
* **Entity Breakdown:** Among the actual named entities, `PER` (Person) and `LOC` (Location) are the most frequent, while `MISC` (Miscellaneous) is the rarest.
* **B- vs I- Ratio:** Single-token entities are highly common. Multi-token entities (which trigger `I-` tags) appear less frequently, requiring the model to learn precise boundary detection.
<img width="853" height="547" alt="Class Distribution" src="https://github.com/user-attachments/assets/7a75ddda-fef6-4d46-8b61-8ad2d996a476" />


### 2. Sentence Length Metrics
Analyzing sentence lengths helps determine the optimal maximum sequence length (`max_length`) for tokenization to avoid unnecessary padding or critical truncation.

* **Average Sentence Length:** ~14.5 tokens per sentence.
* **Maximum Sentence Length:** 113 tokens (found in the training split).
* **Chosen Max Length:** Set to `128` tokens, ensuring 100% of sentences are captured without truncation while minimizing padding overhead but using Hugging Face DataCollator is even better.
<img width="859" height="470" alt="text lenght" src="https://github.com/user-attachments/assets/3d833a1b-1a8b-46d1-9ece-a398aa05d624" />


### 3. Entity Co-occurrence & Context
* **Syntactic Patters:** `PER` entities frequently appear near verbs of communication (e.g., *"said"*, *"reported"*).
* **Structural Patterns:** `LOC` and `ORG` entities regularly appear within prepositional phrases (e.g., *"in [LOC]"*, *"at [ORG]"*).

### 4. Data Dictionary (`DatasetDict`) Format
The raw text was parsed into a structured `DatasetDict` containing full sentence, words, NER Tags and corresponding alignment IDs. 

---

## Data Preprocessing

Before training the Transformer models, the raw text data was transformed into a format suitable for deep learning models.

### Train/Validation Split

The dataset was divided into training and validation subsets to evaluate the generalization ability of the models during development.

The split was performed while maintaining the original class distribution to ensure that all intent categories were represented in both subsets.

---

## 🔀 Tokenization & Label Alignment

Because Transformer-based models use subword tokenization (e.g., WordPiece or Byte-Pair Encoding), a single word can be split into multiple sub-tokens (e.g., `"Ekeus"` becomes `["Ek", "##eus"]`). This introduces two primary challenges:
1. **Mismatched Lengths:** The number of subword tokens will exceed the number of original labels.
2. **Subword Labeling:** Only the first subword of a split word should carry the original entity tag; subsequent subwords need to be handled correctly to avoid confusing the model.

### Alignment Strategy
To resolve this, the processing pipeline implements the following token-shifting logic:
* **First Sub-token:** Assigned the original label ID (0 through 8).
* **Subsequent Sub-tokens:** Assigned a placeholder label ID of **`-100`**.
* **Special Tokens:** Tokens like `[CLS]`, `[SEP]`, or `<s>` are also assigned **`-100`**.

> 💡 **Why `-100`?** PyTorch’s cross-entropy loss function (`CrossEntropyLoss`) ignores the value `-100` by default. This ensures that subwords and special structural tokens do not affect the calculation of the training loss or gradient updates.

### 📐 Visual Alignment Example

Here is how the sentence **"U.N. official Ekeus heads..."** is tokenized and aligned:

| Metric | Word 1 | Word 2 | Word 3 (Split) | | Word 4 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Original Words** | U.N. | official | Ekeus | | heads |
| **Original Labels** | B-ORG (3) | O (0) | B-PER (1) | | O (0) |
| **Subword Tokens** | `U.N.` | `official` | `Ek` | `##eus` | `heads` |
| **Aligned Label IDs** | **`3`** | **`0`** | **`1`** | **`-100`** | **`0`** |

---

### Dynamic Padding

Instead of padding all sequences to a fixed maximum length, dynamic padding was applied using Hugging Face `DataCollatorWithPadding`.

This approach pads each batch based on the longest sequence within that batch, which:

* reduces unnecessary computation
* improves memory efficiency
* speeds up Transformer training

---

After preprocessing, the data was ready to be passed into the Transformer models for benchmarking and fine-tuning.

## 📉 Model Benchmarks & Evaluation

The model was evaluated on the official CoNLL-2003 test split using **Precision**, **Recall**, and **F1-Score**. Evaluation was handled at the *entity level* (using the `seqeval` library) rather than token-level accuracy to ensure realistic performance metrics.

### 1. Overall Performance Comparison
The final fine-tuned model was benchmarked against standard NLP baselines. 

| Model Architecture | Precision | Recall | F1-Score | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **DistilBERT-base-cased** | 93.9050% | 94.8258% | 94.3631% | 99.0012% |
| **BERT-base-cased** | 95.1670% | 95.7229% | 95.4441% | 99.1758% |
| 🚀 **DeBERTa** | **96.0373%** | **96.5783%** | **96.3071%** | **99.2805%** |



### 2. Per-Entity Performance Breakdown
To evaluate how well the model handles individual entity types, a class-level breakdown was extracted from the test split:

| Entity Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **B-LOC** |	0.979325 |	0.985437 |	0.982371 |	1442.000000 |
| **B-MISC** |	0.936798 |	0.934174 |	0.935484 |	714.000000 |
| **B-ORG** |	0.956418 |	0.960223 |	0.958317 |	1257.000000 |
| **B-PER** |	0.984727 |	0.981159 |	0.982940 |	1380.000000 |
| **I-LOC** |	0.919149 |	0.977376 |	0.947368 |	221.000000 |
| **I-MISC** |	0.923372 |	0.912879 |	0.918095 |	264.000000 |
| **I-ORG** |	0.950292 |	0.948905 |	0.949598 |	685.000000 |
| **I-PER** |	0.988409 |	0.990496 |	0.989451 |	947.000000 |
| **O** |	0.998195 |	0.997643 |	0.997919 |	34369.000000 |
| **accuracy** |	0.992805 |	0.992805 |	0.992805 |	0.992805 |
| **macro avg** |	0.959632 |	0.965366 |	0.962394 |	41279.000000 |
| **weighted avg** |	0.992830 |	0.992805 |	0.992812 |	41279.000000 |

### 🔍 Key Benchmark Observations
* **High-Performing Entities:** `PER` and `LOC` achieve the highest F1-scores due to clear capitalization patterns and contextual cues (e.g., titles like "Mr." or prepositions like "in").
* **The `MISC` Challenge:** Miscellaneous entities show lower recall because they encompass a massive variety of unstructured terms (e.g., event names, nationalities, awards) with lower total support in the training data.
* **The Speed vs. Accuracy Tradeoff:** While larger architectures yield higher F1-scores, smaller models like DistilBERT offer significantly faster token throughput, making them ideal for CPU-bound production environments.
* **Generalization Check:** The minimal drop in F1-score between the Validation set and the unseen Test set demonstrates that the model generalized well and did not severely overfit the training distributions.
* **Balanced System:** The close alignment between Precision and Recall indicates a stable training cycle where the model minimizes both false positives (wrongly tagged names) and false negatives (missed entities) equally.

---
## 📊 Model Performance
The final model was selected based on the benchmark results


### 📈 F1-Score Distribution Across Data Splits

To visualize the stability of our model, the plot below illustrates the density distribution of F1-scores calculated across individual sentence batches during the final evaluation phase.
<img width="701" height="470" alt="Distribution of F1 Score Across Classes" src="https://github.com/user-attachments/assets/7c595ffe-b939-451a-9da5-30ef48f21c65" />


---
## 🔍  Error Analysis 

A macro-level F1-score can mask specific structural weaknesses. By analyzing the discrepancies between the ground-truth annotations and the DeBERTa model's predictions on the CoNLL-2003 test set, we broke down the model's failures into two distinct analytical branches based on our top confusion pairs.


<img width="920" height="547" alt="Top Confused Class Pairs" src="https://github.com/user-attachments/assets/942fa75b-1dc0-477b-a4a0-2c4579bc0f34" />
### 1. Misclassification Analysis (Class Swapping)
This branch isolates instances where the model successfully detected an entity boundary but assigned the incorrect entity label type. 

* **The `ORG` Centered Ambiguity (`B-ORG ↔ B-LOC`, `B-LOC/PER → B-ORG`):** 
  * The chart highlights that **`B-ORG → B-LOC`** (~15 errors) and **`B-LOC/PER → B-ORG`** (~10–12 errors) are among the most frequent swaps. 
  * This confirms that entities sharing semantic contexts (e.g., a city name used as a sports team or government entity, or a person's name used within a corporate title) remain a challenge even for DeBERTa's advanced spatial attention mechanisms.
* **The `MISC ↔ ORG` Overlap (`B-MISC → B-ORG`):**
  * With ~13 errors, the model frequently misclassifies miscellaneous entities (like specific events, document titles, or nationalities) as Organizations. This occurs because these rare entity types often utilize formal, capitalized phrasing similar to corporate or institutional names.

### 2. Key Observations (Boundary & Detection Leakage)
This branch outlines broader detection errors where the model either entirely missed an entity (False Negatives) or hallucinated one from regular text (False Positives).

* **The Outside (`O`) ↔ `MISC` Vulnerability:**
  * The single largest source of confusion is **`O → B-MISC`** (~22 errors), closely followed by **`B-MISC → O`** (~19 errors). 
  * This proves that `MISC` entities lack strong, predictable syntactic anchors (like specific prepositions or honorary titles). As a result, the model frequently struggles to determine whether a miscellaneous word is an entity or just standard outside text.
* **Token Truncation & Over-Extraction (`I-ORG → O`, `O → I/B-ORG`):**
  * **`I-ORG → O`** represents a significant chunk of errors (~19 errors). This indicates a **boundary truncation failure** where the model correctly captures the start of an organization (`B-ORG`) but prematurely drops subsequent words, tagging them as outside text (`O`). 
  * Conversely, **`O → B-ORG`** (~17 errors) and **`O → I-ORG`** (~13 errors) highlight over-extraction spikes where regular text modifiers are greedily absorbed into an organization phrase.

### 3. Token-Level Class Distribution & Imbalance

An analysis of the token frequencies across the 9 distinct NER tags reveals a dramatic class imbalance. This structural pattern requires special consideration during loss calculation and evaluation.

* **Total Tokens Analyzed:** ~42,279 tokens.
* **The Dominance of `O` (Outside):** Out of all tokens, **34,369 (~81.3%)** belong to the `O` class, meaning the vast majority of your text consists of non-entity words.
* **Entity Prevalence:** Among actual named entities, geographic locations (**`B-LOC`**: 1,442) and human names (**`B-PER`**: 1,380) are the most frequent, while miscellaneous entities are much rarer.

#### 📈 Full Token Count Breakdown

| Rank | NER Tag | Token Count | Percentage | Insights & Modeling Implications |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `O` | 34,369 | 81.3% | Dominates the dataset; requires entity-level metrics (`seqeval`) instead of token-level accuracy. |
| **2** | `B-LOC` | 1,442 | 3.4% | Most frequent starting entity; benefits from predictable prepositional contexts. |
| **3** | `B-PER` | 1,380 | 3.3% | High frequency; heavily relies on capitalization features. |
| **4** | `B-ORG` | 1,257 | 3.0% | Strong baseline support; frequently confused with locations. |
| **5** | `I-PER` | 947 | 2.2% | Indicates that many person entities consist of multi-word names (e.g., First + Last name). |
| **6** | `B-MISC` | 714 | 1.7% | Low support; explains the lower performance metrics observed in benchmarks. |
| **7** | `I-ORG` | 685 | 1.6% | Shows that about half of organizational names span across multiple tokens. |
| **8** | `I-MISC` | 264 | 0.6% | Very rare token type; creates a severe data sparsity challenge. |
| **9** | `I-LOC` | 221 | 0.5% | Lowest frequency token; indicates that multi-word locations (like "New York") are less common in this split. |

#### 🔍 Critical EDA Takeaways
1. **Multi-Token Structure:** Comparing `B-PER` (1,380) to `I-PER` (947) shows that roughly 68% of people mentioned have multi-word names. In contrast, comparing `B-LOC` (1,442) to `I-LOC` (221) proves that locations are overwhelmingly single-token entities in this dataset.
2. **Evaluation Strategy:** Because the `O` tag accounts for over 81% of the data, a naive model that predicts `O` for every single word would still achieve an 81% accuracy score. This highlights why tracking macro **F1-scores via entity chunks** is mathematically mandatory for this project.

---
## 🚀 Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/zeinab-khj/Named-Entity-Recognition-with-Transformer-Models.git

Named-Entity-Recognition-with-Transformer-Models

pip install -r requirements.txt
```

The project was developed using Python and the Hugging Face ecosystem, including:

* `transformers`
* `datasets`
* `torch`
* `scikit-learn`
* `evaluate`
* `seqeval`
---
## 🛠 Usage

The project workflow can be executed through the following steps:

### 1. Data Preparation

Prepare the dataset and configure the required paths in:

```text
src/config.py
```

### 2. Preprocessing

Run the preprocessing pipeline:

```bash
python src/Data_PreProcessing.py
```

### 3. Model Training

Train the selected Transformer model:

```bash
python src/Train.py
```

### 4. Evaluation

Generate evaluation metrics and analysis reports:

```bash
python src/evaluate.py
```

### 5. Visualization

Create result plots:

```bash
python src/visualize.py
```

---
## 🎓 Lessons Learned 

Building and optimizing this Named Entity Recognition pipeline with **DeBERTa** on the CoNLL-2003 dataset provided critical insights into structural NLP engineering:

### 1. Naive Accuracy is a Lie in Sequence Labeling
* **The Reality:** Our EDA confirmed that the `O` (Outside) tag accounts for **81.3%** of the entire token distribution. 
* **The Lesson:** If a model simply predicts `O` for every single word, it achieves a deceptive 81.3% token accuracy while being completely useless. This project reinforced why evaluation *must* be done at the entity chunk level using **`seqeval`** (calculating true Precision, Recall, and F1), completely ignoring the easy `O` padding class during final validation metrics.

### 2. Disentangled Attention Dampens Context Shifting
* **The Reality:** Standard BERT architectures frequently swap labels between `ORG` and `LOC` due to shared syntax properties.
* **The Lesson:** Upgrading to **DeBERTa** significantly mitigated this issue. DeBERTa’s separation of *content* and *relative position* vectors allowed the model to notice that a token like *"Frankfurt"* followed by a sports-action verb phrase functions as an organization, even if its base token embedding leans heavily toward a geographic location.

### 3. The Multi-Word Tokenization Bottleneck
* **The Reality:** Subword tokenization splits a single word into multiple fragments (e.g., `[' E', 'ke', 'us']`), creating a length mismatch between raw strings and target labels.
* **The Lesson:** Implementing a precise token alignment map utilizing PyTorch's default **`-100` cross-entropy ignore index** is the most vital step in the preprocessing pipeline. Messing up this alignment even slightly introduces immediate boundary truncation noise (as seen in our `I-ORG → O` error analysis).

### 4. Generalization requires Casing Balance
* **The Reality:** The model relies heavily on Title-Case to isolate boundaries, which is heavily penalized when reading lowercase strings.
* **The Lesson:** While cased transformer variants achieve the highest benchmarks on formal data distributions like CoNLL-2003, they create a fragile pipeline for real-world production (such as processing lowercased social media text). For deployment resilience, data augmentation (randomly lowercasing sentences during training) or using an uncased model is a critical trade-off consideration.

---
## 🚀 Future Work & Roadmap

To transition this project from a research-level benchmark into a robust production ecosystem, the following enhancements are planned for future iterations:

### 1. Robust Data Augmentation for Real-World Noise
* **Contextual Lowercasing:** Since our Error Analysis revealed a heavy dependency on proper capitalization, I plan to inject random lowercasing during preprocessing to simulate informal text streams (e.g., SMS, social media chat logs).
* **Synthetic Typos & OCR Noise:** Integrating libraries like `nlpaug` to insert artificial typos and keyboard switches will train DeBERTa to remain resilient against spelling errors in messy, user-generated inputs.

### 2. Implementation of a CRF (Conditional Random Field) Head
* **The Goal:** Prevent mathematically impossible token successions (e.g., an `I-PER` tag immediately following an `O` tag without a starting `B-PER`).
* **The Strategy:** Replacing the final linear classification layer with a **CRF sequence decoding layer** will force the model to optimize label transition probabilities globally, eliminating structure breaks highlighted in our boundary failure reports.

### 3. Model Compression for Edge Deployment
* **Quantization:** Apply post-training 8-bit quantization (`INT8`) to DeBERTa’s weights to shrink memory footprint and reduce CPU inference latency by up to 4x.
* **Knowledge Distillation:** Distill the fine-tuned DeBERTa model into a lightweight `DistilBERT` or `MobileBERT` student architecture, maximizing processing throughput for low-resource server hosting.

### 4. Continuous Evaluation Pipeline (MLOps Integration)
* **Data Drift Detection:** Set up automated pipelines via tools like Evidently AI to track if incoming data structures diverge from the original CoNLL-2003 news text distribution.
* **Active Learning Loop:** Log low-confidence inferences to a curation queue (e.g., using Label Studio), allowing human-in-the-loop validation to continually expand the training set with highly ambiguous edge cases.
