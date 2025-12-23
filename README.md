# Machine Learning multi-class classification of Wikipedia articles
This repository contains an end-to-end **big data / NLP** pipeline with the goal of exploring English Wikipedia content adn building an **automatic multi-class text classifier** able to assign each article to one of **15 thematic categories** (e.g., *Culture, Economics, Medicine, Technology, Politics, Science*) using **PySpark's Machine Learning Library (MLlib)**. The full project can be visualized in the [`wikipedia_ml_classification.ipynb`](https://github.com/lgucrl/machine-learning-wikipedia-classification/blob/main/wikipedia_ml_classification.ipynb) notebook.

---

## Dataset

The project uses a dataset of English Wikipedia articles (`wikipedia.csv`) originally containing **~150.000 rows** in raw form. Each row represents a Wikipedia article with the following columns:

- `title`: full title of the article  
- `summary`: brief introduction of the article  
- `documents`: complete content of the article  
- `category`: one of **15** article categories (including Culture, Economics, Medicine, Technology, Politics, Science, and others)

During the exploratory phase, null values and duplicated rows are removed, resulting in a smaller but cleaner working dataset.

---

## Project workflow

1. **Data ingestion with Spark**  
   The pipeline starts by initializing a Spark session and loading the CSV into a **Spark DataFrame** (with schema inference and quoting/escaping for long text fields). This step enables fast aggregations, transformations, and model training for large text corpora without moving the entire dataset into local memory.

2. **Exploratory Data Analysis (EDA)**  
   Before the modeling process, the dataset is profiled to measure size, check schema, and quantify missing values. Category counts are computed to reveal **class imbalance** (some labels less represented than others), which is important to consider when interpreting metrics. A **text-length** analysis (words per article) is then perforemed to highlight if categories differ in typical article size, followe by a **term frequency** exploration (including word clouds) to provide intuition about the vocabulary that characterizes each class.

3. **Data cleaning and text merging**  
   Rows with null values in essential text fields are dropped, and duplicates are removed to exclude repeated samples. Then, the `summary` and `documents` fields are concatenated into a single text column, combining short introductions with full content to provide richer context for classification.

4. **Stratified train/test split**  
   To preserve realistic label proportions, the dataset is split into training and test sets using a **stratified sampling** strategy (with 80% of each class for training). This reduces the risk that minority classes vanish from the test set and makes performance comparisons more meaningful under imbalance.

5. **Feature engineering with a PySpark ML pipeline**  
   A PySpark ML `Pipeline` is defined to get a reusable preprocessing procedure, including tokenization, stopword removal, TF vectors creation (`CountVectorizer`), TF-IDF weighting (`IDF`), standardization (`StandardScaler`) and label indexing (`StringIndexer`). This produces a sparse `features` vector and a numeric label column (`category_index`), ensuring that the same transformations fit on the training set are applied consistently to the test set.

6. **Training and comparing multi-class classifiers**  
   Two models are trained and evaluated: **multinomial Logistic Regression** and **Multinomial Naive Bayes**. Different values of vocabulary size, which determines the number of feature in the datasets, are choosen for each model (representing key hyperparameters). Models are trained on Spark to keep a distributed computation.

7. **Evaluating performances**  
   Predictions are assessed using Spark evaluators (accuracy, F1, weighted precision/recall) and extended analysis with **confusion matrices** and **ROC/Precision–Recall curves** computed from predicted probabilities. This allows to clearly observe which categories are more confused and how imbalance impacts minority-class performance.

---

## Tech stack

- **Python**
- **Apache Spark (PySpark)**
- **WordCloud** (token frequency visualization)
- **PySpark MLlib** (Pipeline, feature transformers, classifiers)
- **scikit-learn** (extra evaluation metrics)

