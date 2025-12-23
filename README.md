# Machine Learning multi-class classification of Wikipedia articles
This repository contains an end-to-end **big data / NLP** pipeline with the goal of exploring English Wikipedia content adn building an **automatic multi-class text classifier** able to assign each article to one of **15 thematic categories** (e.g., *Culture, Economics, Medicine, Technology, Politics, Science*) using **PySpark's Machine Learning Library (MLlib)**. The full project can be visualized in the `wikipedia_ml_classification.ipynb` notebook.

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
