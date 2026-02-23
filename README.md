# Diabetes Early Detection (Beginner ML Project)

This is a simple **classification** project using the BRFSS 2015 diabetes indicators dataset.

**Goal:** predict whether someone is **at risk of diabetes** using basic health indicators.

- Original labels: `0 = healthy`, `1 = prediabetes`, `2 = diabetes`
- For this beginner version, we convert it to a **binary** target:
  - `0 -> Healthy`
  - `1 or 2 -> At risk`

## Project structure

- `data/sample/diabetes_sample.csv` – small sample dataset (for demo)
- `notebooks/` – EDA + model training notebooks
- `src/train_logreg.py` – small script to train and evaluate Logistic Regression

## How to run

### 1) Create environment
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Run notebooks
```bash
jupyter notebook
```

### 3) Run training script
```bash
python src/train_logreg.py --data data/sample/diabetes_sample.csv
```

## Tools
Python, pandas, scikit-learn, matplotlib
