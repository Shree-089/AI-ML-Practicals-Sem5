# AI & ML Practicals — Semester V

**Course:** Artificial Intelligence and Machine Learning Lab  
**Course Code:** 23CSE3089  
**Institution:** MIT Art, Design and Technology University, Pune  
**Academic Year:** 2026–2027

This repository contains standalone Python reference implementations for the 12 experiments in the Semester V practical plan.

## Practical index

| No. | Experiment | Program |
|---:|---|---|
| 01 | Breadth First Search (BFS) | [`practicals/01_breadth_first_search.py`](practicals/01_breadth_first_search.py) |
| 02 | A* Search Algorithm | [`practicals/02_a_star_search.py`](practicals/02_a_star_search.py) |
| 03 | Tic-Tac-Toe game | [`practicals/03_tic_tac_toe.py`](practicals/03_tic_tac_toe.py) |
| 04 | Arithmetic and statistical calculations | [`practicals/04_arithmetic_statistics.py`](practicals/04_arithmetic_statistics.py) |
| 05 | Math, NumPy, SciPy, Pandas, Matplotlib and Seaborn | [`practicals/05_python_libraries.py`](practicals/05_python_libraries.py) |
| 06 | Data preprocessing: missing values, encoding and normalization | [`practicals/06_data_preprocessing.py`](practicals/06_data_preprocessing.py) |
| 07 | Linear Regression — Automobile fuel efficiency | [`practicals/07_automobile_linear_regression.py`](practicals/07_automobile_linear_regression.py) |
| 08 | Logistic Regression — Breast Cancer classification | [`practicals/08_breast_cancer_logistic_regression.py`](practicals/08_breast_cancer_logistic_regression.py) |
| 09 | Bayesian Classifier — Iris dataset | [`practicals/09_iris_bayesian_classifier.py`](practicals/09_iris_bayesian_classifier.py) |
| 10 | Decision Tree and Random Forest — Heart Disease | [`practicals/10_heart_disease_tree_forest.py`](practicals/10_heart_disease_tree_forest.py) |
| 11 | K-Means clustering on a multidimensional dataset | [`practicals/11_kmeans_clustering.py`](practicals/11_kmeans_clustering.py) |
| 12 | Hierarchical clustering — Iris dataset | [`practicals/12_iris_hierarchical_clustering.py`](practicals/12_iris_hierarchical_clustering.py) |

## Project structure

```text
AI-ML-Practicals-Sem5/
├── README.md
├── requirements.txt
├── .gitignore
├── practicals/
│   ├── 01_breadth_first_search.py
│   ├── 02_a_star_search.py
│   ├── 03_tic_tac_toe.py
│   ├── 04_arithmetic_statistics.py
│   ├── 05_python_libraries.py
│   ├── 06_data_preprocessing.py
│   ├── 07_automobile_linear_regression.py
│   ├── 08_breast_cancer_logistic_regression.py
│   ├── 09_iris_bayesian_classifier.py
│   ├── 10_heart_disease_tree_forest.py
│   ├── 11_kmeans_clustering.py
│   └── 12_iris_hierarchical_clustering.py
└── datasets/
    └── README.md
```

## Environment setup

Python 3.10 or newer is recommended.

### 1. Create and activate a virtual environment

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run a practical

Run commands from the repository root. For example:

```bash
python practicals/01_breadth_first_search.py
python practicals/08_breast_cancer_logistic_regression.py
```

Some programs display plots. Close a plot window to allow the script to continue.

## Dataset notes

- Practical 08 and Practical 09 use datasets bundled with scikit-learn; no separate CSV is required.
- Practical 11 uses the Wine dataset bundled with scikit-learn by default. It can also accept a local CSV path.
- Practical 07 uses the UCI Auto MPG dataset. See [`datasets/README.md`](datasets/README.md).
- Practical 10 uses the UCI Heart Disease Cleveland dataset. See [`datasets/README.md`](datasets/README.md).
- Practical 06 creates a small illustrative DataFrame when no CSV path is supplied. You can pass your own CSV path to demonstrate preprocessing on another open dataset.

## Important

These are educational reference implementations based on the experiment titles in the university practical plan. Read each script and adjust file paths, dataset columns, and presentation details to match the instructions given in your lab. Dataset versions can differ.

**Execution status:** The files have been prepared but have not been run or tested in this environment.
