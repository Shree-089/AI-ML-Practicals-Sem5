# Dataset instructions

This directory is kept for local dataset files used by the practicals. Dataset files are not included in this repository.

## Practical 07 — UCI Auto MPG

- Dataset page: https://archive.ics.uci.edu/dataset/9/auto+mpg
- The script can download the classic Auto MPG text data automatically when no local file is supplied.
- Expected fields in the classic data format: `mpg`, `cylinders`, `displacement`, `horsepower`, `weight`, `acceleration`, `model_year`, `origin`, and `car_name`.
- Target variable: `mpg` (miles per gallon).

To use a local file, place the classic `auto-mpg.data` file in this folder, or pass its path as the first command-line argument to the script.

## Practical 10 — UCI Heart Disease

- Dataset page: https://archive.ics.uci.edu/dataset/45/heart+disease
- The script uses the Cleveland processed data format and can retrieve `processed.cleveland.data` when no local file is supplied.
- The classic processed file has 14 columns: `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, and `num`.
- The script converts the original target `num` into a binary target: `0` means no disease; values greater than `0` mean presence of disease.

For a CSV with headers, the script also accepts a target column named `num` or `target`.

## Practical 11 — K-Means

The default dataset is the Wine dataset made available by scikit-learn (the Wine recognition dataset, originally from UCI). No local file is needed. To use another multidimensional dataset, pass a CSV path as the first command-line argument. The script selects numeric columns for clustering.
