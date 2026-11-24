"""
Practical 05: Demonstrate basic Python libraries.

Covers:
- math: mathematical functions
- NumPy: arrays and numerical operations
- SciPy: descriptive statistics
- Pandas: tabular data handling
- Matplotlib and Seaborn: visualization
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def main():
    print("1. Math library")
    print("Square root of 81:", math.sqrt(81))
    print("Factorial of 5:", math.factorial(5))
    print("Value of pi:", math.pi)

    print("\n2. NumPy")
    values = np.array([10, 20, 30, 40, 50])
    print("Array:", values)
    print("Mean:", np.mean(values))
    print("Median:", np.median(values))
    print("Standard deviation:", np.std(values))
    print("Element-wise multiplication by 2:", values * 2)

    print("\n3. SciPy")
    print("Descriptive statistics:", stats.describe(values))
    print("Z-scores:", stats.zscore(values))

    print("\n4. Pandas")
    data = {
        "Student": ["A", "B", "C", "D", "E"],
        "Marks": [78, 85, 90, 67, 88],
        "Attendance": [82, 90, 95, 75, 92],
    }
    dataframe = pd.DataFrame(data)
    print(dataframe)
    print("\nData summary:")
    print(dataframe.describe())

    # 5. Matplotlib and Seaborn visualizations.
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 4))
    plt.plot(dataframe["Student"], dataframe["Marks"], marker="o")
    plt.title("Student Marks")
    plt.xlabel("Student")
    plt.ylabel("Marks")
    plt.tight_layout()

    plt.figure(figsize=(7, 4))
    sns.barplot(data=dataframe, x="Student", y="Attendance")
    plt.title("Student Attendance")
    plt.tight_layout()

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        dataframe[["Marks", "Attendance"]].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()

    print("\nDisplaying plots. Close each plot window to finish.")
    plt.show()


if __name__ == "__main__":
    main()
