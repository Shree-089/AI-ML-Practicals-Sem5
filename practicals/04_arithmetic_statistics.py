"""
Practical 04: Arithmetic and statistical calculations.

Demonstrates arithmetic operations and descriptive statistics using
Python's math and statistics standard-library modules.
"""

import math
import statistics


def main():
    print("Arithmetic and Statistical Calculations")
    print("----------------------------------------")

    try:
        first = float(input("Enter the first number: "))
        second = float(input("Enter the second number: "))
    except ValueError:
        print("Please enter valid numeric values.")
        return

    print("\nArithmetic operations")
    print("Addition:", first + second)
    print("Subtraction:", first - second)
    print("Multiplication:", first * second)

    if second != 0:
        print("Division:", first / second)
        print("Floor division:", first // second)
        print("Remainder:", first % second)
    else:
        print("Division, floor division and remainder are undefined for zero divisor.")

    print("Power (first ** second):", first ** second)
    if first >= 0:
        print("Square root of first number:", math.sqrt(first))
    else:
        print("Square root of a negative real number is not defined.")

    raw_values = input(
        "\nEnter at least two numbers separated by spaces for statistics: "
    ).strip()

    try:
        values = [float(value) for value in raw_values.split()]
    except ValueError:
        print("The list contains an invalid number.")
        return

    if not values:
        print("No values were entered.")
        return

    print("\nStatistical calculations")
    print("Count:", len(values))
    print("Mean:", statistics.mean(values))
    print("Median:", statistics.median(values))
    print("Minimum:", min(values))
    print("Maximum:", max(values))
    print("Range:", max(values) - min(values))

    if len(values) >= 2:
        print("Sample variance:", statistics.variance(values))
        print("Sample standard deviation:", statistics.stdev(values))
    else:
        print("Sample variance and standard deviation need at least two values.")


if __name__ == "__main__":
    main()
