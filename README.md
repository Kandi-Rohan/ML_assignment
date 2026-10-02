# Polynomial Regression Assignment

This repository contains the training and inference code for roll number `BT2024259`.
The implementation uses the assigned datasets only:

- `BT2024259/BT2024259_train_var1.csv` and `BT2024259/BT2024259_test_var1.csv`
- `BT2024259/BT2024259_train_var2.csv` and `BT2024259/BT2024259_test_var2.csv`

## Reproduce the predictions

Run the inference script from the repository root:

```text
python solve.py
```

The script evaluates every allowed polynomial degree with five-fold shuffled cross-validation, selects the degree with the highest mean validation R2, fits that model on all training rows, and writes:

- `BT2024259_pred_var1.csv`
- `BT2024259_pred_var2.csv`

Each prediction file contains only the `y` column, in the same row order as its corresponding test file.

## Model selection

| Variant | Features | Degrees searched | Selected degree | Mean CV R2 | Mean CV MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| var1 | 6 | 1-10 | 4 | 0.9331 | 0.6799 |
| var2 | 3 | 1-20 | 8 | 0.9928 | 0.2878 |

The full approach and rationale are documented in [BT2024259_report.pdf](BT2024259_report.pdf).

Repository link: https://github.com/Kandi-Rohan/ML_assignment