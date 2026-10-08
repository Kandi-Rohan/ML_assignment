# Polynomial Regression Assignment

This repository contains the training and inference code for roll number `BT2024259`.
The implementation uses the assigned datasets only:

- `BT2024259/BT2024259_train_var1.csv` and `BT2024259/BT2024259_test_var1.csv`
- `BT2024259/BT2024259_train_var2.csv` and `BT2024259/BT2024259_test_var2.csv`

## Reproduce the predictions

Run the inference script from the repository root:

```text
python solve.py
python generate_plots.py
python make_report.py
```

The script compares ordinary least squares, Ridge, Lasso, and Elastic Net polynomial regression. It evaluates every allowed degree and penalty strength with the same five-fold shuffled cross-validation, selects the candidate with the highest mean validation R2, fits that model on all training rows, and writes:

- `BT2024259_pred_var1.csv`
- `BT2024259_pred_var2.csv`

Each prediction file contains only the `y` column, in the same row order as its corresponding test file.

The complete model-comparison results are also exported to:

- `model_comparison_var1.csv`
- `model_comparison_var2.csv`

Each row records the degree, method, alpha value (when applicable), mean CV R2, and mean CV MSE.

## Model selection

| Variant | Features | Selected method | Degree | Alpha | Mean CV R2 | Mean CV MSE |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| var1 | 6 | Lasso | 5 | 0.01 | 0.9703 | 0.3013 |
| var2 | 3 | Ridge | 12 | 1 | 0.9935 | 0.2566 |

The full approach and rationale are documented in [BT2024259_report.pdf](BT2024259_report.pdf).

Repository link: https://github.com/Kandi-Rohan/ML_assignment