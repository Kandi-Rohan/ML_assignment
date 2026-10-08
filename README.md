# Polynomial Regression Assignment

This repository contains the training and inference code for roll number `BT2024259`.
The implementation uses the assigned datasets only:

- `BT2024259/BT2024259_train_var1.csv` and `BT2024259/BT2024259_test_var1.csv`
- `BT2024259/BT2024259_train_var2.csv` and `BT2024259/BT2024259_test_var2.csv`

## Included results

The prediction files, model-comparison results, report, and plots are already
included in this repository. The inference code is provided in `solve.py` for
reference or for generating the predictions again.

The script compares ordinary least squares, Ridge, Lasso, and Elastic Net polynomial regression. It evaluates every allowed degree and penalty strength with the same five-fold shuffled cross-validation, selects the candidate with the highest mean validation R2, fits that model on all training rows, and writes:

- `BT2024259_pred_var1.csv`
- `BT2024259_pred_var2.csv`

Each prediction file contains only the `y` column, in the same row order as its corresponding test file.

The complete model-comparison results are also exported to:

- `model_comparison_var1.csv`
- `model_comparison_var2.csv`

Each row records the degree, method, alpha value (when applicable), mean CV R2, and mean CV MSE.

The generated report and plots are included in the repository:

- `BT2024259_report.pdf`
- `plots/fig1_var1_mse_and_residuals.png`
- `plots/fig2_cv_mse_comparison.png`
- `plots/fig3_var1_best_mse_bars.png`
- `plots/fig4_actual_vs_predicted.png`
- `plots/fig5_cv_r2_curves.png`
- `plots/fig6_var2_best_mse_bars.png`
- `plots/fig_residuals.png`
- `plots/fig_residuals_var2.png`

The original assignment instructions are included in `ML_Assignment_1.pdf`.

## Model selection

| Variant | Features | Selected method | Degree | Alpha | Mean CV R2 | Mean CV MSE |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| var1 | 6 | Lasso | 5 | 0.01 | 0.9703 | 0.3013 |
| var2 | 3 | Ridge | 12 | 1 | 0.9935 | 0.2566 |

The full approach and rationale are documented in [BT2024259_report.pdf](BT2024259_report.pdf).

Repository link: https://github.com/Kandi-Rohan/ML_assignment