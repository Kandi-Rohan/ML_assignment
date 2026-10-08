from pathlib import Path

import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROLL_NO = "BT2024259"
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / ROLL_NO
# Use the same shuffled folds for every candidate model.
CV = KFold(n_splits=5, shuffle=True, random_state=42)


# These values are tried only for models that use regularization.
ALPHAS = (0.01, 0.1, 1.0, 10.0)


def build_model(degree, method, alpha=None):
    # Keep polynomial expansion and scaling inside the CV pipeline.
    if method == "linear":
        regressor = LinearRegression()
    elif method == "ridge":
        regressor = Ridge(alpha=alpha)
    elif method == "lasso":
        regressor = Lasso(alpha=alpha, max_iter=100000)
    elif method == "elastic_net":
        regressor = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=100000)
    else:
        raise ValueError(f"Unknown regression method: {method}")

    return Pipeline(
        [
            ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scaler", StandardScaler()),
            ("regressor", regressor),
        ]
    )


def select_model(train_df, degree_limit):
    features = train_df.drop(columns="y")
    target = train_df["y"]
    results = []

    # Compare every allowed degree and regularization setting.
    for degree in range(1, degree_limit + 1):
        candidates = [("linear", None)]
        candidates.extend(
            (method, alpha)
            for method in ("ridge", "lasso", "elastic_net")
            for alpha in ALPHAS
        )
        for method, alpha in candidates:
            model = build_model(degree, method, alpha)
            scores = cross_validate(
                model,
                features,
                target,
                cv=CV,
                scoring=("r2", "neg_mean_squared_error"),
                error_score="raise",
                n_jobs=-1,
            )
            result = {
                "degree": degree,
                "method": method,
                "alpha": alpha,
                "r2": scores["test_r2"].mean(),
                "mse": -scores["test_neg_mean_squared_error"].mean(),
            }
            results.append(result)
            alpha_text = "" if alpha is None else f", alpha={alpha:g}"
            print(
                f"Degree {degree:2d}, {method:12s}{alpha_text:>12s}: "
                f"CV R2={result['r2']:.4f}, CV MSE={result['mse']:.4f}"
            )

    # Select the candidate with the highest mean validation R2.
    best = max(results, key=lambda result: result["r2"])
    return best, results


def solve_variant(variant, degree_limit):
    # Keep each variant's input, predictions, and comparison results separate.
    train_path = DATA_DIR / f"{ROLL_NO}_train_{variant}.csv"
    test_path = DATA_DIR / f"{ROLL_NO}_test_{variant}.csv"
    output_path = ROOT / f"{ROLL_NO}_pred_{variant}.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    best, results = select_model(train_df, degree_limit)
    # Refit the selected pipeline on all training rows before prediction.
    comparison_path = ROOT / f"model_comparison_{variant}.csv"
    # Save all CV results so the selected model can be checked later.
    pd.DataFrame(results).to_csv(comparison_path, index=False)
    model = build_model(best["degree"], best["method"], best["alpha"])
    model.fit(train_df.drop(columns="y"), train_df["y"])
    # The test set is used only after model selection is complete.
    predictions = model.predict(test_df)

    pd.DataFrame({"y": predictions}).to_csv(output_path, index=False)
    alpha_text = "" if best["alpha"] is None else f", alpha={best['alpha']:g}"
    print(
        f"{variant}: method={best['method']}, degree={best['degree']}"
        f"{alpha_text}, CV R2={best['r2']:.4f}, CV MSE={best['mse']:.4f}"
    )
    print(f"Saved {comparison_path}")
    print(f"Saved {output_path}")
    return {"variant": variant, "best": best, "results": results}


if __name__ == "__main__":
    solve_variant("var1", 10)
    solve_variant("var2", 20)
