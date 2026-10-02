from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROLL_NO = "BT2024259"
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / ROLL_NO
CV = KFold(n_splits=5, shuffle=True, random_state=42)


def build_model(degree):
    return Pipeline(
        [
            ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )


def select_degree(train_df, degree_limit):
    features = train_df.drop(columns="y")
    target = train_df["y"]
    results = []

    for degree in range(1, degree_limit + 1):
        model = build_model(degree)
        r2 = cross_val_score(model, features, target, cv=CV, scoring="r2")
        mse = -cross_val_score(model, features, target, cv=CV, scoring="neg_mean_squared_error")
        results.append({"degree": degree, "r2": r2.mean(), "mse": mse.mean()})
        print(f"Degree {degree:2d}: CV R2={r2.mean():.4f}, CV MSE={mse.mean():.4f}")

    best = max(results, key=lambda result: result["r2"])
    return best, results


def solve_variant(variant, degree_limit):
    train_path = DATA_DIR / f"{ROLL_NO}_train_{variant}.csv"
    test_path = DATA_DIR / f"{ROLL_NO}_test_{variant}.csv"
    output_path = ROOT / f"{ROLL_NO}_pred_{variant}.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    best, results = select_degree(train_df, degree_limit)
    model = build_model(best["degree"])
    model.fit(train_df.drop(columns="y"), train_df["y"])
    predictions = model.predict(test_df)

    pd.DataFrame({"y": predictions}).to_csv(output_path, index=False)
    print(f"{variant}: degree={best['degree']}, CV R2={best['r2']:.4f}, CV MSE={best['mse']:.4f}")
    print(f"Saved {output_path}")
    return {"variant": variant, "best": best, "results": results}


if __name__ == "__main__":
    solve_variant("var1", 10)
    solve_variant("var2", 20)
