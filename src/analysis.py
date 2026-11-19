"""Análise exploratória e modelagem de salário por experiência."""

from pathlib import Path

import pandas as pd

from src.model import (
    cross_validation_metrics,
    evaluate_model,
    split_data,
    train_model,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "salary_data.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    print("=== VISÃO GERAL ===")
    print(f"Observações: {len(df)}")
    print(f"Valores ausentes: {df.isna().sum().sum()}")
    print(f"Duplicatas: {df.duplicated().sum()}")
    print(
        "Correlação experiência x salário:",
        f"{df['YearsExperience'].corr(df['Salary']):.3f}",
    )

    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    cv = cross_validation_metrics(df)

    print("\n=== MODELO ===")
    print(f"Intercepto: {model.intercept_:.2f}")
    print(f"Coeficiente da experiência: {model.coef_[0]:.2f}")

    print("\n=== TESTE ===")
    print(f"R²: {metrics['r2']:.3f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")

    print("\n=== VALIDAÇÃO CRUZADA (5 FOLDS) ===")
    print(f"R² médio: {cv['r2_mean']:.3f}")
    print(f"MAE médio: {cv['mae_mean']:.2f}")
    print(f"RMSE médio: {cv['rmse_mean']:.2f}")


if __name__ == "__main__":
    main()
