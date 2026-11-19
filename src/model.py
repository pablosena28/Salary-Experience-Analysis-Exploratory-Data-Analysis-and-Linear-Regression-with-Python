"""Treinamento e avaliação de regressão linear salário x experiência."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

FEATURE = "YearsExperience"
TARGET = "Salary"


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """Separa os dados em treino e teste."""
    X = df[[FEATURE]]
    y = df[TARGET]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Treina uma regressão linear simples."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: LinearRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Calcula R², MAE e RMSE no conjunto de teste."""
    predictions = model.predict(X_test)
    return {
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
    }


def cross_validation_metrics(
    df: pd.DataFrame,
    folds: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """Calcula métricas médias com validação cruzada."""
    X = df[[FEATURE]]
    y = df[TARGET]
    cv = KFold(n_splits=folds, shuffle=True, random_state=random_state)

    r2 = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2")
    mae = -cross_val_score(
        LinearRegression(),
        X,
        y,
        cv=cv,
        scoring="neg_mean_absolute_error",
    )
    rmse = np.sqrt(
        -cross_val_score(
            LinearRegression(),
            X,
            y,
            cv=cv,
            scoring="neg_mean_squared_error",
        )
    )

    return {
        "r2_mean": float(r2.mean()),
        "mae_mean": float(mae.mean()),
        "rmse_mean": float(rmse.mean()),
    }
