import pandas as pd

from src.model import evaluate_model, split_data, train_model


def sample_data():
    return pd.DataFrame(
        {
            "YearsExperience": [1, 2, 3, 4, 5, 6, 7, 8],
            "Salary": [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],
        }
    )


def test_train_model_has_positive_experience_coefficient():
    df = sample_data()
    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=0.25,
        random_state=42,
    )
    model = train_model(X_train, y_train)
    assert model.coef_[0] > 0


def test_evaluate_model_returns_expected_metrics():
    df = sample_data()
    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=0.25,
        random_state=42,
    )
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    assert set(metrics) == {"r2", "mae", "rmse"}
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
