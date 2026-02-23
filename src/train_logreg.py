import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


def to_binary_target(y_series: pd.Series) -> pd.Series:
    # 0 = healthy, 1 = prediabetes, 2 = diabetes
    # binary: 0 = healthy, 1 = at risk
    return (y_series != 0).astype(int)


def main(data_path: str) -> None:
    df = pd.read_csv(data_path)

    # NOTE: in this dataset, the target column is usually named 'Diabetes_012'
    # If yours differs, change it here.
    target_col = "Diabetes_012"
    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' not found. Columns: {list(df.columns)[:10]}...")

    X = df.drop(columns=[target_col])
    y = to_binary_target(df[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # probability for ROC-AUC
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print("ROC-AUC:", round(auc, 4))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV file")
    args = parser.parse_args()
    main(args.data)
