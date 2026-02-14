# Machine learning models require structured tabular data. We transformed images into numerical features. Now we train on those numbers

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


def main():
    df = pd.read_csv("reports/features.csv")

    X = df[["sharpness", "contrast", "brightness", "entropy", "snr"]].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\nconfusion matrix:")
    print(confusion_matrix(y_test, preds))

    print("\nclassification report:")
    print(classification_report(y_test, preds, digits=3))

    print("\nbaseline done.")


if __name__ == "__main__":
    main()
