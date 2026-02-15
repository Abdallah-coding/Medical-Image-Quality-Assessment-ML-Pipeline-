# Machine learning models require structured tabular data. We transformed images into numerical features. Now we train on those numbers

"""
This project involves a binary classification problem (good vs poor image quality). After researching common baseline models for binary classification, I chose
logistic regression because:

- It is simple and widely used
- It works well with small numerical feature sets
- It provides a clear decision boundary
- It is appropriate as a first baseline before exploring more complex models

"""

"""
Baseline classification step

In this file, I train a simple Logistic Regression model on the
image quality features extracted earlier.

The goal is not to build a complex model but to verify whether
the handcrafted features (sharpness, contrast, entropy, etc.)
are sufficient to separate good and poor quality images.

Since the problem is binary (good vs poor), Logistic Regression
is a natural first choice which is used for classification, especially binary classification: it is simple, interpretable and
appropriate for baseline evaluation. In our case Good quality --> 1 / Poor quality --> 0


I split the dataset into training and testing sets to evaluate
generalization. Features are standardized to avoid scale bias.

This baseline allows me to:
- validate the feature engineering step
- understand model evaluation metrics
- establish a reference before exploring more advanced methods
"""




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
