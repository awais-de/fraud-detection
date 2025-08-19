import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Train candidate models and evaluate performance.
    Returns trained models and evaluation results.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():
        print(f"\n🔹 Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_proba)

        results[name] = {"report": report, "roc_auc": auc, "model": model}

        # Print metrics
        print(classification_report(y_test, y_pred))
        print("ROC-AUC:", auc)
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return results


def plot_roc_curves(results, X_test, y_test, output_path="models/roc_curves.png"):
    """
    Plot ROC curves for all models.
    """
    plt.figure(figsize=(8, 6))
    for name, info in results.items():
        model = info["model"]
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name)

    plt.title("ROC Curves for Fraud Detection Models")
    plt.savefig(output_path)
    plt.close()
    print(f"✅ ROC curves saved to {output_path}")


def save_best_model(results, save_dir="models"):
    """
    Save the best model based on ROC-AUC score.
    """
    os.makedirs(save_dir, exist_ok=True)
    best_model = max(results.items(), key=lambda x: x[1]["roc_auc"])
    best_model_name, best_model_info = best_model
    model_path = os.path.join(save_dir, f"{best_model_name.replace(' ', '_').lower()}_fraud.pkl")

    joblib.dump(best_model_info["model"], model_path)
    print(f"✅ Best model saved: {best_model_name} -> {model_path}")