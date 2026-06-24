from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.logger import logger


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    logger.info(f"Accuracy: {accuracy:.4f}")

    print("\n==============================")
    print("Classification Report")
    print("==============================")

    print(classification_report(y_test, predictions))

    print("\n==============================")
    print("Confusion Matrix")
    print("==============================")

    print(confusion_matrix(y_test, predictions))

    # Return macro F1 so main.py can compare models
    report = classification_report(y_test, predictions, output_dict=True)
    macro_f1 = report["macro avg"]["f1-score"]

    return macro_f1