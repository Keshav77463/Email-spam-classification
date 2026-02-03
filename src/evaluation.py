import logging
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

logger = logging.getLogger("model_evaluation")


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data.

    Args:
        model: Trained ML model
        X_test: TF-IDF transformed test features
        y_test: True test labels
    """

    try:
        logger.info("Starting model evaluation")

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        logger.info("Evaluation completed successfully")
        logger.info("Accuracy: %.4f", acc)
        logger.info("Confusion Matrix:\n%s", cm)
        logger.info("Classification Report:\n%s", report)

        # Also print for user visibility
        print("\nModel Evaluation Results")
        print("-" * 30)
        print(f"Accuracy: {acc:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(report)

    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise
