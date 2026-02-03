import logging
from sklearn.naive_bayes import MultinomialNB

logger = logging.getLogger("model_training")


def train_model(X_train, y_train, alpha: float = 1.0):
    """
    Train a Naive Bayes model.

    Args:
        X_train: TF-IDF transformed training features
        y_train: Training labels
        alpha: Smoothing parameter for MultinomialNB

    Returns:
        Trained model
    """

    try:
        logger.info("Starting model training")
        logger.info("Using MultinomialNB with alpha=%s", alpha)

        model = MultinomialNB(alpha=alpha)
        model.fit(X_train, y_train)

        logger.info("Model training completed successfully")
        return model

    except Exception as e:
        logger.error("Error during model training: %s", e)
        raise
