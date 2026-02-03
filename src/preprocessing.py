import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger("data_preprocessing")


def preprocess_and_vectorize(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Preprocess data and apply TF-IDF vectorization.

    Args:
        df: Input DataFrame with columns ['target', 'text']
        test_size: Test split ratio
        random_state: Random seed

    Returns:
        X_train_vec, X_test_vec, y_train, y_test, vectorizer
    """

    try:
        logger.info("Starting preprocessing")

        # Encode target labels
        df["target"] = df["target"].map({"ham": 0, "spam": 1})

        # Safety check
        df = df.dropna(subset=["text", "target"])

        X = df["text"]
        y = df["target"]

        logger.info("Splitting data into train and test sets")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        logger.info("Applying TF-IDF vectorization")

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=3000
        )

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        logger.info("Preprocessing completed successfully")
        logger.info(
            "Train shape: %s | Test shape: %s",
            X_train_vec.shape,
            X_test_vec.shape
        )

        return X_train_vec, X_test_vec, y_train, y_test, vectorizer

    except Exception as e:
        logger.error("Error during preprocessing: %s", e)
        raise
