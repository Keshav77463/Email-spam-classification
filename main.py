import yaml

from src.data_ingestion import load_data, preprocess_data
from src.preprocessing import preprocess_and_vectorize
from src.train import train_model
from src.evaluation import evaluate_model


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as file:
        return yaml.safe_load(file)


def main():
    # Load configuration
    config = load_config()

    # 1. Data ingestion
    df = load_data(config["data"]["url"])
    df = preprocess_data(df)

    # 2. Preprocessing + TF-IDF
    X_train, X_test, y_train, y_test, vectorizer = preprocess_and_vectorize(
        df,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        max_features = config["tfidf"]["max_features"],
        stop_words = config["tfidf"]["stop_words"]
    )

    # 3. Model training
    model = train_model(
        X_train,
        y_train,
        alpha=config["model"]["alpha"]
    )

    # 4. Model evaluation
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
