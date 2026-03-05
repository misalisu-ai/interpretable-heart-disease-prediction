from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.preprocessing import preprocess, encode_features
from src.model import build_pipeline, tune_model
from src.evaluation import evaluate
import joblib
import os

def main():

    df = load_data()
    df = preprocess(df)
    X, y = encode_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    best_model, params = tune_model(pipe, X_train, y_train)

    print("\n--- Tuning Results ---")
    print("Best parameters:", params)

    evaluate(best_model, X_train, y_train, X_test, y_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/heart_disease_model.pkl")
    print("\n Model saved to models/heart_disease_model.pkl")

if __name__ == "__main__":
    main()
