import os

from src.config import (
    DATA_PATH,
    MODEL_DIR
)

from src.data_loader import load_data

from src.preprocessing import preprocess_data

from src.train import train_models

from src.evaluate import evaluate_model

from src.utils import save_model


def main():

    # Create model directory if not exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data
    df = load_data(DATA_PATH)

    # Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train models
    models = train_models(X_train, y_train)

    # Evaluate and save all models
    best_name = None
    best_score = 0

    for name, model in models.items():

        print(f"\n\n========== {name.upper()} ==========")

        macro_f1 = evaluate_model(model, X_test, y_test)

        model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
        save_model(model, model_path)

        # Track the best model by Macro F1
        if macro_f1 > best_score:
            best_score = macro_f1
            best_name = name

    # Save the best model separately for Streamlit
    best_model = models[best_name]
    best_model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    save_model(best_model, best_model_path)

    print(f"\n✅ Best model: {best_name.upper()}")
    print(f"   Macro F1 : {best_score:.4f}")
    print(f"   Saved at : {best_model_path}")


if __name__ == "__main__":
    main()