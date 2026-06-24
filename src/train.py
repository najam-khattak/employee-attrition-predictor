from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.logger import logger

from src.config import RANDOM_STATE


def train_models(X_train, y_train):

    logger.info("Training models...")

    models = {

        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=1000
            ))
        ]),

        "random_forest": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=5,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),

        "xgboost": Pipeline([
            ("model", XGBClassifier(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=4,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                scale_pos_weight=5,
                eval_metric="logloss",
                random_state=RANDOM_STATE
            ))
        ])
    }

    trained_models = {}

    for name, model in models.items():

        logger.info(f"Training {name}...")

        model.fit(X_train, y_train)

        trained_models[name] = model

        logger.info(f"{name} trained successfully.")

    return trained_models