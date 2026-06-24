import pandas as pd

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

from src.config import (
    DROP_COLUMNS,
    TEST_SIZE,
    RANDOM_STATE
)

from src.logger import logger


def preprocess_data(df: pd.DataFrame):

    logger.info("Starting preprocessing...")

    # Drop unnecessary columns
    df = df.drop(columns=DROP_COLUMNS)

    # Encode target variable
    le = LabelEncoder()

    df["Attrition"] = le.fit_transform(df["Attrition"])

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    # Split features and target
    X = df.drop("Attrition", axis=1)

    y = df["Attrition"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    smote = SMOTE(random_state=RANDOM_STATE)

    X_train, y_train = smote.fit_resample(X_train, y_train)

    logger.info("Preprocessing completed.")

    return X_train, X_test, y_train, y_test