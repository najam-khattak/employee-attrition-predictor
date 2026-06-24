import pandas as pd

from src.logger import logger


def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    """

    logger.info("Loading dataset...")

    df = pd.read_csv(path)

    logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

    return df