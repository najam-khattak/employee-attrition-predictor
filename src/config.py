import os

# Absolute path — works on Windows, Mac, Linux, Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")

RANDOM_STATE = 42

TEST_SIZE = 0.2

DROP_COLUMNS = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]