import pandas as pd
from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directory
DATA_DIR = PROJECT_ROOT / "data"


def _load_csv(file_path):
    """
    Load a CSV file with basic error handling.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Invalid file path: {file_path}")

    data = pd.read_csv(file_path)

    if data.empty:
        raise ValueError(f"Dataset is empty: {file_path}")

    return data


def load_items():
    """
    Load the items dataset from data/items.csv.
    """
    items_path = DATA_DIR / "items.csv"
    return _load_csv(items_path)


def load_users():
    """
    Load the users dataset from data/users.csv.
    """
    users_path = DATA_DIR / "users.csv"
    return _load_csv(users_path)