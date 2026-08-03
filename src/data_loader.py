import pandas as pd
from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directory
DATA_DIR = PROJECT_ROOT / "data"


def load_items():
    """
    Load the items dataset from data/items.csv.
    """
    items_path = DATA_DIR / "items.csv"
    return pd.read_csv(items_path)


def load_users():
    """
    Load the users dataset from data/users.csv.
    """
    users_path = DATA_DIR / "users.csv"
    return pd.read_csv(users_path)