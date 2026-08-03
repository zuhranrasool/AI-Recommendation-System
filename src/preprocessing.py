import pandas as pd


def preprocess_items(items_df):
    """
    Preprocess the items dataset.

    Parameters:
        items_df (pd.DataFrame): Raw items dataset.

    Returns:
        pd.DataFrame: Preprocessed items dataset.
    """
    items = items_df.copy()

    # Convert text columns to strings and remove unnecessary spaces
    text_columns = ["Category", "Genre", "Tags"]

    for column in text_columns:
        items[column] = (
            items[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    return items


def preprocess_users(users_df):
    """
    Preprocess the users dataset.

    Parameters:
        users_df (pd.DataFrame): Raw users dataset.

    Returns:
        pd.DataFrame: Preprocessed users dataset.
    """
    users = users_df.copy()

    # Convert user preference columns to strings
    preference_columns = [
        "Preferred_Category",
        "Preferred_Genre",
        "Preferred_Tags"
    ]

    for column in preference_columns:
        users[column] = (
            users[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    return users


def preprocess_data(items_df, users_df):
    """
    Preprocess both items and users datasets.

    Returns:
        tuple: Preprocessed items and users DataFrames.
    """
    processed_items = preprocess_items(items_df)
    processed_users = preprocess_users(users_df)

    return processed_items, processed_users