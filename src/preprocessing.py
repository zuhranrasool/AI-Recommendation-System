import pandas as pd


def clean_text(value):
    """
    Clean a text value by removing extra spaces
    and converting it to lowercase.
    """
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def clean_tags(value):
    """
    Clean comma-separated tags.

    Example:
        "AI, Cyberpunk, Virtual Reality"
        ->
        "ai, cyberpunk, virtual reality"
    """
    if pd.isna(value):
        return ""

    tags = str(value).split(",")

    cleaned_tags = [
        tag.strip().lower()
        for tag in tags
        if tag.strip()
    ]

    return ", ".join(cleaned_tags)


def preprocess_items(items_df):
    """
    Preprocess the items dataset.
    """
    items = items_df.copy()

    items["Category"] = items["Category"].apply(clean_text)
    items["Genre"] = items["Genre"].apply(clean_text)
    items["Tags"] = items["Tags"].apply(clean_tags)

    return items


def preprocess_users(users_df):
    """
    Preprocess the users dataset.
    """
    users = users_df.copy()

    users["Preferred_Category"] = (
        users["Preferred_Category"].apply(clean_text)
    )

    users["Preferred_Genre"] = (
        users["Preferred_Genre"].apply(clean_text)
    )

    users["Preferred_Tags"] = (
        users["Preferred_Tags"].apply(clean_tags)
    )

    return users


def preprocess_data(items_df, users_df):
    """
    Preprocess both items and users datasets.
    """
    processed_items = preprocess_items(items_df)
    processed_users = preprocess_users(users_df)

    return processed_items, processed_users