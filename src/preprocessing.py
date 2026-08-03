import pandas as pd


def clean_text(value):
    """
    Clean a text value by removing extra spaces
    and converting it to lowercase.

    Missing values are replaced with an empty string.
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

    Missing values are replaced with an empty string.
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


def handle_missing_values(df):
    """
    Handle missing values in text-based columns.

    Missing values are replaced with an empty string
    so that they do not cause errors during preprocessing
    or recommendation matching.
    """
    data = df.copy()

    text_columns = [
        "Category",
        "Genre",
        "Tags",
        "Preferred_Category",
        "Preferred_Genre",
        "Preferred_Tags"
    ]

    for column in text_columns:
        if column in data.columns:
            data[column] = data[column].fillna("")

    return data


def combine_item_features(items_df):
    """
    Combine relevant item attributes into a single
    feature column for recommendation matching.

    The combined features include:
    - Category
    - Genre
    - Tags
    """
    items = items_df.copy()

    items["Combined_Features"] = (
        items["Category"].fillna("")
        + " "
        + items["Genre"].fillna("")
        + " "
        + items["Tags"].fillna("")
    ).str.strip()

    return items


def preprocess_items(items_df):
    """
    Preprocess the items dataset.

    Steps:
    1. Handle missing values.
    2. Clean category.
    3. Clean genre.
    4. Clean tags.
    5. Combine relevant features.
    """
    items = handle_missing_values(items_df)

    items["Category"] = items["Category"].apply(clean_text)
    items["Genre"] = items["Genre"].apply(clean_text)
    items["Tags"] = items["Tags"].apply(clean_tags)

    items = combine_item_features(items)

    return items


def preprocess_users(users_df):
    """
    Preprocess the users dataset.

    Steps:
    1. Handle missing values.
    2. Clean preferred category.
    3. Clean preferred genre.
    4. Clean preferred tags.
    """
    users = handle_missing_values(users_df)

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

    Returns:
        processed_items: Cleaned items DataFrame
        processed_users: Cleaned users DataFrame
    """
    processed_items = preprocess_items(items_df)
    processed_users = preprocess_users(users_df)

    return processed_items, processed_users