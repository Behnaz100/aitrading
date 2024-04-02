def split_data(df, train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15, target_col='target', drop_cols=[]):
    """
    Splits the DataFrame into training, validation, and test sets based on specified ratios. Additionally,
    it separates the features from the target variable.

    Parameters:
    - df: pandas DataFrame, the dataset to split.
    - train_ratio: float, the proportion of the dataset to include in the train split.
    - validation_ratio: float, the proportion of the dataset to include in the validation split.
    - test_ratio: float, the proportion of the dataset to include in the test split.
    - target_col: str, the name of the target variable column.
    - drop_cols: list of str, names of the columns to drop from the features.

    Returns:
    - X_train, y_train: Features and target variable for the training set.
    - X_val, y_val: Features and target variable for the validation set.
    - X_test, y_test: Features and target variable for the test set.
    """
    # Validate ratios sum to 1
    if not (train_ratio + validation_ratio + test_ratio) == 1:
        raise ValueError("The sum of ratios must be equal to 1.")

    # Calculate indices for splits
    total_samples = len(df)
    train_end = int(total_samples * train_ratio)
    validation_end = int(train_end + total_samples * validation_ratio)

    # Split the dataset
    train_data = df.iloc[:train_end]
    validation_data = df.iloc[train_end:validation_end]
    test_data = df.iloc[validation_end:]

    # Prepare features and target variables
    X_train, y_train = train_data.drop([target_col] + drop_cols, axis=1), train_data[target_col]
    X_val, y_val = validation_data.drop([target_col] + drop_cols, axis=1), validation_data[target_col]
    X_test, y_test = test_data.drop([target_col] + drop_cols, axis=1), test_data[target_col]

    return X_train, y_train, X_val, y_val, X_test, y_test
