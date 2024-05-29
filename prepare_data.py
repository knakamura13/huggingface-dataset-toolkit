import os
import logging
import pandas as pd
from pathlib import Path
from ucimlrepo import fetch_ucirepo
from datasets import load_dataset, Split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_columns(df, column_names, use_one_hot_encoding=True, verbose=False):
    """
    Encodes specified columns in the DataFrame using one-hot encoding or ordinal encoding.

    Parameters:
        df (pd.DataFrame): The DataFrame to encode.
        column_names (list): List of columns to encode.
        use_one_hot_encoding (bool): If True, use one-hot encoding; otherwise, use ordinal encoding.
        verbose (bool): If True, print verbose output.

    Returns:
        pd.DataFrame: The DataFrame with encoded columns.
    """
    if verbose:
        logger.info("Encoding columns...")
    if use_one_hot_encoding:
        for name in column_names:
            enc = OneHotEncoder(handle_unknown="ignore")
            new_cols = enc.fit_transform(df[[name]]).toarray()
            new_col_names = [f"{name}_{n}".lower().replace("group ", "") for n in enc.categories_[0].tolist()]
            new_cols_df = pd.DataFrame(new_cols, columns=new_col_names, dtype=int)
            df = pd.concat([new_cols_df, df], axis=1)
        df.drop(columns=column_names, inplace=True)
    else:
        enc = OrdinalEncoder()
        df[column_names] = enc.fit_transform(df[column_names])
    if verbose:
        logger.info("Columns encoded successfully.")
    return df


def download_dataset(name, source, split, variant, verbose):
    if source.lower() == 'huggingface':
        return download_huggingface_dataset(name, variant, split, verbose)
    elif source.lower() == 'uci':
        return download_uci_dataset(int(name), verbose)
    raise ValueError(f"Unsupported source specified: {source}")


def download_huggingface_dataset(name, variant=None, split=Split.ALL, verbose=False):
    """
    Downloads a dataset from Huggingface and returns it as a DataFrame with standardized column names.

    Parameters:
        name (str): The name of the dataset to download.
        variant (str): The variant of the dataset.
        split (datasets.Split): The dataset split to download.
        verbose (bool): If True, print verbose output.

    Returns:
        pd.DataFrame: The downloaded dataset as a DataFrame.
    """
    if verbose:
        logger.info(f"\nDownloading dataset {name} from Huggingface...")
    df = load_dataset(name, variant, split=split, download_mode="reuse_cache_if_exists").to_pandas()
    if verbose:
        logger.info("Download complete.")

    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df


def download_uci_dataset(dataset_id, verbose=False):
    """
    Downloads a dataset from the UCI ML Repository and returns it as a DataFrame with standardized column names.

    Parameters:
        dataset_id (int): The ID of the UCI dataset to download.
        verbose (bool): If True, print verbose output.

    Returns:
        pd.DataFrame: The downloaded dataset as a DataFrame.
    """
    if verbose:
        logger.info(f"\nDownloading dataset {dataset_id} from UCI...")
    uci_dataset = fetch_ucirepo(id=dataset_id)
    df = pd.concat([uci_dataset.data.features, uci_dataset.data.targets], axis=1)
    if verbose:
        logger.info("Download complete.")

    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df


def convert_float_columns_to_int(df, verbose=False):
    """
    Converts float columns in the DataFrame to integers where possible.

    Parameters:
        df (pd.DataFrame): The DataFrame to convert.
        verbose (bool): If True, print verbose output.

    Returns:
        pd.DataFrame: The DataFrame with converted columns.
    """
    if verbose:
        logger.info("Converting float columns to integers where possible...")
    for column in df.select_dtypes(include=['float', 'float64']):
        if df[column].apply(lambda x: x.is_integer()).all():
            df[column] = df[column].astype(int)
    if verbose:
        logger.info("Conversion complete.")
    return df


def handle_missing_data(df, strategy="drop", fill_value=None, verbose=False):
    """
    Handles missing data in the DataFrame according to the specified strategy.

    Parameters:
        df (pd.DataFrame): The DataFrame to handle.
        strategy (str): The strategy to handle missing data ('drop', 'fill', 'impute').
        fill_value (any): The value to fill missing data with if strategy is 'fill'.
        verbose (bool): If True, print verbose output.

    Returns:
        pd.DataFrame: The DataFrame with missing data handled.
    """
    if verbose:
        logger.info(f"Handling missing data using strategy: {strategy}...")
    if strategy == "drop":
        df = df.dropna()
    elif strategy == "fill":
        if fill_value is not None:
            df = df.fillna(fill_value)
        else:
            df = df.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x.fillna(x.mode()[0]))
    elif strategy == "impute":
        imputer = SimpleImputer(strategy="mean")
        for column in df.select_dtypes(include=['float', 'float64', 'int']):
            df[column] = imputer.fit_transform(df[[column]])
    if verbose:
        logger.info("Missing data handled.")
    return df


def scale_data(df, columns=None, scale_type='standardize', verbose=False):
    """
    Scales data in the DataFrame using the specified scaling type.

    Parameters:
        df (pd.DataFrame): The DataFrame to scale.
        columns (list): The list of columns to scale. If None, all numeric columns are scaled.
        scale_type (str): The type of scaling ('standardize' or 'normalize').
        verbose (bool): If True, print verbose output.

    Returns:
        pd.DataFrame: The scaled DataFrame.
    """
    if verbose:
        logger.info(f"Scaling data using {scale_type} scaling...")

    scaler = StandardScaler() if scale_type == 'standardize' else MinMaxScaler()

    if columns:
        numeric_columns = df[columns].select_dtypes(include=['float64', 'int']).columns
    else:
        numeric_columns = df.select_dtypes(include=['float64', 'int']).columns

    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    if verbose:
        logger.info("Data scaling complete.")

    return df


def auto_balance_dataset(df, target_column, verbose=False):
    """
    Balances the dataset using random oversampling to ensure equal class distribution.

    Parameters:
        df (pd.DataFrame): The DataFrame to balance.
        target_column (str): The target column to balance.
        verbose (bool): If True, print verbose output.

    Returns:
        pd.DataFrame: The balanced DataFrame.
    """
    if verbose:
        logger.info("Balancing classes using random oversampling...")
    majority_class = df[target_column].value_counts().idxmax()
    minority_class = df[target_column].value_counts().idxmin()
    majority_df = df[df[target_column] == majority_class]
    minority_df = df[df[target_column] == minority_class]
    n_samples_to_duplicate = len(majority_df) - len(minority_df)
    if n_samples_to_duplicate <= 0:
        if verbose:
            logger.info("Classes are already balanced.")
        return df
    minority_upsampled = minority_df.sample(n=n_samples_to_duplicate, replace=True, random_state=1)
    balanced_df = pd.concat([majority_df, minority_df, minority_upsampled])
    if verbose:
        logger.info("Classes balanced.")
    return balanced_df.sample(frac=1, random_state=1).reset_index(drop=True)


def apply_stratified_sampling(df, stratify_column, stratify_sample_size):
    """
    Applies stratified sampling to the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to sample.
        stratify_column (str): The column to stratify by.
        stratify_sample_size (float): The proportion of the dataset to sample.

    Returns:
        pd.DataFrame: The sampled DataFrame.
    """
    if not stratify_column or not stratify_sample_size:
        return df
    if stratify_column not in df.columns:
        raise ValueError(f"The specified stratify_column '{stratify_column}' does not exist in the DataFrame.")
    sampled_df, _ = train_test_split(df, test_size=stratify_sample_size, stratify=df[stratify_column], random_state=1)
    return sampled_df


def drop_columns(df, cols_to_drop):
    """
    Drops specified columns from the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to modify.
        cols_to_drop (list): List of columns to drop.

    Returns:
        pd.DataFrame: The DataFrame with columns dropped.
    """
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        if df.empty:
            raise ValueError("All columns have been dropped; no data remains for processing.")
    return df


def get_dataset_nickname(dataset_name: str, nickname=None):
    """ Create a nickname for the dataset to be used in CSV filenames. """
    if not nickname:
        if "/" in dataset_name:
            nickname = dataset_name.split("/")[1]
        else:
            nickname = dataset_name
    nickname = nickname.lower().replace("-dataset", "").replace("-", "_")
    return nickname


def save_dataframe(df, file_path, verbose=False):
    """
    Saves the DataFrame to a CSV file.

    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        file_path (str): The path to save the CSV file.
        verbose (bool): If True, print verbose output.
    """
    Path(file_path).parent.mkdir(exist_ok=True)
    df.to_csv(file_path, index=False)
    if verbose:
        logger.info(f"DataFrame saved to {file_path}")


def process_data(df, cols_to_encode=None, use_one_hot_encoding=True, missing_data_strategy="drop",
                 missing_data_fill_value=None, scale_type=None, scale_columns=None,
                 convert_floats_to_ints=True, as_int=False, auto_balance=False,
                 stratify_column=None, verbose=False):
    """
    Processes the DataFrame by encoding columns, handling missing data, scaling, and balancing.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        cols_to_encode (list): List of columns to encode.
        use_one_hot_encoding (bool): If True, use one-hot encoding; otherwise, use ordinal encoding.
        missing_data_strategy (str): The strategy to handle missing data ('drop', 'fill', 'impute').
        missing_data_fill_value (any): The value to fill missing data with if strategy is 'fill'.
        scale_type (str): The type of scaling ('standardize' or 'normalize').
        scale_columns (list): List of columns to scale.
        convert_floats_to_ints (bool): If True, convert float columns to integers where possible.
        as_int (bool): If True, convert all columns to integers.
        auto_balance (bool): If True, balance the dataset using random oversampling.
        stratify_column (str): The column to stratify by for balancing.
        verbose (bool): If True, print verbose output.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    if cols_to_encode:
        if not set(cols_to_encode).issubset(df.columns):
            raise ValueError("One or more columns specified for encoding do not exist in the DataFrame.")
        df = encode_columns(df, cols_to_encode, use_one_hot_encoding, verbose)
    df = handle_missing_data(df, strategy=missing_data_strategy, fill_value=missing_data_fill_value, verbose=verbose)
    if scale_type:
        df = scale_data(df, columns=scale_columns, scale_type=scale_type, verbose=verbose)
    if convert_floats_to_ints:
        df = convert_float_columns_to_int(df, verbose=verbose)
    if as_int:
        df = df.astype(int)
    if auto_balance and stratify_column:
        df = auto_balance_dataset(df, stratify_column, verbose)
    return df


def download_clean_and_save_dataset(name, source='huggingface', variant=None, split=Split.ALL,
                                    data_dir="data", nickname=None, cols_to_drop=None, cols_to_encode=None,
                                    use_one_hot_encoding=True, missing_data_strategy="drop",
                                    missing_data_fill_value=None, scale_type=None, scale_columns=None,
                                    stratify_column=None, stratify_sample_size=0.1, convert_floats_to_ints=True,
                                    as_int=False, auto_balance=False, verbose=False):
    """
    Downloads, cleans, and saves a dataset from Huggingface or UCI.

    Parameters:
        name (str): The name or ID of the dataset.
        source (str): The source of the dataset ('huggingface' or 'uci').
        variant (str): The variant of the dataset.
        split (datasets.Split): The dataset split to download.
        data_dir (str): The directory to save the dataset.
        nickname (str): A nickname for the dataset file.
        cols_to_drop (list): List of columns to drop.
        cols_to_encode (list): List of columns to encode.
        use_one_hot_encoding (bool): If True, use one-hot encoding; otherwise, use ordinal encoding.
        missing_data_strategy (str): The strategy to handle missing data ('drop', 'fill', 'impute').
        missing_data_fill_value (any): The value to fill missing data with if strategy is 'fill'.
        scale_type (str): The type of scaling ('standardize' or 'normalize').
        scale_columns (list): List of columns to scale.
        stratify_column (str): The column to stratify by for balancing.
        stratify_sample_size (float): The proportion of the dataset to sample.
        convert_floats_to_ints (bool): If True, convert float columns to integers where possible.
        as_int (bool): If True, convert all columns to integers.
        auto_balance (bool): If True, balance the dataset using random oversampling.
        verbose (bool): If True, print verbose output.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    nickname = get_dataset_nickname(dataset_name=name, nickname=nickname)

    # Download the dataset and save the unmodified data to a CSV
    df = download_dataset(name, source, split, variant, verbose)
    save_dataframe(df, os.path.join(data_dir, f"{nickname}_original.csv"), verbose)

    # Process the dataset and save the processed data to a separate CSV
    df = drop_columns(df, cols_to_drop)
    df = apply_stratified_sampling(df, stratify_column, stratify_sample_size)
    df = process_data(df, cols_to_encode, use_one_hot_encoding, missing_data_strategy, missing_data_fill_value,
                      scale_type, scale_columns, convert_floats_to_ints, as_int, auto_balance, stratify_column, verbose)
    save_dataframe(df, os.path.join(data_dir, f"{nickname}.csv"), verbose)

    if verbose:
        logger.info(f"Dataset processing complete.")
        logger.info(f"Original dataset saved to {os.path.join(data_dir, f'{nickname}_original.csv')}")
        logger.info(f"Processed dataset saved to to {os.path.join(data_dir, f'{nickname}.csv')}")

    return df
