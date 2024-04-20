import os
import pandas as pd
from pathlib import Path
from ucimlrepo import fetch_ucirepo
from datasets import load_dataset, Split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler


def encode_columns(df, column_names, use_one_hot_encoding=True):
    """
    Apply encoding to a list of column names on a DataFrame, dropping the original columns.

    Parameters:
        df (pandas.DataFrame): DataFrame to encode.
        column_names (list of str): Column names to be encoded.
        use_one_hot_encoding (bool): True to use one-hot encoding, False to use ordinal encoding.

    Returns:
        pandas.DataFrame: DataFrame with original columns dropped and encoded columns added.
    """
    print("Encoding columns...")

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
        df_encoded = df.copy()
        df_encoded[column_names] = enc.fit_transform(df[column_names])
        df = df_encoded

    print("Columns encoded successfully.")
    return df


def download_huggingface_dataset(name, variant=None, split=Split.ALL):
    """
    Download a Huggingface dataset and convert it to a DataFrame.

    Parameters:
        name (str): Name of the dataset to download from Huggingface.
        variant (str, optional): Variant of the dataset, if applicable.
        split (datasets.Split, optional): Which split of the dataset to load, default is all data.

    Returns:
        pandas.DataFrame: DataFrame representation of the dataset.
    """
    print(f"\nDownloading dataset {name} from Huggingface...")

    df = load_dataset(name, variant, split=split, download_mode="reuse_cache_if_exists").to_pandas()

    print("Download complete.")
    return df


def convert_float_columns_to_int(df):
    """
    Converts float columns in a DataFrame to int where possible, only if all float values have no decimal part.

    Parameters:
        df (pandas.DataFrame): DataFrame to process.

    Returns:
        pandas.DataFrame: DataFrame with float columns converted to int where applicable.
    """
    print("Converting float columns to integers where possible...")

    for column in df.select_dtypes(include=['float', 'float64']):
        if df[column].apply(lambda x: x.is_integer()).all():
            df[column] = df[column].astype(int)

    print("Conversion complete.")
    return df


def handle_missing_data(df, strategy="drop", fill_value=None):
    """
    Handles missing data in the DataFrame according to the specified strategy.

    Parameters:
        df (pandas.DataFrame): DataFrame to process.
        strategy (str): Strategy to handle missing data ('drop', 'fill', 'impute').
        fill_value (any, optional): Value used for filling missing data if strategy is 'fill'.

    Returns:
        pandas.DataFrame: DataFrame with missing data handled.
    """
    print(f"Handling missing data using strategy: {strategy}...")

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

    print("Missing data handled.")
    return df


def scale_data(df, columns=None, scale_type='standardize'):
    """
    Scale data columns in the DataFrame using specified scaling type.

    Parameters:
        df (pandas.DataFrame): DataFrame containing the data.
        columns (list of str, optional): List of column names to scale. If None, all columns are scaled.
        scale_type (str): Type of scaling ('standardize' or 'normalize').

    Returns:
        pandas.DataFrame: DataFrame with scaled columns.
    """
    print(f"Scaling data using {scale_type} scaling...")

    scaler = StandardScaler() if scale_type == 'standardize' else MinMaxScaler()
    if columns:
        df[columns] = scaler.fit_transform(df[columns])
    else:
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    print("Data scaling complete.")
    return df


def download_clean_and_save_dataset(name, source='huggingface', variant=None, split=Split.ALL,
                                    data_dir="data", nickname=None, cols_to_drop=None, cols_to_encode=None,
                                    use_one_hot_encoding=True, missing_data_strategy="drop",
                                    missing_data_fill_value=None, scale_type=None, scale_columns=None,
                                    stratify_column=None, stratify_sample_size=0.1, convert_floats_to_ints=True,
                                    as_int=False):
    """
    Download a dataset from Huggingface or UCI, clean, and save it as CSV with additional preprocessing options.

    Parameters:
        name (str): Name of the dataset or the UCI dataset ID (number).
        source (str): Source of the dataset ('huggingface', 'uci').
        variant (str, optional): Variant of the dataset, if applicable.
        split (datasets.Split, optional): Which split of the dataset to download (e.g., train, test).
        data_dir (str): Directory to save the dataset files.
        nickname (str, optional): Nickname to use for saving the dataset files.
        cols_to_drop (list of str, optional): List of columns to drop from the DataFrame.
        cols_to_encode (list of str, optional): List of columns to apply encoding.
        use_one_hot_encoding (bool): Whether to use one-hot encoding for categorical variables.
        missing_data_strategy (str): Strategy for handling missing data ('drop', 'fill', 'impute').
        missing_data_fill_value (any, optional): Value used for filling missing data if strategy is 'fill'.
        scale_type (str, optional): Specifies the type of scaling ('standardize', 'normalize'); if None, no scaling is applied.
        scale_columns (list of str, optional): Specifies the columns to scale; if None and scaling is applied, all columns are scaled.
        stratify_column (str, optional): Column name to use for stratifying the data, enables stratified sampling if provided.
        stratify_sample_size (float): Fraction of the dataset to retain in sampling.
        convert_floats_to_ints (bool): Whether to automatically convert float columns to integers where possible.
        as_int (bool): Whether to convert all columns to integer type after preprocessing.

    Returns:
        pandas.DataFrame: The processed dataset.
    """
    # Create a nickname for naming the CSV files
    nickname = nickname or name.split("/")[1] if "/" in name else name
    nickname = nickname.lower().replace("-dataset", "").replace("-", "_")

    # Download the dataset from the specified source
    if source == 'huggingface':
        df = download_huggingface_dataset(name, variant, split)
    elif source == 'uci':
        print(f"\nDownloading dataset {name} from UCI...")
        uci_dataset = fetch_ucirepo(id=int(name))
        df = pd.concat([uci_dataset.data.features, uci_dataset.data.targets], axis=1)
        print("Download complete.")
    else:
        raise ValueError(f"Unsupported source specified: {source}")

    # Validate if the stratify_column exists in the DataFrame
    if stratify_column and stratify_column not in df.columns:
        raise ValueError(f"The specified stratify_column '{stratify_column}' does not exist in the DataFrame.")

    # Save the unprocessed DataFrame
    Path(data_dir).mkdir(exist_ok=True)
    df.to_csv(os.path.join(data_dir, f"{nickname}_original.csv"), index=False)

    # Drop specified columns and check if any columns remain
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        if df.empty:
            raise ValueError("All columns have been dropped; no data remains for processing.")

    # Encode specified columns
    if cols_to_encode:
        if not set(cols_to_encode).issubset(df.columns):
            raise ValueError("One or more columns specified for encoding do not exist in the DataFrame.")
        df = encode_columns(df, cols_to_encode, use_one_hot_encoding)

    # Handle missing data
    df = handle_missing_data(df, strategy=missing_data_strategy, fill_value=missing_data_fill_value)

    # Stratified sampling
    if stratify_column and stratify_sample_size:
        df, _ = train_test_split(df, test_size=stratify_sample_size, stratify=df[stratify_column], random_state=1)

    # Data scaling
    if scale_type:
        df = scale_data(df, columns=scale_columns, scale_type=scale_type)

    # Convert float columns to int if applicable
    if convert_floats_to_ints:
        df = convert_float_columns_to_int(df)

    # Convert the entire DataFrame to int if needed
    if as_int:
        df = df.astype(int)

    # Clean and save the processed DataFrame
    df.to_csv(os.path.join(data_dir, f"{nickname}.csv"), index=False)

    print(f"Dataset processing complete. Saved processed data to {data_dir}.")
    return df


def main():
    # Example usage: loading the "hitorilabs/iris" dataset from Huggingface
    iris_dataset = download_clean_and_save_dataset("hitorilabs/iris")
    print(iris_dataset.head())

    # Example usage: loading the "Iris" dataset from UCI ML Repository
    iris_dataset = download_clean_and_save_dataset("53", source='uci')
    print(iris_dataset.head())


if __name__ == "__main__":
    main()
