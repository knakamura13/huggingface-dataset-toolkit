import os
import string
import pandas as pd
from pathlib import Path
from datasets import load_dataset, Split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def encode_columns(df, column_names, use_one_hot_encoding=True):
    """
    Apply encoding to a list of column names on a DataFrame, dropping the original columns.

    :param df: Pandas DataFrame to one-hot encode
    :param column_names: List of column names to be encoded
    :param use_one_hot_encoding: Boolean to use one-hot encoding as opposed to ordinal
    :return: Pandas DataFrame with original columns dropped and encoded columns added
    """
    if use_one_hot_encoding:
        for name in column_names:
            enc = OneHotEncoder(handle_unknown="ignore")
            new_cols = enc.fit_transform(df[[name]]).toarray()
            new_col_names = [
                f"{name}_{n}".lower().replace("group ", "")
                for n in enc.categories_[0].tolist()
            ]
            new_cols_df = pd.DataFrame(new_cols, columns=new_col_names, dtype=int)
            df = pd.concat([new_cols_df, df], axis=1)
        df.drop(columns=column_names, inplace=True)
    else:
        # Initialize the OrdinalEncoder
        enc = OrdinalEncoder()

        # Perform the encoding and replace the original columns
        df_encoded = df.copy()
        df_encoded[column_names] = enc.fit_transform(df[column_names])
        df = df_encoded

    return df


def download_dataset(name, variant=None, split=Split.ALL):
    """
    Download a Huggingface dataset and convert it to a DataFrame.

    :param name: Name of the dataset to download from Huggingface
    :param variant: Optional, variant of the dataset
    :param split: Which split of the dataset to load. Default is all data
    :return: Pandas DataFrame representation of the dataset
    """
    return load_dataset(
        name, variant, split=split, download_mode="reuse_cache_if_exists"
    ).to_pandas()


def download_clean_and_save_dataset(
    name,
    variant=None,
    nickname=None,
    split=Split.ALL,
    cols_to_drop=None,
    cols_to_encode=None,
    use_one_hot_encoding=True,
    data_dir="data",
    as_int=False,
) -> pd.DataFrame:
    """
    Download a dataset, optionally clean it, and save it as a CSV file.

    :param name: Name of the dataset to download
    :param variant: Optional, variant of the dataset
    :param nickname: Optional, custom name for saving the dataset
    :param split: Which split of the dataset to load. Default is all data
    :param cols_to_drop: List of column names to drop
    :param cols_to_encode: List of column names to encode
    :param use_one_hot_encoding: Boolean to use one-hot encoding as opposed to ordinal
    :param data_dir: Directory to save the datasets. Default is "data"
    :param as_int: Boolean to indicate if DataFrame should be converted to int
    :return: Cleaned and processed Pandas DataFrame
    """
    # Create a nickname for naming the csv files
    if not nickname:
        nickname = name.split("/")[1] if "/" in name else name
    nickname = nickname.replace("-dataset", "").replace("-", "_").lower()

    # Download the dataset from Huggingface
    df = download_dataset(name, variant, split)

    # Save the unprocessed DataFrame
    Path(data_dir).mkdir(exist_ok=True)
    df.to_csv(os.path.join(data_dir, f"{nickname}_original.csv"), index=False)

    # Drop specific columns
    if cols_to_drop:
        try:
            df.drop(columns=cols_to_drop, inplace=True)
        except Exception as e:
            print(e)

    # One-hot encode specific columns
    if cols_to_encode:
        try:
            df = encode_columns(
                df, cols_to_encode, use_one_hot_encoding=use_one_hot_encoding
            )
        except Exception as e:
            print(e)

    # Convert the DataFrame type to int
    if as_int:
        df = df.astype(int)

    # Rename the last column to 'target'
    df.rename(columns={df.columns[-1]: "target"}, inplace=True)

    # Replace booleans with ints
    df.replace({"True": 1, "False": 0, True: 1, False: 0}, inplace=True)

    # Remove extra punctuation and white spaces
    df.replace("[{}]".format(string.punctuation), "", regex=True, inplace=True)
    df.replace("  ", " ", regex=True, inplace=True)

    # Save the cleaned DataFrame
    df.to_csv(os.path.join(data_dir, f"{nickname}.csv"), index=False)

    return df


def main():
    # Example usage: load the "hitorilabs/iris" dataset
    iris_dataset = download_clean_and_save_dataset("hitorilabs/iris")
    print(iris_dataset.head())


if __name__ == "__main__":
    main()
