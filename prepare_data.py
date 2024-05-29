import os
import sys
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from skimage.transform import resize
from datasets import load_dataset, Split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo, DatasetNotFoundError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Helper functions


def resize_image(flat_img, target_width, verbose=False):
    """
    Resizes a flattened image to a specified target width while maintaining the aspect ratio.

    Parameters:
    flat_img (numpy.ndarray): The flattened image array.
    target_width (int): The target width to resize the image to. If the target width is larger than the current size or less than 1, resizing is skipped.
    verbose (bool): If True, logs additional information during the resizing process.

    Returns:
    numpy.ndarray: The resized and flattened image array.
    """
    original_size = flat_img.size
    current_size = int(np.sqrt(original_size))

    if target_width < 1:
        # Skipping resize if target width is smaller than 1
        return flat_img

    # If the image is not square, reshape it to a square of the largest dimension
    height = int(np.floor(np.sqrt(original_size)))
    width = int(np.ceil(np.sqrt(original_size)))
    if height * width != original_size:
        padded_img = np.pad(flat_img, (0, height * width - original_size), mode='constant')
        image_reshaped = padded_img.reshape((height, width))
    else:
        image_reshaped = flat_img.reshape((height, width))

    # Resize while keeping the aspect ratio, then crop or pad to the target size
    aspect_ratio = image_reshaped.shape[1] / image_reshaped.shape[0]
    if aspect_ratio > 1:
        # Width is greater than height
        new_height = target_width
        new_width = int(target_width * aspect_ratio)
    else:
        # Height is greater than width or they are equal
        new_width = target_width
        new_height = int(target_width / aspect_ratio)

    # Ensure that neither new_width nor new_height is zero
    new_width = max(1, new_width)
    new_height = max(1, new_height)

    resized_image = resize(image_reshaped, (new_height, new_width), anti_aliasing=True)

    # Crop or pad to make it square
    start_height = (new_height - target_width) // 2
    start_width = (new_width - target_width) // 2
    if new_height > target_width:
        resized_image = resized_image[start_height:start_height + target_width, :]
    if new_width > target_width:
        resized_image = resized_image[:, start_width:start_width + target_width]

    if resized_image.shape != (target_width, target_width):
        # Pad if the size is not exact
        padded_image = np.zeros((target_width, target_width))
        padded_image[:resized_image.shape[0], :resized_image.shape[1]] = resized_image
        resized_image = padded_image

    return resized_image.flatten()


def convert_images_to_tabular(dataset, target_image_width=None, verbose=False):
    """
    Converts a dataset containing images into a tabular format.

    Parameters:
    dataset (Dataset): The dataset containing images and other data.
    target_image_width (int, optional): The target width to resize the images to. If None, images are not resized.
    verbose (bool, optional): If True, logs additional information during the conversion process.

    Returns:
    DataFrame: A pandas DataFrame where each image is represented as a series of pixel values, and other data is included as columns.
    """
    if verbose:
        logger.info(f"Received target_image_width: {target_image_width}")

    data = []
    for index, item in tqdm(enumerate(dataset), desc="Converting images to tabular format", total=len(dataset)):
        row = {}
        for col, value in item.items():
            if col == 'image':
                image_array = np.array(value.convert('L')).flatten()  # Convert to grayscale and flatten
                if target_image_width:
                    image_array = resize_image(image_array, target_image_width, verbose)
                row.update({f'image_{i}': v for i, v in enumerate(image_array)})
            else:
                row[col] = value
        data.append(row)

    df = pd.DataFrame(data)
    if verbose:
        logger.info("Image conversion complete.")
    return df


def encode_columns(df, column_names, use_one_hot_encoding=True, verbose=False):
    """
    Encodes specified columns in a DataFrame using either One-Hot Encoding or Ordinal Encoding.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the columns to be encoded.
    column_names (list of str): The list of column names to be encoded.
    use_one_hot_encoding (bool, optional): If True, applies One-Hot Encoding; otherwise, applies Ordinal Encoding. Default is True.
    verbose (bool, optional): If True, logs additional information during the encoding process. Default is False.

    Returns:
    pandas.DataFrame: The DataFrame with the specified columns encoded.
    """
    if verbose:
        logger.info("Encoding columns...")
    missing_columns = [col for col in column_names if col not in df.columns]
    if missing_columns:
        raise ValueError(f"The following columns do not exist in the DataFrame: {missing_columns}")

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


def convert_float_columns_to_int(df, verbose=False):
    """
    Converts float columns in a DataFrame to integers where possible.

    This function checks each float column in the DataFrame to see if all values in the column are integers.
    If so, it converts the column's data type to integer.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the columns to be converted.
    verbose (bool, optional): If True, logs additional information during the conversion process. Default is False.

    Returns:
    pandas.DataFrame: The DataFrame with float columns converted to integers where applicable.
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
    Handles missing data in a DataFrame using the specified strategy.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing missing data to be handled.
    strategy (str, optional): The strategy to handle missing data. Options are:
        - "drop": Drops rows with missing values.
        - "fill": Fills missing values with a specified value or the mean/mode of the column.
        - "impute": Imputes missing values using the mean of the column. Default is "drop".
    fill_value (any, optional): The value to fill missing data with when strategy is "fill". If None, uses mean for numeric columns and mode for non-numeric columns. Default is None.
    verbose (bool, optional): If True, logs additional information during the process. Default is False.

    Returns:
    pandas.DataFrame: The DataFrame with missing data handled according to the specified strategy.
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
    Scales numeric data in a DataFrame using the specified scaling method.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to be scaled.
    columns (list of str, optional): The list of column names to be scaled. If None, all numeric columns are scaled. Default is None.
    scale_type (str, optional): The type of scaling to apply. Options are 'standardize' for StandardScaler and 'normalize' for MinMaxScaler. Default is 'standardize'.
    verbose (bool, optional): If True, logs additional information during the scaling process. Default is False.

    Returns:
    pandas.DataFrame: The DataFrame with the specified columns scaled.
    """
    if verbose:
        logger.info(f"Scaling data using {scale_type} scaling...")

    # Choose the appropriate scaler based on the scale_type parameter
    scaler = StandardScaler() if scale_type == 'standardize' else MinMaxScaler()

    # Determine which columns to scale
    if columns:
        numeric_columns = df[columns].select_dtypes(include=['float64', 'int']).columns
    else:
        numeric_columns = df.select_dtypes(include=['float64', 'int']).columns

    # Apply the scaler to the selected columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    if verbose:
        logger.info("Data scaling complete.")

    return df


def auto_balance_dataset(df, target_column, verbose=False):
    """
    Balances the classes in a DataFrame using random oversampling.

    This function identifies the majority and minority classes in the specified target column and
    oversamples the minority class to match the number of samples in the majority class.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to be balanced.
    target_column (str): The name of the column containing the class labels to balance.
    verbose (bool, optional): If True, logs additional information during the balancing process. Default is False.

    Returns:
    pandas.DataFrame: The DataFrame with balanced classes.
    """
    if verbose:
        logger.info("Balancing classes using random oversampling...")

    # Identify the majority and minority classes
    majority_class = df[target_column].value_counts().idxmax()
    minority_class = df[target_column].value_counts().idxmin()

    # Separate the majority and minority class samples
    majority_df = df[df[target_column] == majority_class]
    minority_df = df[df[target_column] == minority_class]

    # Calculate the number of samples needed to balance the classes
    n_samples_to_duplicate = len(majority_df) - len(minority_df)

    if n_samples_to_duplicate <= 0:
        if verbose:
            logger.info("Classes are already balanced.")
        return df

    # Oversample the minority class
    minority_upsampled = minority_df.sample(n=n_samples_to_duplicate, replace=True, random_state=1)

    # Combine the majority class, original minority class, and upsampled minority class
    balanced_df = pd.concat([majority_df, minority_df, minority_upsampled])

    if verbose:
        logger.info("Classes balanced.")

    # Shuffle the DataFrame and reset the index
    return balanced_df.sample(frac=1, random_state=1).reset_index(drop=True)


def apply_stratified_sampling(df, stratify_column, stratify_sample_size, verbose=False):
    """
    Applies stratified sampling to a DataFrame based on a specified column.

    This function performs stratified sampling on the DataFrame, ensuring that the sample
    maintains the same distribution of values in the specified column as the original DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to sample from.
    stratify_column (str): The name of the column to use for stratification.
    stratify_sample_size (float): The proportion of the dataset to include in the sample.
    verbose (bool, optional): If True, logs additional information during the sampling process. Default is False.

    Returns:
    pandas.DataFrame: The stratified sample of the original DataFrame.

    Raises:
    ValueError: If the specified stratify_column does not exist in the DataFrame.
    """
    if not stratify_column or not stratify_sample_size:
        return df
    if stratify_column not in df.columns:
        raise ValueError(f"The specified stratify_column '{stratify_column}' does not exist in the DataFrame.")
    sampled_df, _ = train_test_split(df, test_size=stratify_sample_size, stratify=df[stratify_column], random_state=1)
    if verbose:
        logger.info("Stratified sampling complete.")
    return sampled_df


def drop_columns(df, cols_to_drop, verbose=False):
    """
    Drops specified columns from a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame from which columns are to be dropped.
    cols_to_drop (list of str): The list of column names to drop from the DataFrame.
    verbose (bool, optional): If True, logs additional information during the process. Default is False.

    Returns:
    pandas.DataFrame: The DataFrame with the specified columns dropped.

    Raises:
    ValueError: If all columns are dropped, resulting in an empty DataFrame.
    """
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        if df.empty:
            raise ValueError("All columns have been dropped; no data remains for processing.")
    if verbose:
        logger.info("Specified columns dropped.")
    return df


def get_dataset_nickname(dataset_name: str, nickname=None, verbose=False):
    """
    Generates a standardized nickname for a dataset based on its name.

    This function takes a dataset name and optionally a nickname, and returns a standardized nickname.
    If no nickname is provided, it generates one by processing the dataset name.

    Parameters:
    dataset_name (str): The name of the dataset.
    nickname (str, optional): An optional nickname for the dataset. If not provided, a nickname is generated from the dataset name.
    verbose (bool, optional): If True, logs additional information during the nickname generation process. Default is False.

    Returns:
    str: The generated or provided nickname, standardized to lowercase and with certain characters replaced.
    """
    if not nickname:
        if "/" in dataset_name:
            nickname = dataset_name.split("/")[1]
        else:
            nickname = dataset_name
    nickname = nickname.lower().replace("-dataset", "").replace("-", "_")
    if verbose:
        logger.info(f"Generated nickname: {nickname}")
    return nickname


def save_dataframe(df, file_path, verbose=False):
    """
    Saves a pandas DataFrame to a CSV file.

    This function saves the provided DataFrame to the specified file path in CSV format.
    It ensures that the directory structure exists before saving the file.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be saved.
    file_path (str): The path where the CSV file will be saved.
    verbose (bool, optional): If True, logs additional information during the saving process. Default is False.

    Returns:
    None
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
    Processes a DataFrame by encoding specified columns, handling missing data, scaling data,
    converting float columns to integers, and optionally balancing the dataset.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be processed.
    cols_to_encode (list of str, optional): The list of column names to be encoded. Default is None.
    use_one_hot_encoding (bool, optional): If True, applies One-Hot Encoding; otherwise, applies Ordinal Encoding. Default is True.
    missing_data_strategy (str, optional): The strategy to handle missing data. Options are "drop", "fill", and "impute". Default is "drop".
    missing_data_fill_value (any, optional): The value to fill missing data with when strategy is "fill". Default is None.
    scale_type (str, optional): The type of scaling to apply. Options are 'standardize' for StandardScaler and 'normalize' for MinMaxScaler. Default is None.
    scale_columns (list of str, optional): The list of column names to be scaled. If None, all numeric columns are scaled. Default is None.
    convert_floats_to_ints (bool, optional): If True, converts float columns to integers where possible. Default is True.
    as_int (bool, optional): If True, converts the entire DataFrame to integers. Default is False.
    auto_balance (bool, optional): If True, balances the classes in the DataFrame using random oversampling. Default is False.
    stratify_column (str, optional): The name of the column to use for stratification when balancing the dataset. Default is None.
    verbose (bool, optional): If True, logs additional information during the processing. Default is False.

    Returns:
    pandas.DataFrame: The processed DataFrame.

    Raises:
    ValueError: If one or more columns specified for encoding do not exist in the DataFrame.
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


# Download functions

def download_huggingface_dataset(name, variant=None, split=Split.ALL, verbose=False, target_image_width=None):
    """
    Downloads a dataset from the Huggingface repository.

    Parameters:
    name (str): The name or identifier of the dataset to download.
    variant (str, optional): The variant of the dataset to download. Default is None.
    split (str, optional): The dataset split to download (e.g., 'train', 'test', 'validation'). Default is Split.ALL.
    verbose (bool, optional): If True, logs additional information during the download process. Default is False.
    target_image_width (int, optional): The target width to resize images to. If None, images are not resized. Default is None.

    Returns:
    pandas.DataFrame: The downloaded dataset as a pandas DataFrame.
    """
    if verbose:
        logger.info(f"Downloading dataset {name} from Huggingface...")

    # Load the dataset from Huggingface
    dataset = load_dataset(name, variant, split=split, download_mode="reuse_cache_if_exists")

    # Convert images to tabular format if the dataset contains images
    if 'image' in dataset.column_names:
        df = convert_images_to_tabular(dataset, target_image_width=target_image_width, verbose=verbose)
    else:
        df = dataset.to_pandas()

    if verbose:
        logger.info("Download complete.")

    # Standardize column names to lowercase and replace spaces with underscores
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df


def download_uci_dataset(dataset_id, verbose=False):
    """
    Downloads a dataset from the UCI Machine Learning Repository.

    Parameters:
    dataset_id (int): The identifier of the dataset to download from the UCI repository.
    verbose (bool, optional): If True, logs additional information during the download process. Default is False.

    Returns:
    pandas.DataFrame: The downloaded dataset as a pandas DataFrame with standardized column names.

    Raises:
    DatasetNotFoundError: If the dataset with the specified ID is not available for import.
    """
    try:
        if verbose:
            logger.info(f"Downloading dataset {dataset_id} from UCI...")

        # Fetch the dataset from UCI repository using the provided dataset ID
        uci_dataset = fetch_ucirepo(id=dataset_id)

        # Combine features and targets into a single DataFrame
        df = pd.concat([uci_dataset.data.features, uci_dataset.data.targets], axis=1)

        if verbose:
            logger.info("Download complete.")

        # Standardize column names to lowercase and replace spaces with underscores
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        return df

    except DatasetNotFoundError as e:
        logger.error(f"Dataset with ID {dataset_id} is not available for import. "
                     f"Note that the ucimlrepo library does not support image-based datasets.")
        sys.tracebacklimit = 0
        raise e


def download_dataset(name, source, split, variant, verbose=False, target_image_width=None):
    """
    Downloads a dataset from a specified source.

    Parameters:
    name (str): The name or identifier of the dataset to download.
    source (str): The source from which to download the dataset. Supported sources are 'huggingface' and 'uci'.
    split (str): The dataset split to download (e.g., 'train', 'test', 'validation'). Used for Huggingface datasets.
    variant (str): The variant of the dataset to download. Used for Huggingface datasets.
    verbose (bool, optional): If True, logs additional information during the download process. Default is False.
    target_image_width (int, optional): The target width to resize images to. If None, images are not resized. Default is None.

    Returns:
    pandas.DataFrame: The downloaded dataset as a pandas DataFrame.

    Raises:
    ValueError: If an unsupported source is specified.
    """
    if verbose:
        logger.info(f"Downloading dataset {name} from source {source}...")
    if source.lower() == 'huggingface':
        return download_huggingface_dataset(name, variant, split, verbose, target_image_width)
    elif source.lower() == 'uci':
        return download_uci_dataset(int(name), verbose)
    raise ValueError(f"Unsupported source specified: {source}")


# Main function

def download_clean_and_save_dataset(name, source='huggingface', variant=None, split=Split.ALL,
                                    data_dir="data", nickname=None, cols_to_drop=None, cols_to_encode=None,
                                    use_one_hot_encoding=True, missing_data_strategy="drop",
                                    missing_data_fill_value=None, scale_type=None, scale_columns=None,
                                    stratify_column=None, stratify_sample_size=0.1, convert_floats_to_ints=True,
                                    as_int=False, auto_balance=False, target_image_width=None, verbose=False):
    """
    Downloads, processes, and saves a dataset.

    This function downloads a dataset from a specified source, processes it according to various parameters,
    and saves both the original and processed datasets to CSV files.

    Parameters:
    name (str): The name or identifier of the dataset to download.
    source (str, optional): The source from which to download the dataset. Supported sources are 'huggingface' and 'uci'. Default is 'huggingface'.
    variant (str, optional): The variant of the dataset to download. Used for Huggingface datasets. Default is None.
    split (str, optional): The dataset split to download (e.g., 'train', 'test', 'validation'). Used for Huggingface datasets. Default is Split.ALL.
    data_dir (str, optional): The directory where the CSV files will be saved. Default is "data".
    nickname (str, optional): An optional nickname for the dataset. If not provided, a nickname is generated from the dataset name. Default is None.
    cols_to_drop (list of str, optional): The list of column names to drop from the DataFrame. Default is None.
    cols_to_encode (list of str, optional): The list of column names to be encoded. Default is None.
    use_one_hot_encoding (bool, optional): If True, applies One-Hot Encoding; otherwise, applies Ordinal Encoding. Default is True.
    missing_data_strategy (str, optional): The strategy to handle missing data. Options are "drop", "fill", and "impute". Default is "drop".
    missing_data_fill_value (any, optional): The value to fill missing data with when strategy is "fill". Default is None.
    scale_type (str, optional): The type of scaling to apply. Options are 'standardize' for StandardScaler and 'normalize' for MinMaxScaler. Default is None.
    scale_columns (list of str, optional): The list of column names to be scaled. If None, all numeric columns are scaled. Default is None.
    stratify_column (str, optional): The name of the column to use for stratification when balancing the dataset. Default is None.
    stratify_sample_size (float, optional): The proportion of the dataset to include in the sample. Default is 0.1.
    convert_floats_to_ints (bool, optional): If True, converts float columns to integers where possible. Default is True.
    as_int (bool, optional): If True, converts the entire DataFrame to integers. Default is False.
    auto_balance (bool, optional): If True, balances the classes in the DataFrame using random oversampling. Default is False.
    target_image_width (int, optional): The target width to resize images to. If None, images are not resized. Default is None.
    verbose (bool, optional): If True, logs additional information during the process. Default is False.

    Returns:
    pandas.DataFrame: The processed DataFrame.

    Raises:
    ValueError: If one or more columns specified for encoding do not exist in the DataFrame.
    """
    nickname = get_dataset_nickname(dataset_name=name, nickname=nickname, verbose=verbose)

    df = download_dataset(name, source, split, variant, verbose, target_image_width)
    if df is None:
        logger.error(f"Failed to download dataset {name} from source {source}.")
        return None

    save_dataframe(df, os.path.join(data_dir, f"{nickname}_original.csv"), verbose)

    df = drop_columns(df, cols_to_drop, verbose=verbose)
    df = apply_stratified_sampling(df, stratify_column, stratify_sample_size, verbose=verbose)
    df = process_data(df, cols_to_encode, use_one_hot_encoding, missing_data_strategy, missing_data_fill_value,
                      scale_type, scale_columns, convert_floats_to_ints, as_int, auto_balance, stratify_column, verbose)
    save_dataframe(df, os.path.join(data_dir, f"{nickname}.csv"), verbose)

    if verbose:
        logger.info(f"Dataset processing complete.")
        logger.info(f"Original dataset saved to {os.path.join(data_dir, f'{nickname}_original.csv')}")
        logger.info(f"Processed dataset saved to {os.path.join(data_dir, f'{nickname}.csv')}")

    return df
