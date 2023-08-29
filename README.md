# Huggingface Dataset Toolkit

## Overview

This repository contains Python code that automates the process of downloading, cleaning, and saving datasets. 

The code uses Huggingface's `datasets` library to download the datasets and utilizes pandas and scikit-learn for preprocessing tasks such as one-hot encoding.

You can browse Huggingface datasets here: https://huggingface.co/datasets.

## Requirements

- Python 3.7 or higher
- `pandas`
- `scikit-learn`
- Huggingface `datasets`

Install the required packages using pip:

```bash
pip install pandas scikit-learn huggingface datasets
```

## Functions

### one_hot_encode_columns(df, column_names)

This function applies one-hot encoding to specified columns of a DataFrame and drops the original columns.

**Parameters:**
- `df`: DataFrame to which one-hot encoding is applied
- `column_names`: List of column names to be one-hot encoded

**Returns:**
- DataFrame with original columns dropped and one-hot encoded columns added

### download_dataset(name, variant=None, split=Split.ALL)

Downloads a dataset from Huggingface and converts it to a DataFrame.

**Parameters:**
- `name`: Name of the dataset
- `variant`: Optional, variant of the dataset
- `split`: Optional, the split of data you want (default is all data)

**Returns:**
- The dataset represented as a DataFrame

### download_clean_and_save_dataset(...)

Downloads a dataset, optionally cleans it, and saves it as CSV files.

**Parameters:**
- `name`: Name of the dataset to download. 
  - Example: 'hitorilabs/iris' is the "name" of this Iris dataset https://huggingface.co/datasets/hitorilabs/iris
- `variant`: Variant of the dataset to load (applies to some Huggingface datasets)
- `nickname`: Custom name for saving the dataset
- `split`: Which split of the dataset to load. Default is all data (applies to some Huggingface datasets)
- `cols_to_drop`: List of column names to drop
- `cols_to_encode`: List of column names to one-hot encode
- `data_dir`: Directory to save the dataset. Default is "data"
- `as_int`: Boolean to indicate if DataFrame should be converted to int

**Returns:**
- Cleaned and processed DataFrame

## Usage

1. Clone this repository.

2. Install the required packages.

3. Use `download_clean_and_save_dataset` for an all-in-one approach to fetching and cleaning datasets:

    `df = download_clean_and_save_dataset('hitorilabs/iris', cols_to_drop=['unnecessary_column'], cols_to_encode=['categorical_column'], variant=None, nickname=None, split=Split.ALL, data_dir="data", as_int=False)`

## Collaboration

Feel free to open an issue or submit a pull request if you have suggestions for improvement. 

Some examples for improvements that would be nice to have:
- The ability to load datasets from [UCI's ML Repository](https://archive.ics.uci.edu/datasets) or [Kaggle](https://www.kaggle.com/datasets/).
- Automatic one-hot encoding for categorical data.
- Automatic conversion to `int` when all existing `float`s have no decimal values.
- The option to apply stratified sampling to reduce a dataset's size without altering its distribution of target values.
- The option to normalize or standardize numerical features.
- Functionality to handle missing data by either dropping, filling with a specific value, or using techniques like imputation.
- Batch downloading of multiple datasets based on a list of dataset names.
- Any other improvements to the code or the README to improve usability.

## License

This project is open-source and available under the MIT License.
