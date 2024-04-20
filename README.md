# Huggingface Dataset Toolkit

## Overview

This repository contains a Python script that automates the process of downloading, cleaning, and saving datasets from multiple sources including Huggingface and the UCI Machine Learning Repository. 
The toolkit now supports various data preprocessing functionalities such as encoding, normalization, standardization, and handling missing data, making it suitable for preparing datasets for machine learning models.

Browse Huggingface datasets here: https://huggingface.co/datasets

Browse UCI datasets here: https://archive.ics.uci.edu/datasets

## Features

- **Multiple Data Sources**: Download datasets from Huggingface or UCI ML Repository.
- **Data Encoding**: Supports both one-hot encoding and ordinal encoding of categorical variables.
- **Data Scaling**: Includes options for standardizing or normalizing numerical features.
- **Missing Data Handling**: Provides strategies such as drop, fill, and imputation to manage missing values.
- **Data Transformation**: Automatic conversion of float columns to integers where possible.
- **Stratified Sampling**: Reduce dataset size while maintaining the distribution of target variables.
- **Informative Logging**: Print statements have been added to inform the user about the progress and status of data processing.

## Installation

Install the necessary dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

Here is how you can use the script to download and preprocess a dataset:

```python
from prepare_data import download_clean_and_save_dataset

# Download and preprocess dataset from Huggingface
dataset = download_clean_and_save_dataset(
    name="hitorilabs/iris",
    source='huggingface',
    scale_type='normalize',
    missing_data_strategy='impute'
)

# Download and preprocess dataset from UCI
uci_dataset = download_clean_and_save_dataset(
    name="53",  # Dataset ID for UCI
    source='uci',
    scale_type='standardize',
    missing_data_strategy='fill',
    missing_data_fill_value=0
)
```

## Contributions

Feel free to contribute to this project by submitting pull requests or suggesting new features or enhancements through the issues tab.

## License

This project is open-source and available under the MIT License.
