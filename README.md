# Huggingface + UCI Dataset Toolkit

## Overview

This repository contains a Python script that automates the process of downloading, cleaning, and saving datasets from multiple sources including Huggingface and the UCI Machine Learning Repository. 

The toolkit supports various data preprocessing functionalities such as encoding, normalization, standardization, handling missing data, and automatic class balancing.

- Browse Huggingface datasets here: [Huggingface Datasets](https://huggingface.co/datasets)

- Browse UCI datasets here: [UCI Datasets](https://archive.ics.uci.edu/datasets)

## Features

- **Multiple Data Sources**: Download datasets from Huggingface or UCI ML Repository.
- **Data Encoding**: Supports both one-hot encoding and ordinal encoding of categorical variables.
- **Data Scaling**: Includes options for standardizing or normalizing numerical features.
- **Missing Data Handling**: Provides strategies such as drop, fill, and imputation to manage missing values.
- **Data Transformation**: Automatic conversion of float columns to integers where possible.
- **Stratified Sampling**: Reduce dataset size while maintaining the distribution of target variables.
- **Class Balancing**: Automatically balance classes using random oversampling.

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
    scale_type='normalize'
)

# Download and preprocess dataset from UCI
uci_dataset = download_clean_and_save_dataset(
    name="53",  # ID for the Iris dataset
    source='uci',
    scale_type='standardize',
    missing_data_strategy='fill',
    missing_data_fill_value=0,
    auto_balance=True,
    verbose=True
)
```

## Contributions

Feel free to contribute to this project by submitting pull requests or suggesting new enhancements through the issues tab.

## License

This project is open-source and available under the MIT License.