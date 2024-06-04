# Huggingface (and UCI or local) Dataset Toolkit

## Overview

This repository contains a Python script that automates the process of downloading, cleaning, and saving datasets from multiple sources including Huggingface, the UCI Machine Learning Repository, and local dataset files.

The toolkit supports various data preprocessing functionalities such as encoding, normalization, standardization, handling missing data, automatic class balancing, and image resizing.

- Browse Huggingface datasets here: [Huggingface Datasets](https://huggingface.co/datasets)
- Browse UCI datasets here: [UCI Datasets](https://archive.ics.uci.edu/ml/datasets.php)

## Features

- **Multiple Data Sources**: Download datasets from Huggingface, UCI ML Repository, or use local dataset files.
- **Local Dataset Compatibility**: Load datasets directly from local storage, supporting various file formats such as CSV, Excel, Parquet, and more.
- **Multiple Data Types**: Handle both tabular and image datasets, including complex operations such as image resizing and flattening.
- **Data Encoding**: Apply one-hot encoding and ordinal encoding to categorical variables.
- **Data Scaling**: Options for standardizing or normalizing numerical features.
- **Missing Data Handling**: Strategies include dropping, filling, or imputing missing values.
- **Data Transformation**: Convert float columns to integers where applicable.
- **Stratified Sampling**: Reduce dataset size while preserving the distribution of target variables.
- **Class Balancing**: Automatically balance classes using random oversampling.
- **Image Resizing**: Resize images to specified target widths, maintaining aspect ratio and converting to tabular format.

## Installation

Install the necessary dependencies using:

```bash
pip install -r requirements.txt
