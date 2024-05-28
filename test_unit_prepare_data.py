import unittest
from unittest.mock import patch
import pandas as pd
from prepare_data import encode_columns, download_huggingface_dataset, download_uci_dataset, \
    convert_float_columns_to_int, \
    handle_missing_data, scale_data, auto_balance_dataset, download_clean_and_save_dataset, apply_stratified_sampling, \
    save_dataframe, drop_columns, process_data


class TestPrepareData(unittest.TestCase):

    @patch('prepare_data.load_dataset')
    def test_download_huggingface_dataset(self, mock_load_dataset):
        # Mocking the Huggingface dataset download
        mock_load_dataset.return_value.to_pandas.return_value = pd.DataFrame({
            'sepal_length': [5.1, 4.9, 4.7, 4.6, 5.0],
            'sepal_width': [3.5, 3.0, 3.2, 3.1, 3.6],
            'petal_length': [1.4, 1.4, 1.3, 1.5, 1.4],
            'petal_width': [0.2, 0.2, 0.2, 0.2, 0.2],
            'species': ['setosa', 'setosa', 'setosa', 'setosa', 'setosa']
        })
        df = download_huggingface_dataset('hitorilabs/iris')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], 5)
        self.assertIn('sepal_length', df.columns)

    @patch('prepare_data.fetch_ucirepo')
    def test_download_uci_dataset(self, mock_fetch_ucirepo):
        # Mocking the UCI dataset download
        mock_fetch_ucirepo.return_value.data.features = pd.DataFrame({
            'sepal_length': [5.1, 4.9, 4.7, 4.6, 5.0],
            'sepal_width': [3.5, 3.0, 3.2, 3.1, 3.6],
            'petal_length': [1.4, 1.4, 1.3, 1.5, 1.4],
            'petal_width': [0.2, 0.2, 0.2, 0.2, 0.2]
        })
        mock_fetch_ucirepo.return_value.data.targets = pd.DataFrame({
            'species': ['setosa', 'setosa', 'setosa', 'setosa', 'setosa']
        })

        df = download_uci_dataset(53)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], 5)
        self.assertIn('sepal_length', df.columns)

    def test_encode_columns(self):
        df = pd.DataFrame({
            'species': ['setosa', 'versicolor', 'setosa', 'virginica', 'versicolor'],
            'sepal_length': [5.1, 6.0, 5.4, 5.6, 5.7]
        })
        encoded_df = encode_columns(df, ['species'], use_one_hot_encoding=True)
        self.assertIn('species_setosa', encoded_df.columns)
        self.assertIn('species_versicolor', encoded_df.columns)
        self.assertIn('species_virginica', encoded_df.columns)
        self.assertNotIn('species', encoded_df.columns)

    def test_convert_float_columns_to_int(self):
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.5, 5.0, 6.1],
            'C': [7.0, 8.2, 9.3]
        })
        result = convert_float_columns_to_int(df)
        self.assertEqual(result['A'].dtype, 'int64')
        self.assertEqual(result['B'].dtype, 'float64')
        self.assertEqual(result['C'].dtype, 'float64')

    def test_handle_missing_data(self):
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, None, 6],
            'C': [None, 8, 9]
        })
        # Test drop strategy
        result = handle_missing_data(df, strategy='drop')
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], 3)

        # Test fill strategy with fill_value
        df_with_nans = pd.DataFrame({
            'A': [1, 2, None],
            'B': [4, None, 6],
            'C': [None, None, 9]
        })
        result = handle_missing_data(df_with_nans, strategy='fill', fill_value=0)
        self.assertEqual(result.isnull().sum().sum(), 0)
        self.assertEqual(result.iloc[0]['C'], 0)

        # Test fill strategy without fill_value (mean/mode filling)
        result = handle_missing_data(df_with_nans, strategy='fill')
        self.assertEqual(result.isnull().sum().sum(), 0)

        # Test impute strategy
        result = handle_missing_data(df_with_nans, strategy='impute')
        self.assertEqual(result.isnull().sum().sum(), 0)

    def test_scale_data(self):
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        result = scale_data(df, columns=['A'], scale_type='standardize')
        self.assertAlmostEqual(result['A'].mean(), 0, places=6)

    def test_auto_balance_dataset(self):
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'target': [0, 0, 1, 1]
        })
        balanced_df = auto_balance_dataset(df, 'target')
        self.assertEqual(balanced_df['target'].value_counts()[0], balanced_df['target'].value_counts()[1])

    def test_apply_stratified_sampling(self):
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
            'feature2': [8, 7, 6, 5, 4, 3, 2, 1],
            'target': [0, 0, 0, 0, 1, 1, 1, 1]
        })
        stratify_column = 'target'
        stratify_sample_size = 0.5
        sampled_df = apply_stratified_sampling(df, stratify_column, stratify_sample_size)
        self.assertEqual(sampled_df['target'].value_counts()[0], sampled_df['target'].value_counts()[1])

    @patch('prepare_data.download_huggingface_dataset')
    @patch('prepare_data.download_uci_dataset')
    @patch('prepare_data.save_dataframe')
    def test_download_clean_and_save_dataset(self, mock_save_dataframe, mock_download_uci_dataset,
                                             mock_download_huggingface_dataset):
        mock_data = pd.DataFrame({
            'sepal_length': [5.1, 4.9, 4.7, 4.6, 5.0],
            'sepal_width': [3.5, 3.0, 3.2, 3.1, 3.6],
            'petal_length': [1.4, 1.4, 1.3, 1.5, 1.4],
            'petal_width': [0.2, 0.2, 0.2, 0.2, 0.2],
            'species': ['setosa', 'setosa', 'setosa', 'setosa', 'setosa']
        })

        # Mocking Huggingface dataset download
        mock_download_huggingface_dataset.return_value = mock_data.copy()
        df = download_clean_and_save_dataset("hitorilabs/iris", source='huggingface')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('sepal_length', df.columns)
        self.assertIn('species', df.columns)
        mock_save_dataframe.assert_called()

        # Mocking UCI dataset download
        mock_download_uci_dataset.return_value = mock_data.copy()
        df = download_clean_and_save_dataset("53", source='uci')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('sepal_length', df.columns)
        self.assertIn('species', df.columns)
        mock_save_dataframe.assert_called()

    def test_save_dataframe(self):
        df = pd.DataFrame({
            'sepal_length': [5.1, 4.9],
            'species': ['setosa', 'setosa']
        })
        with patch('prepare_data.Path.mkdir') as mock_mkdir:
            with patch('prepare_data.pd.DataFrame.to_csv') as mock_to_csv:
                save_dataframe(df, "data/test.csv")
                mock_mkdir.assert_called_once()
                mock_to_csv.assert_called_once_with("data/test.csv", index=False)

    def test_drop_columns(self):
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })
        result = drop_columns(df, ['B'])
        self.assertNotIn('B', result.columns)

    def test_process_data(self):
        df = pd.DataFrame({
            'species': ['setosa', 'versicolor', 'setosa', 'virginica', 'versicolor'],
            'sepal_length': [5.1, 6.0, 5.4, 5.6, 5.7],
            'sepal_width': [3.5, 3.0, 3.2, 3.1, 3.6],
            'petal_length': [1.4, 1.4, 1.3, 1.5, 1.4],
            'petal_width': [0.2, 0.2, 0.2, 0.2, 0.2]
        })
        processed_df = process_data(df, cols_to_encode=['species'], use_one_hot_encoding=True,
                                    missing_data_strategy='fill',
                                    scale_type='standardize', scale_columns=['sepal_length'],
                                    convert_floats_to_ints=True, as_int=False, auto_balance=False,
                                    stratify_column='species')
        self.assertIn('species_setosa', processed_df.columns)
        self.assertAlmostEqual(processed_df['sepal_length'].mean(), 0, places=6)
        # Ensure 'sepal_width' is converted to int if applicable
        if processed_df['sepal_width'].apply(lambda x: x.is_integer()).all():
            self.assertEqual(processed_df['sepal_width'].dtype, 'int64')
        else:
            self.assertEqual(processed_df['sepal_width'].dtype, 'float64')


if __name__ == '__main__':
    unittest.main()
