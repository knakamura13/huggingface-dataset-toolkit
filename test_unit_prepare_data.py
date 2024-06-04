import unittest
import pandas as pd
from io import BytesIO
from unittest.mock import patch, mock_open, MagicMock
from prepare_data import encode_df_columns, download_huggingface_dataset, download_uci_dataset, \
    cast_floats_to_ints, resolve_missing_data, apply_data_scaling, balance_class_distribution, \
    process_and_store_dataset, sample_with_stratification, write_df_to_csv, remove_df_columns, load_local_dataset


class TestPrepareData(unittest.TestCase):

    @patch('prepare_data.load_dataset')
    def test_download_huggingface_dataset(self, mock_load_dataset):
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

    @patch('prepare_data.zipfile.ZipFile')
    @patch('prepare_data.Path.exists')
    def test_load_local_dataset_zip(self, mock_exists, mock_zipfile):
        # Mock the Path.exists() to return True
        mock_exists.return_value = True

        # Create a mock zip file containing a single CSV file
        mock_zip = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip
        mock_zip.namelist.return_value = ['test.csv']

        # Create a mock CSV content
        csv_content = b'sepal_length,sepal_width,petal_length,petal_width,species\n5.1,3.5,1.4,0.2,setosa\n'
        mock_zip.open.return_value.__enter__.return_value = BytesIO(csv_content)

        # Call the function
        df = load_local_dataset('data/test.zip')

        # Assertions
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], 1)
        self.assertIn('sepal_length', df.columns)
        self.assertEqual(df.iloc[0]['species'], 'setosa')

    @patch('prepare_data.zipfile.ZipFile')
    @patch('prepare_data.Path.exists')
    def test_load_local_dataset_zip_multiple_files(self, mock_exists, mock_zipfile):
        # Mock the Path.exists() to return True
        mock_exists.return_value = True

        # Create a mock zip file containing multiple files
        mock_zip = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip
        mock_zip.namelist.return_value = ['test.csv', 'other.csv']

        # Create a mock CSV content for the first file
        csv_content = b'sepal_length,sepal_width,petal_length,petal_width,species\n5.1,3.5,1.4,0.2,setosa\n'
        mock_zip.open.return_value.__enter__.return_value = BytesIO(csv_content)

        # Call the function
        df = load_local_dataset('data/test.zip')

        # Assertions
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], 1)
        self.assertIn('sepal_length', df.columns)
        self.assertEqual(df.iloc[0]['species'], 'setosa')

    @patch('prepare_data.zipfile.ZipFile')
    @patch('prepare_data.Path.exists')
    def test_load_local_dataset_zip_no_csv(self, mock_exists, mock_zipfile):
        # Mock the Path.exists() to return True
        mock_exists.return_value = True

        # Create a mock zip file containing no CSV files
        mock_zip = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip
        mock_zip.namelist.return_value = []

        # Call the function and expect an error
        with self.assertRaises(ValueError):
            load_local_dataset('data/test.zip')

    @patch('prepare_data.zipfile.ZipFile')
    @patch('prepare_data.Path.exists')
    def test_load_local_dataset_zip_invalid_csv(self, mock_exists, mock_zipfile):
        mock_exists.return_value = True
        mock_zip = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip
        mock_zip.namelist.return_value = ['test.csv']

        invalid_csv_content = BytesIO(b"")
        mock_zip.open.return_value.__enter__.return_value = invalid_csv_content

        with self.assertRaises(pd.errors.EmptyDataError):
            load_local_dataset('data/test.zip', verbose=True)

    @patch('prepare_data.Path.exists')
    def test_load_local_dataset_file_not_found(self, mock_exists):
        # Mock the Path.exists() to return False
        mock_exists.return_value = False

        # Call the function and expect a FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            load_local_dataset('data/non_existent.zip')

    def test_encode_df_columns(self):
        df = pd.DataFrame({
            'species': ['setosa', 'versicolor', 'setosa', 'virginica', 'versicolor'],
            'sepal_length': [5.1, 6.0, 5.4, 5.6, 5.7]
        })
        encoded_df = encode_df_columns(df, ['species'], use_one_hot_encoding=True)
        self.assertIn('species_setosa', encoded_df.columns)
        self.assertIn('species_versicolor', encoded_df.columns)
        self.assertIn('species_virginica', encoded_df.columns)
        self.assertNotIn('species', encoded_df.columns)

    def test_encode_df_columns_non_existent_column(self):
        df = pd.DataFrame({
            'species': ['setosa', 'versicolor', 'setosa', 'virginica', 'versicolor'],
            'sepal_length': [5.1, 6.0, 5.4, 5.6, 5.7]
        })
        with self.assertRaises(ValueError):
            encode_df_columns(df, ['non_existent_column'], use_one_hot_encoding=True)

    def test_cast_floats_to_ints(self):
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.5, 5.0, 6.1],
            'C': [7.0, 8.2, 9.3]
        })
        result = cast_floats_to_ints(df)
        self.assertEqual(result['A'].dtype, 'int64')
        self.assertEqual(result['B'].dtype, 'float64')
        self.assertEqual(result['C'].dtype, 'float64')

    def test_resolve_missing_data(self):
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, None, 6],
            'C': [None, 8, 9]
        })
        result_drop = resolve_missing_data(df, strategy='drop')
        self.assertEqual(result_drop.shape[0], 1)
        self.assertEqual(result_drop.shape[1], 3)

    def test_apply_data_scaling_standardize(self):
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': ['a', 'b', 'c']
        })
        result = apply_data_scaling(df, columns=['A', 'B'], scale_type='standardize')
        self.assertAlmostEqual(result['A'].mean(), 0, places=6)
        self.assertAlmostEqual(result['B'].mean(), 0, places=6)

    def test_balance_class_distribution(self):
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'target': [0, 0, 1, 1]
        })
        balanced_df = balance_class_distribution(df, 'target')
        self.assertEqual(balanced_df['target'].value_counts()[0], balanced_df['target'].value_counts()[1])

    def test_sample_with_stratification(self):
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
            'feature2': [8, 7, 6, 5, 4, 3, 2, 1],
            'target': [0, 0, 0, 0, 1, 1, 1, 1]
        })
        stratify_column = 'target'
        stratify_sample_size = 0.5
        sampled_df = sample_with_stratification(df, stratify_column, stratify_sample_size)
        self.assertEqual(sampled_df['target'].value_counts()[0], sampled_df['target'].value_counts()[1])

    def test_process_and_store_dataset_integration(self):
        mock_data = pd.DataFrame({
            'sepal_length': [5.1, 4.9, 4.7, 4.6, 5.0],
            'sepal_width': [3.5, 3.0, 3.2, 3.1, 3.6],
            'petal_length': [1.4, 1.4, 1.3, 1.5, 1.4],
            'petal_width': [0.2, 0.2, 0.2, 0.2, 0.2],
            'species': ['setosa', 'setosa', 'setosa', 'setosa', 'setosa']
        })
        df = process_and_store_dataset("hitorilabs/iris", source='huggingface')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('sepal_length', df.columns)
        self.assertIn('species', df.columns)

    def test_write_df_to_csv(self):
        df = pd.DataFrame({
            'sepal_length': [5.1, 4.9],
            'species': ['setosa', 'setosa']
        })
        with patch('prepare_data.Path.mkdir') as mock_mkdir:
            with patch('prepare_data.pd.DataFrame.to_csv') as mock_to_csv:
                write_df_to_csv(df, "data/test.csv")
                mock_mkdir.assert_called_once()
                mock_to_csv.assert_called_once_with("data/test.csv", index=False)

    def test_remove_df_columns(self):
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })
        result = remove_df_columns(df, ['B'])
        self.assertNotIn('B', result.columns)


if __name__ == '__main__':
    unittest.main()
