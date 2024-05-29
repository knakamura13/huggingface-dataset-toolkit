import unittest
import pandas as pd
from prepare_data import download_huggingface_dataset, download_uci_dataset, process_data


class TestIntegration(unittest.TestCase):
    def download_and_process_dataset(self, download_func, *args):
        try:
            df = download_func(*args)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
            self.assertIn('sepal_length', df.columns)

            # Process the dataset
            processed_df = process_data(df, scale_type='standardize', verbose=True)
            self.assertIsInstance(processed_df, pd.DataFrame)
            self.assertIn('sepal_length', processed_df.columns)
        except Exception as e:
            self.fail(f"Failed to download and process dataset: {str(e)}")

    def test_download_and_process_huggingface_dataset(self):
        self.download_and_process_dataset(download_huggingface_dataset, 'hitorilabs/iris')

    def test_download_and_process_uci_dataset(self):
        self.download_and_process_dataset(download_uci_dataset, 53)  # 53 is the ID for the Iris dataset


if __name__ == '__main__':
    unittest.main()
