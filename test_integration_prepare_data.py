import unittest
import pandas as pd
from prepare_data import download_huggingface_dataset, download_uci_dataset


class TestIntegration(unittest.TestCase):
    def test_download_huggingface_dataset(self):
        try:
            df = download_huggingface_dataset('hitorilabs/iris')
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
            self.assertIn('sepal_length', df.columns)
        except Exception as e:
            self.fail(f"Failed to download dataset from Huggingface: {str(e)}")

    def test_download_uci_dataset(self):
        try:
            df = download_uci_dataset(53)  # 53 is the ID for the Iris dataset
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
            self.assertIn('sepal_length', df.columns)
        except Exception as e:
            self.fail(f"Failed to download dataset from UCI: {str(e)}")


if __name__ == '__main__':
    unittest.main()
