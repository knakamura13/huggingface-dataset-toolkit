import os
import time
import unittest
import requests
import pandas as pd
from prepare_data import download_huggingface_dataset, download_uci_dataset


def is_uci_available():
    try:
        response = requests.get('https://archive.ics.uci.edu/ml/index.php', timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False


def download_with_retry(download_func, *args, retries=0, delay=0):
    for attempt in range(retries):
        try:
            return download_func(*args)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            else:
                raise e


class TestPrepareData(unittest.TestCase):

    def test_download_huggingface_dataset(self):
        df = download_huggingface_dataset('hitorilabs/iris')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn('sepal_length', df.columns)

    def test_download_and_process_uci_dataset(self):
        if not is_uci_available():
            self.skipTest("UCI dataset repository is currently offline.")
        df = download_with_retry(download_uci_dataset, 53)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn('sepal_length', df.columns)

    def tearDown(self):
        # Clean up created files to avoid clutter and ensure test isolation
        output_path = 'data/processed_iris.csv'
        if os.path.exists(output_path):
            os.remove(output_path)


if __name__ == '__main__':
    unittest.main()
