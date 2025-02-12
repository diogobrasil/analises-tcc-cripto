import os
import pandas as pd
from typing import List

class DataLoader:
    @staticmethod
    def load_single_crypto(file_path: str) -> pd.DataFrame:
        """Load data for a single cryptocurrency."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        data = pd.read_csv(file_path)
        print(f"Data loaded for file: {file_path}")
        return data

    @staticmethod
    def load_multiple_cryptos(folder_path: str) -> pd.DataFrame:
        """Load data for multiple cryptocurrencies from a folder."""
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        all_data = []
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith('.csv'):
                data = pd.read_csv(file_path)
                data['B3'] = os.path.splitext(file_name)[0]  
                all_data.append(data)
                print(f"Data loaded for file: {file_name}")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data

if __name__ == "__main__":
    # Example usage

    base_path = "../../datasets/b3_dados/"

    # Assuming data is manually placed in the dataset folder
    raw = os.path.join(base_path, "raw")

    # Load single crypto data
    single_crypto_data = DataLoader.load_single_crypto(os.path.join(raw, "VALE3.csv"))
    print(single_crypto_data.head())

    # Load multiple cryptos data from the folder
    # all_cryptos_data = DataLoader.load_multiple_cryptos(raw)
    # print(all_cryptos_data.head())
