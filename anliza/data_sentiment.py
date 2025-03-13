import pandas as pd
import os
import requests

class DataLoader:
    def __init__(self, urls, save_path):
        self.urls = urls
        self.save_path = save_path

    def download_data(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        for url in self.urls:
            # Extract file name from URL
            filename = url.split('/')[-1]
            file_path = os.path.join(self.save_path, filename)

            # Download the file if not already downloaded
            if not os.path.exists(file_path):
                print(f"Downloading {filename} from {url}...")
                response = requests.get(url)
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"{filename} downloaded successfully to {file_path}")
            else:
                print(f"{filename} already exists at {file_path}")


def main():
    urls = [
        'https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv',
        'https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv',
        'https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv'
    ]

    data_path = 'data/full_dataset/'

    data_loader = DataLoader(urls, data_path)
    data_loader.download_data()

    goemotions_1 = pd.read_csv(os.path.join(data_path, 'goemotions_1.csv'))
    goemotions_2 = pd.read_csv(os.path.join(data_path, 'goemotions_2.csv'))
    goemotions_3 = pd.read_csv(os.path.join(data_path, 'goemotions_3.csv'))

    combined_df = pd.concat([goemotions_1, goemotions_2, goemotions_3], ignore_index=True)

    combined_df['created_utc'] = pd.to_datetime(combined_df['created_utc'], unit='s', errors='coerce')

    combined_df.to_csv(os.path.join(data_path, 'goemotions_combined.csv'), index=False)
    print("the merged file was successfully saved ti the path:", os.path.join(data_path, 'goemotions_combined.csv'))

