"""
Dataset: 2AIRTC - Amharic Adhoc Information Retrieval Test Collection

Description:
The 2AIRTC dataset supports Amharic information retrieval research. It includes:
- A document set of 12,586 items from sources like Amharic news agencies, social media, and historical texts, spanning topics such as business, sports, and technology.
- 240 topics, each with Amharic and English titles, descriptions, and narratives representing real-world information needs.
- Relevance judgments (QRELs) marking at least 10 relevant documents per topic, structured in TREC format.

Citation:
Please cite if using this collection:
Yeshambel, T., Mothe, J., & Assabie, Y. (2020). 2AIRTC: The Amharic Adhoc Information Retrieval Test Collection.
In CLEF 2020 Conference Proceedings, Springer.

BibTeX:
@inproceedings{yeshambel20202airtc,
  title={2AIRTC: The Amharic Adhoc Information Retrieval Test Collection},
  author={Yeshambel, Tilahun and Mothe, Josiane and Assabie, Yaregal},
  booktitle={CLEF 2020},
  pages={55--66},
  year={2020},
  organization={Springer}
}

Download link:
https://www.irit.fr/AmharicResources/airtc-the-amharic-adhoc-information-retrieval-test-collection/
"""

import os
import requests
import zipfile
import tarfile
from tqdm import tqdm

DATA_URL = 'https://www.irit.fr/AmharicResources/wp-content/uploads/2021/03/Archive-collection-Zip.zip'
DATA_DIR = 'data/raw/2airtc_dataset'
FILE_NAME = 'Archive-collection-Zip.zip'
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)


def download_dataset(url=DATA_URL, file_path=FILE_PATH):
    """Download the dataset from the specified URL with a progress bar."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # User-Agent header to mimic a real browser request
    # Explanation:
    # - Some servers block automated requests, returning a 403 Forbidden error if the request doesn't appear to come from a web browser.
    # - This 'User-Agent' string represents a common web browser (Chrome on Windows), tricking the server into treating our request as a regular browser request.
    # - Without this, the server may deny access to the dataset, preventing the download.

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers, stream=True)
    
    if response.status_code == 403:
        print("Access to the resource is forbidden. Check the URL or permissions.")
        return

    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192  # 8KB

    print(f"Downloading dataset from {url}...")
    with open(file_path, 'wb') as file, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=FILE_NAME
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=block_size):
            file.write(chunk)
            progress_bar.update(len(chunk))
    print(f"Dataset downloaded and saved to {file_path}")



def unzip_dataset(file_path=FILE_PATH, extract_dir=DATA_DIR):
    """Unzip the dataset if it hasn't been extracted."""
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted ZIP archive to {extract_dir}")
    elif file_path.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(file_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_dir)
        print(f"Extracted TAR.GZ archive to {extract_dir}")
    else:
        print("Unsupported archive format. Please use a .zip or .tar.gz file.")


if __name__ == "__main__":
    # download_dataset()
    unzip_dataset()