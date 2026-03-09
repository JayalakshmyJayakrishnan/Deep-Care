import os
import requests
import zipfile
import io

def download_file(url, target_path):
    print(f"Downloading from {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            
            with open(target_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        percent = (downloaded / total_size) * 100
                        print(f"Progress: {percent:.1f}%", end='\r')
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"Error downloading: {e}")
        return False

def setup_galar_dataset():
    # Helper for the Galar dataset from Figshare
    # Based on URL provided: https://plus.figshare.com/articles/dataset/Galar_-_a_large_multi-label_video_capsule_endoscopy_dataset/25304616?file=44724651
    # The file ID is 44724651. 
    # The direct download link for figshare files is usually https://figshare.com/ndownloader/files/ID
    
    # Galar Metadata and Splits Only (Avoids the 300GB frames)
    # 44755283 = Galar_labels_and_metadata.7z
    # 45307744 = Galar_splits.zip
    
    METADATA_URL = "https://figshare.com/ndownloader/files/44755283"
    SPLITS_URL = "https://figshare.com/ndownloader/files/45307744"
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATASET_DIR = os.path.join(BASE_DIR, 'datasets', 'galar')
    
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    print("--- Setting up Galar Clinical Reference Subsystem ---")
    print(f"Target Directory: {DATASET_DIR}")
    
    # 1. Download structural files
    download_file(SPLITS_URL, os.path.join(DATASET_DIR, "Galar_splits.zip"))
    download_file(METADATA_URL, os.path.join(DATASET_DIR, "Galar_labels_and_metadata.7z"))
    
    # 2. Extract splits (standard zip)
    splits_zip = os.path.join(DATASET_DIR, "Galar_splits.zip")
    if os.path.exists(splits_zip):
        print("Extracting splits...")
        try:
            with zipfile.ZipFile(splits_zip, 'r') as zip_ref:
                zip_ref.extractall(DATASET_DIR)
            print("Extraction of splits complete.")
        except Exception as e:
            print(f"Error extracting splits: {e}")
            
    # 3. Inform about metadata (7z)
    metadata_7z = os.path.join(DATASET_DIR, "Galar_labels_and_metadata.7z")
    if os.path.exists(metadata_7z):
        print(f"Metadata downloaded to: {metadata_7z}")
        print("Note: Please use a tool like 7-Zip manually if you need to inspect the metadata contents, or install py7zr.")
        
    print("\n==============================================")
    print("Galar Dataset Reference Architecture Ready.")
    print("Hospital IT Note: Set the GALAR_MOUNT_PATH environment")
    print("variable to the location of the fully uncompressed 300GB")
    print("frame dataset to enable the clinical reference engine.")
    print("==============================================")

if __name__ == "__main__":
    setup_galar_dataset()
