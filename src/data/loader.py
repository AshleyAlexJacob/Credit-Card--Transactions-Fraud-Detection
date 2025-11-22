import pandas as pd
from tqdm import tqdm

class DataLoader:
    def __init__(self, file_path:str):
        self.file_path:str = file_path
        self.data:pd.Dataframe = None
    
    def load_data(self, file_path:str=None) -> pd.DataFrame:
        """Load data from a CSV file into a pandas DataFrame."""
        try: 
            path = file_path if file_path else self.file_path

            chunk_size = 10000
            total_lines = sum(1 for line in open(path, 'r', encoding='utf-8', errors='ignore')) - 1 # Subtract 1 for header
            chunks = []
            print("Starting to read CSV in chunks...")
            with tqdm(total=total_lines, desc="Reading CSV", unit="lines") as pbar:
                for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
                    chunks.append(chunk)
                    pbar.update(len(chunk)) # Update the progress bar with the number of rows in the chunk
            print("Data loading complete.")
            self.data = pd.concat(chunks, ignore_index=True)
            return self.data
        except FileNotFoundError:
            print(f"File not found: {path}")
            exit(1)
        except Exception as e:
            print(f"Error loading data from {path}: {e}")
            exit(1)
    