from src.utils import sub_sample_data
from sklearn.utils import shuffle

if __name__ == "__main__":
    # Example usage
    csv_file_path = "data/processed/data_resampled_v1.csv"
    sample_size = 300000
    sampled_data = sub_sample_data(csv_file_path, sample_size)
    print(sampled_data.head())
    sampled_data = shuffle(sampled_data)
    print(sampled_data.head())
    
    sampled_data.to_csv("data/processed/data_subsampled_v1.csv", index=False)