import pandas as pd

def sub_sample_data(csv_file_path:str, sample_size:int, random_state:int=42) -> pd.DataFrame:
    """
    Reads a CSV file into a DataFrame and returns a random sample of the specified size.

    Parameters:
    csv_file_path (str): The path to the CSV file.
    sample_size (int): The number of samples to return.
    random_state (int): The seed for the random number generator for reproducibility.

    Returns:
    pd.DataFrame: A DataFrame containing the random sample.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    is_fraud_samples = df[df['is_fraud'] == 1]
    non_fraud_samples = df[df['is_fraud'] == 0]

    print(f"Total fraud samples available: {len(is_fraud_samples)}")
    print(f"Total non-fraud samples available: {len(non_fraud_samples)}")

    sub_sample_data = pd.concat([
        is_fraud_samples.sample(n=sample_size // 2, random_state=random_state),
        non_fraud_samples.sample(n=sample_size // 2, random_state=random_state)
    ])

    print(f"Sub-sampled data contains {len(sub_sample_data)}")
    # Return a random sample of the specified size
    return sub_sample_data


