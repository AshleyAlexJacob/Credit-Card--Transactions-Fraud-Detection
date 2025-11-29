import pandas as pd

def convert_dicitonary_to_df(dictionary_data:dict)->pd.DataFrame:
    """Converts a dictionary to a pandas DataFrame.
    Args:
        json_data (dict): Input dictionary data.
    Returns:
        pd.DataFrame: Converted pandas DataFrame.
    """
    df = pd.DataFrame([dictionary_data])
    return df