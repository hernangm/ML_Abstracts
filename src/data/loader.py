import pandas as pd

import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

def load_excel(file_path, columns):
    """
    Load data from an Excel file into a pandas DataFrame.

    Parameters:
    - file_path (str): Path to the Excel file.

    Returns:
    - pd.DataFrame: Loaded data.
    """
    try:
        df = pd.read_excel(file_path, sheet_name=0, usecols=columns)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
    except Exception as e:
        print(f"Error loading Excel file: {e}")