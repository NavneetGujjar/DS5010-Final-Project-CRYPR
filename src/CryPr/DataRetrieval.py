import pandas as pd

def retrieve_file_data(file_path: str) -> pd.DataFrame: #load data
    """
    Read data from CSV, XLSX, or JSON file.

    :param file_path: str, path to the input file
    :return: pd.DataFrame containing the loaded data
    """
    file_extension = file_path.split('.')[-1]

    if file_extension == 'csv':
        data = pd.read_csv(file_path)
    elif file_extension == 'xlsx':
        data = pd.read_excel(file_path, engine='openpyxl')
    elif file_extension == 'json':
        data = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}. Supported formats are CSV, XLSX, and JSON.")

    return data
