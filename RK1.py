import pandas as pd

class A:
    """
    A class to handle reading data from an Excel file.

    Attributes:
        data (str): The path to the Excel file.
    """
    def __init__(self, data):
        self.data = data

    def read_data(self):
        """
        Reads data from an Excel file.
        """
        try:
            data = pd.read_excel(self.data)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.data}")
        except ValueError:
            raise ValueError(f"Invalid Excel file format: {self.data}")


if __name__ == "__main__":
    data_loader = A(r"Rotten_Tomatoes_Movies3.xls")
    try:
        data = data_loader.read_data()
        print(data)
    except Exception as e:
        print(f"Error: {e}")
