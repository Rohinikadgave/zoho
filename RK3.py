from sklearn.model_selection import train_test_split
from RK1 import A
from RK2 import B
import re

class C:
    def __init__(self, text_columns, numerical_columns, datetime_columns, categorical_columns):
        # Initialize class with passed column lists
        self.text_columns = text_columns
        self.numerical_columns = numerical_columns
        self.datetime_columns = datetime_columns
        self.categorical_columns = categorical_columns

        # Instantiate the required objects using the passed columns
        self.obj1 = A("Rotten_Tomatoes_Movies3.xls")
        self.obj2 = B(self.text_columns, self.numerical_columns, self.datetime_columns, self.categorical_columns)

    def split_data(self):
        # Read raw data
        data = self.obj1.read_data()

        # Preprocess data using the obj2 instance
        data_tfidf = self.obj2.preprocess_and_encode(data)  # Pass `data` as an argument
        
        # Separate features and target variable
        X = data_tfidf.drop("audience_rating", axis=1)
        y = data_tfidf["audience_rating"]

        return X, y

    def train_test_split(self):
        X, y = self.split_data()
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    text_columns = ['movie_title', 'movie_info', 'genre', 'directors', 'writers', 'cast']
    numerical_columns = ['runtime_in_minutes', 'tomatometer_rating', 'tomatometer_count', 'audience_rating']
    datetime_columns = ['in_theaters_date', 'on_streaming_date']
    categorical_columns = ['rating', 'studio_name']
    
    # Perform train-test split
    splitter = C(text_columns, numerical_columns, datetime_columns, categorical_columns)
    X_train, X_test, y_train, y_test = splitter.train_test_split()
    
    print("Train features:")
    print(X_train.head())
    print("Test features:")
    print(X_test.head())
    print("Train target:")
    print(y_train.head())
    print("Test target:")
    print(y_test.head())
