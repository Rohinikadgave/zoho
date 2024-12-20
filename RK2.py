import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from RK1 import A
import re
import warnings
warnings.filterwarnings("ignore")

class B:
    def __init__(self, text_columns, numerical_columns, datetime_columns, categorical_columns):
        self.text_columns = text_columns
        self.numerical_columns = numerical_columns
        self.datetime_columns = datetime_columns
        self.categorical_columns = categorical_columns

    def remove_null_values(self, data):
        """
        Handles missing values in the dataset.

        """

        # 'critics_consensus' columns have 8329 missing values hence droping this column
        if 'critics_consensus' in data.columns:
            data = data.drop('critics_consensus', axis=1)

        for col in self.text_columns:
            default = 'Unknown' if col != 'movie_info' else 'No summary available'
            data[col] = data[col].fillna(default)

        for col in self.datetime_columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')
            data[col].fillna(data[col].median(), inplace=True)

        for col in self.numerical_columns:
            data[col] = data[col].fillna(data[col].median())

        for col in self.categorical_columns:
            data[col] = data[col].fillna(data[col].mode()[0])

        return data



    def preprocess_and_encode(self, data):
        """
        Preprocesses the dataset and encodes features.

        """
        data = self.remove_null_values(data)

        def preprocess_text(text):
            if isinstance(text, str):
                text = re.sub(r'[^\w\s]', '', text)
                text = text.lower()
            return text

        # Preprocess and combine text columns
        processed_text = {
            col: data[col].apply(lambda x: preprocess_text(str(x))) for col in self.text_columns
        }
        combined_text = pd.DataFrame(processed_text).apply(
            lambda row: ' '.join([str(item) for item in row]), axis=1
        )

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words='english', max_features=200)
        data_tfidf = pd.DataFrame(
            vectorizer.fit_transform(combined_text).toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        # Encode datetime columns
        for col in self.datetime_columns:
            prefix = col.replace('_date', '')
            data_tfidf[f'{prefix}_year'] = data[col].dt.year
            data_tfidf[f'{prefix}_month'] = data[col].dt.month
            data_tfidf[f'{prefix}_day'] = data[col].dt.day

            # Cyclic encoding
            data_tfidf[f'{prefix}_month_sin'] = np.sin(2 * np.pi * data_tfidf[f'{prefix}_month'] / 12)
            data_tfidf[f'{prefix}_month_cos'] = np.cos(2 * np.pi * data_tfidf[f'{prefix}_month'] / 12)
            data_tfidf[f'{prefix}_day_sin'] = np.sin(2 * np.pi * data_tfidf[f'{prefix}_day'] / 31)
            data_tfidf[f'{prefix}_day_cos'] = np.cos(2 * np.pi * data_tfidf[f'{prefix}_day'] / 31)

        # Encode categorical columns
        for col in self.categorical_columns:
            if col == 'studio_name':
                data_tfidf[col] = data[col].map(data[col].value_counts())
            else:
                dummies = pd.get_dummies(data[col], prefix=col, dtype=int)
                data_tfidf = pd.concat([data_tfidf, dummies], axis=1)

        # Add numerical columns
        data_tfidf = pd.concat([data_tfidf, data[self.numerical_columns]], axis=1)

        return data_tfidf


if __name__ == "__main__":
    text_columns = ['movie_title', 'movie_info', 'genre', 'directors', 'writers', 'cast']
    numerical_columns = ['runtime_in_minutes', 'tomatometer_rating', 'tomatometer_count', 'audience_rating']
    datetime_columns = ['in_theaters_date', 'on_streaming_date']
    categorical_columns = ['rating', 'studio_name']

    obj1 = A("Rotten_Tomatoes_Movies3.xls")
    try:
        raw_data = obj1.read_data()
        obj2 = B(text_columns, numerical_columns, datetime_columns, categorical_columns)
        encoded_data = obj2.preprocess_and_encode(raw_data)
        print("Encoded DataFrame:")
        print(encoded_data.head())
    except Exception as e:
        print(f"Error: {e}")
