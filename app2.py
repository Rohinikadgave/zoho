import pickle
import pandas as pd
from flask import Flask, jsonify, request, render_template

from RK2 import B


app = Flask(__name__)

# Load the trained model once
model = pickle.load(open('Audience_rating_model.pkl', 'rb'))

# Initialize objects from other files
obj2 = B(
    text_columns=['movie_title', 'movie_info', 'genre', 'directors', 'writers', 'cast'],
    numerical_columns=['runtime_in_minutes', 'tomatometer_rating', 'tomatometer_count'],
    datetime_columns=['in_theaters_date', 'on_streaming_date'],
    categorical_columns=['rating', 'studio_name']
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def prediction():
    try:
        # Extract form data from the request
        data = request.form
        movie_title = data.get("movie_title")
        movie_info = data.get("movie_info")
        critics_consensus = data.get("critics_consensus")
        rating = data.get("rating")
        genre = data.get("genre")
        directors = data.get("directors")
        writers = data.get("writers")
        cast = data.get("cast")
        in_theaters_date = data.get("in_theaters_date")
        on_streaming_date = data.get("on_streaming_date")
        runtime_in_minutes = int(data.get("runtime_in_minutes", 0))
        studio_name = data.get("studio_name")
        tomatometer_status = int(data.get("tomatometer_status", 0))
        tomatometer_rating = int(data.get("tomatometer_rating", 0))
        tomatometer_count = int(data.get("tomatometer_count", 0))

        # Create input data as a DataFrame (ensure column alignment with training data)
        input_data = pd.DataFrame([[
            movie_title, movie_info, critics_consensus, rating, genre, directors, writers, cast,
            in_theaters_date, on_streaming_date, runtime_in_minutes, studio_name,
            tomatometer_status, tomatometer_rating, tomatometer_count
        ]], columns=[
            'movie_title', 'movie_info', 'critics_consensus', 'rating', 'genre', 'directors', 
            'writers', 'cast', 'in_theaters_date', 'on_streaming_date', 'runtime_in_minutes', 
            'studio_name', 'tomatometer_status', 'tomatometer_rating', 'tomatometer_count'
        ])

        # Preprocess and encode input data
        preprocessed_data = obj2.preprocess_and_encode()

        # Use the trained model to make predictions
        prediction = model.predict(preprocessed_data)

        return render_template('index.html', prediction_text=f"Predicted Audience Rating: {prediction[0]}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
