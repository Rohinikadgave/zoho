import pickle
import pandas as pd
import streamlit as st
from RK2 import B

# Load the trained model
model = pickle.load(open('Audience_rating_model.pkl', 'rb'))

# Initialize objects from other files
obj2 = B(
    text_columns=['movie_title', 'movie_info', 'genre', 'directors', 'writers', 'cast'],
    numerical_columns=['runtime_in_minutes', 'tomatometer_rating', 'tomatometer_count'],
    datetime_columns=['in_theaters_date', 'on_streaming_date'],
    categorical_columns=['rating', 'studio_name']
)

def main():
    st.title("Audience Rating Prediction")
    st.write("Provide movie details below to predict the audience rating.")

    # Input fields
    movie_title = st.text_input("Movie Title")
    movie_info = st.text_input("Movie Info")
    critics_consensus = st.text_input("Critics Consensus")
    rating = st.selectbox("Rating", ["G", "PG", "PG-13", "R", "NC-17", "Not Rated"])
    genre = st.text_input("Genre")
    directors = st.text_input("Directors")
    writers = st.text_input("Writers")
    cast = st.text_input("Cast")
    in_theaters_date = st.date_input("In Theaters Date")
    on_streaming_date = st.date_input("On Streaming Date")
    runtime_in_minutes = st.number_input("Runtime (in minutes)", min_value=0, step=1)
    studio_name = st.text_input("Studio Name")
    tomatometer_status = st.selectbox("Tomatometer Status", ["0", "1"])
    tomatometer_rating = st.number_input("Tomatometer Rating", min_value=0, step=1)
    tomatometer_count = st.number_input("Tomatometer Count", min_value=0, step=1)

    # Predict button
    if st.button("Predict Audience Rating"):
        try:
            # Prepare input data
            input_data = pd.DataFrame([[
                movie_title, movie_info, critics_consensus, rating, genre, directors, writers, cast,
                in_theaters_date, on_streaming_date, runtime_in_minutes, studio_name,
                int(tomatometer_status), int(tomatometer_rating), int(tomatometer_count)
            ]], columns=[
                'movie_title', 'movie_info', 'critics_consensus', 'rating', 'genre', 'directors', 
                'writers', 'cast', 'in_theaters_date', 'on_streaming_date', 'runtime_in_minutes', 
                'studio_name', 'tomatometer_status', 'tomatometer_rating', 'tomatometer_count'
            ])

            # Preprocess and encode input data
            preprocessed_data = obj2.preprocess_and_encode(input_data)

            # Predict using the model
            prediction = model.predict(preprocessed_data)

            # Display the prediction
            st.success(f"Predicted Audience Rating: {prediction[0]}")

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
