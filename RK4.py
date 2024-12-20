from RK1 import A
from RK2 import B
from RK3 import C
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
import pickle
import warnings
import numpy as np

warnings.filterwarnings("ignore")

class D:
    def __init__(self, text_columns, numerical_columns, datetime_columns, categorical_columns, target_column):
        self.text_columns = text_columns
        self.numerical_columns = numerical_columns
        self.datetime_columns = datetime_columns
        self.categorical_columns = categorical_columns
        self.target_column = target_column

        # Initialize obj3 of C class, passing required arguments
        self.obj3 = C(self.text_columns, self.numerical_columns, self.datetime_columns, self.categorical_columns)

    def gradient_boosting_model(self):
        """
        Train Gradient Boosting Regressor with hyperparameter tuning using RandomizedSearchCV.
        """
        # Retrieve train and test data
        X_train, X_test, y_train, y_test = self.obj3.train_test_split()

        # Define hyperparameter grid
        param_grid_gb = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.05]
        }

        # Perform randomized search
        grid_search_gb = RandomizedSearchCV(
            GradientBoostingRegressor(),
            param_distributions=param_grid_gb,
            n_iter=10, cv=5, scoring='r2', n_jobs=-1, random_state=42
        )
        grid_search_gb.fit(X_train, y_train)

        # Extract best parameters
        best_params_gb = grid_search_gb.best_params_
        print(f"Best Parameters: {best_params_gb}")

        # Train the best model
        best_gb = GradientBoostingRegressor(**best_params_gb)
        best_gb.fit(X_train, y_train)

        # Make predictions
        y_pred_gb = best_gb.predict(X_test)
        return best_gb, y_pred_gb

    def model_evaluation(self):
        """
        Evaluate the trained Gradient Boosting model using MSE, RMSE, and R2 score.
        """
        best_gb, y_pred_gb = self.gradient_boosting_model()
        _, X_test, _, y_test = self.obj3.train_test_split()

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred_gb)
        rmse = np.sqrt(mse)
        r2_score_ = r2_score(y_test, y_pred_gb)

        print(f"Evaluation Metrics - MSE: {mse}, RMSE: {rmse}, R2 Score: {r2_score_}")
        return mse, rmse, r2_score_

    def save_model(self):
        """
        Save the trained model to a pickle file.
        """
        best_gb, _ = self.gradient_boosting_model()
        with open('Audience_rating_model.pkl', 'wb') as file:
            pickle.dump(best_gb, file)
        print("Model saved successfully as 'Audience_rating_model.pkl'.")

if __name__ == "__main__":
    # Column definitions
    text_columns = ['movie_title', 'movie_info', 'genre', 'directors', 'writers', 'cast']
    numerical_columns = ['runtime_in_minutes', 'tomatometer_rating', 'tomatometer_count', 'audience_rating']
    datetime_columns = ['in_theaters_date', 'on_streaming_date']
    categorical_columns = ['rating', 'studio_name']
    target_column = "audience_rating"

    # Instantiate and run the model
    model = D(text_columns, numerical_columns, datetime_columns, categorical_columns, target_column)

    # Save the trained model
    model.save_model()

    # Evaluate the model
    mse, rmse, r2_score_ = model.model_evaluation()

    print("\nModel Evaluation:")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2_score_}")
