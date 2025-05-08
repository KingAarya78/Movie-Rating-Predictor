# Movie Rating Prediction 🎬📊

This project aims to predict movie ratings based on various features like genre, director, actors, year, duration, and user votes. It employs machine learning regression techniques to analyze historical movie data and develop a model that accurately estimates potential ratings.

## ✨ Features

*   **Data Cleaning & Preprocessing:** Handles missing values, cleans data types (e.g., extracting numbers from 'Year' and 'Duration' strings), and prepares data for modeling.
*   **Feature Engineering:**
    *   Utilizes **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert textual features (Genre, Director, Actors) into numerical representations.
    *   Combines multiple actor columns into a single feature.
    *   Includes numerical features like Year, Duration, and Votes.
*   **Model Training & Evaluation:**
    *   Trains several regression models:
        *   Linear Regression
        *   Ridge Regression
        *   Random Forest Regressor
        *   Gradient Boosting Regressor
    *   Evaluates models using standard metrics: MAE (Mean Absolute Error), MSE (Mean Squared Error), RMSE (Root Mean Squared Error), and R² Score.
*   **Model Persistence:** Saves the best-performing model and the associated TF-IDF vectorizers using `joblib` for future predictions.
*   **Example Prediction:** Demonstrates how to load the saved model and make predictions on new, unseen movie data.
*   **Structured Code:** Provided as both a runnable Python script (`movie_rating_predictor.py`) and an explanatory Jupyter Notebook (`movie_rating_predictor.ipynb`).

## 🛠️ Technologies Used

*   **Python 3.x**
*   **Core Libraries:**
    *   `pandas` - Data manipulation and analysis
    *   `numpy` - Numerical operations
    *   `scikit-learn` - Machine learning (TF-IDF, models, metrics, train-test split)
    *   `matplotlib` & `seaborn` - Data visualization
    *   `joblib` - Model saving and loading
    *   `scipy` - For sparse matrix operations (used with TF-IDF)
*   **Development Environment:** Jupyter Notebook, Standard Python Interpreter

## ⚙️ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Make sure your `requirements.txt` file is up-to-date and included in the repository.)*

4.  **Dataset:**
    *   This project expects a CSV file named `movies.csv` in the root directory.
    *   **You need to provide your own dataset.** The script is configured to look for `movies.csv`.
    *   Ensure your dataset contains columns similar to those defined in the script configuration (e.g., `Name`, `Year`, `Duration`, `Genre`, `Rating`, `Votes`, `Director`, `Actor 1`, `Actor 2`, `Actor 3`).
    *   Update the `DATASET_PATH = 'movies.csv'` variable in both `movie_rating_predictor.py` and `movie_rating_predictor.ipynb` if your file has a different name or path.

## 🚀 How to Run

There are two main ways to interact with this project:

### 1. Using the Python Script (`movie_rating_predictor.py`)

This script will perform all steps from data loading to model evaluation and saving.

1.  **Verify Dataset Path:** Ensure the `DATASET_PATH` variable at the top of `movie_rating_predictor.py` points to your dataset file.
2.  **Run the script from your terminal:**
    ```bash
    python movie_rating_predictor.py
    ```
    The script will print outputs to the console, save the best model (`best_movie_rating_model_*.joblib`), vectorizers (`tfidf_vectorizers.joblib`), and a plot (`Actual_vs_Predicted_Ratings_*.png`).

### 2. Using the Jupyter Notebook (`movie_rating_predictor.ipynb`)

This notebook provides a more interactive, step-by-step walkthrough of the project with inline outputs and explanations.

1.  **Start Jupyter Notebook or Jupyter Lab:**
    ```bash
    jupyter notebook
    # OR
    jupyter lab
    ```
2.  Navigate to and open `movie_rating_predictor.ipynb`.
3.  Run the cells sequentially to see the process and results.

## 📊 Dataset Description

The model is trained on a dataset of movies with the following key features (actual column names might vary, adapt the script's configuration variables):

*   `Name`: Title of the movie.
*   `Year`: Release year (e.g., "(2019)").
*   `Duration`: Runtime in minutes (e.g., "109 min").
*   `Genre`:Comma-separated list of genres (e.g., "Action, Drama, Sci-Fi").
*   `Rating`: The target variable; numerical movie rating (e.g., 1.0 to 10.0).
*   `Votes`: Number of votes/reviews (e.g., "150,000").
*   `Director`: Name of the director(s).
*   `Actor 1`, `Actor 2`, `Actor 3`: Names of key actors.

## 📝 Project Workflow

1.  **Load Data:** Read the movie dataset from a CSV file.
2.  **Exploratory Data Analysis (EDA) & Preprocessing:**
    *   Inspect data structure, types, and missing values.
    *   Handle missing values (dropping or imputing).
    *   Clean and convert features (e.g., extracting numerical year/duration, cleaning votes string).
    *   Combine actor columns.
3.  **Feature Engineering:**
    *   Apply TF-IDF vectorization to text features (`Genre`, `Director`, `Actors_Combined`).
    *   Prepare numerical features (`Year`, `Duration`, `Votes`).
4.  **Train-Test Split:** Divide the data into training and testing sets.
5.  **Model Training:** Train various regression models on the training data.
6.  **Model Evaluation:** Evaluate model performance on the test set using MAE, MSE, RMSE, and R² score. Identify the best model.
7.  **Visualization:** Plot actual vs. predicted ratings for the best model.
8.  **Model Saving:** Save the best performing model and the fitted TF-IDF vectorizers for later use.
9.  **Example Prediction:** Demonstrate predicting a rating for a new movie instance using the saved model.

## 📈 Results

The models were evaluated, and the **Gradient Boosting Regressor** typically performed best with the sample dataset structure. For instance, one run yielded:

*   **Best Model:** Gradient Boosting
*   **MAE:** ~0.79
*   **RMSE:** ~1.05
*   **R² Score:** ~0.41

This indicates the model can explain about 41% of the variance in movie ratings based on the provided features. An example prediction for a new movie might look like: `Predicted Rating: 5.19`.

*(Note: Actual results will vary based on your specific dataset and any modifications to the code.)*

Visualizations comparing actual vs. predicted ratings are generated by the script and can be found in the notebook.

## 🔮 Future Improvements

*   **Hyperparameter Tuning:** Implement GridSearchCV or RandomizedSearchCV to find optimal parameters for the models, especially Gradient Boosting or Random Forest.
*   **Advanced Models:** Experiment with XGBoost, LightGBM, or CatBoost.
*   **Feature Scaling:** Apply `StandardScaler` to numerical features, which can benefit linear models and sometimes tree-based models.
*   **Pipeline Implementation:** Use `sklearn.pipeline.Pipeline` and `ColumnTransformer` for a more streamlined and robust preprocessing and modeling workflow.
*   **Advanced Text Features:** Explore word embeddings (Word2Vec, GloVe, FastText) or sentence transformers instead of TF-IDF.
*   **Feature Importance Analysis:** Investigate which features contribute most to the predictions.
*   **Enhanced EDA:** Deeper analysis of feature distributions, correlations, and their relationship with ratings.
*   **Error Analysis:** Analyze where the model makes the largest errors to understand its limitations.

## 📜 License

This project is licensed under the Apache 2.0 License - see the [LICENSE.md](LICENSE.md) file for details.
