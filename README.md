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
    git clone https://github.com/KingAarya78/Movie-Rating-Predictor.git
    cd Movie-Rating-Predictor
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
    *If `requirements.txt` is not provided, install manually: `pip install pandas numpy scikit-learn matplotlib seaborn joblib scipy`*

4.  **Prepare Your Dataset:**
    *   This project requires a **CSV dataset** of movies.
    *   Place your dataset in the root directory of the project (or update the `DATASET_PATH` variable).
    *   **Crucially, you need to configure the script/notebook to match your dataset's column names.**
    *   Open `movie_rating_predictor.py` (and `movie_rating_predictor.ipynb` if using) and update the following **Configuration Variables** near the top of the file:
        ```python
        DATASET_PATH = 'your_dataset_name.csv' # E.g., 'movies.csv'
        NAME_COL = 'Actual_Name_Column_Title'
        GENRE_COL = 'Actual_Genre_Column_Title'
        DIRECTOR_COL = 'Actual_Director_Column_Title'
        ACTORS_COL = 'Actual_Actor_1_Column_Title' # Or your primary actor column
        RATING_COL = 'Actual_Rating_Column_Title'
        YEAR_COL = 'Actual_Year_Column_Title'
        DURATION_COL = 'Actual_Duration_Column_Title'
        VOTES_COL = 'Actual_Votes_Column_Title'

        # Also update 'use_cols' if your actor columns are named differently than 'Actor 1', 'Actor 2', 'Actor 3'
        use_cols = [NAME_COL, YEAR_COL, DURATION_COL, GENRE_COL, RATING_COL, VOTES_COL, DIRECTOR_COL, ACTORS_COL, 'Actual_Actor_2_Column_Title', 'Actual_Actor_3_Column_Title']
        ```
    *   The script's data cleaning logic (e.g., for 'Year', 'Duration', 'Votes') is designed for common string formats. If your dataset has significantly different formats in these columns, you may need to adjust the corresponding cleaning code in Section 3 of the script/notebook.


## 🚀 How to Run

There are two main ways to interact with this project:

### 1. Using the Python Script (`movie_rating_predictor.py`)

The entire pipeline (data loading, preprocessing, training, evaluation, and saving outputs) can be executed using the Python script.

1.  **Ensure Configuration:** Verify that the `DATASET_PATH` and column name variables in `movie_rating_predictor.py` are correctly set for your dataset.
2.  **Execute from Terminal:**
    Navigate to the project's root directory in your terminal and run:
    ```bash
    python movie_rating_predictor.py
    ```
3.  **Outputs:**
    *   Console output will show the progress, model evaluation metrics, and file save confirmations.
    *   An `output/` directory will be created in the project root.
    *   Inside `output/`, you will find:
        *   `Actual_vs_Predicted_Ratings_*.png`: A plot visualizing the best model's performance.
        *   `model_comparison_results.csv`: A CSV file with the performance metrics of all trained models.
        *   `best_movie_rating_model_*.joblib`: The saved best-performing machine learning model.
        *   `tfidf_vectorizers.joblib`: The saved TF-IDF vectorizers.

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
