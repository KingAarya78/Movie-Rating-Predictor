# %% [markdown]
# # Movie Rating Prediction Project
#
# **Goal:** Predict movie ratings based on features like Genre, Director, and Actors using regression techniques.

# %% [markdown]
# ## 1. Import Libraries
# We need libraries for data manipulation, visualization, preprocessing, modeling, and evaluation.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder # Alternative for single-value categories
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.sparse import hstack # To combine sparse matrices from TF-IDF
import joblib # For saving the model (optional)
import warnings
import os

warnings.filterwarnings('ignore') # Ignore potential warnings for cleaner output

# %% [markdown]
# ## 2. Load Data
# Load your dataset. **Replace `'your_movie_dataset.csv'` with the actual path to your file.**

# %%
# --- Configuration ---
DATASET_PATH = 'movies.csv' # <<< REPLACE THIS WITH YOUR FILE PATH
# Define the names of the columns you want to use
# <<< REPLACE THESE WITH YOUR ACTUAL COLUMN NAMES >>>
NAME_COL = 'Name' # Keep movie name for reference if needed, but won't use as feature
GENRE_COL = 'Genre'
DIRECTOR_COL = 'Director'
ACTORS_COL = 'Actor 1' # Assuming multiple actor columns or a single one with combined names
# If actors are in separate columns (e.g., Actor 1, Actor 2, Actor 3), load them and combine later
RATING_COL = 'Rating'
YEAR_COL = 'Year' # Adding Year as it often influences ratings
DURATION_COL = 'Duration' # Adding Duration as another potential feature
VOTES_COL = 'Votes' # Adding Votes as a potentially strong feature
OUTPUT_DIR = 'output' #Output folder directory

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")

# Specify columns to load if the dataset is large or has many irrelevant columns
# If you want to load all columns, set use_cols=None
use_cols = [NAME_COL, YEAR_COL, DURATION_COL, GENRE_COL, RATING_COL, VOTES_COL, DIRECTOR_COL, ACTORS_COL, 'Actor 2', 'Actor 3'] # Example - adjust as needed

# --- Load Data ---
try:
    df = pd.read_csv(DATASET_PATH, usecols=use_cols, encoding='utf-8') # Try utf-8 encoding first
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: Dataset file not found at '{DATASET_PATH}'")
    print("Please make sure the file path is correct.")
    exit() # Stop execution if file not found
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    # Attempt with a different encoding if UTF-8 fails (common issue)
    try:
        df = pd.read_csv(DATASET_PATH, usecols=use_cols, encoding='latin1')
        print("Dataset loaded successfully using latin1 encoding.")
    except Exception as e2:
        print(f"Tried latin1 encoding, but failed again: {e2}")
        exit()

# %% [markdown]
# ## 3. Exploratory Data Analysis (EDA) & Preprocessing
# Understand the data, handle missing values, and prepare features.

# %%
# --- Initial Inspection ---
print("\n--- Dataset Info ---")
df.info()

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Descriptive Statistics (Numerical Columns) ---")
# Select only numeric columns for describe()
numeric_cols = df.select_dtypes(include=np.number).columns
print(df[numeric_cols].describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

# %%
# --- Handle Missing Values ---

# Strategy:
# - Rating: Drop rows where the target variable is missing (essential for training).
# - Genre, Director, Actors: Fill missing values with a placeholder like "Unknown".
# - Numerical (Year, Duration, Votes): Impute with the median (less sensitive to outliers than mean).

# Drop rows with missing ratings
print(f"\nShape before dropping missing ratings: {df.shape}")
df.dropna(subset=[RATING_COL], inplace=True)
print(f"Shape after dropping missing ratings: {df.shape}")

# Fill missing categorical features
for col in [GENRE_COL, DIRECTOR_COL, ACTORS_COL, 'Actor 2', 'Actor 3']: # Adjust actor columns if needed
    if col in df.columns:
        df[col].fillna('Unknown', inplace=True)

# Combine actor columns into a single 'Actors' string if they exist separately
if 'Actor 2' in df.columns and 'Actor 3' in df.columns:
    df['Actors_Combined'] = df[ACTORS_COL].astype(str) + ', ' + df['Actor 2'].astype(str) + ', ' + df['Actor 3'].astype(str)
    # Replace the placeholder actor column name with the new combined one
    ACTORS_COL = 'Actors_Combined'
    # We can drop the original actor columns if desired, but TF-IDF will handle the combined string
    # df.drop(['Actor 1', 'Actor 2', 'Actor 3'], axis=1, inplace=True)
elif ACTORS_COL in df.columns:
     # If only one actor column specified initially, ensure it's treated as the combined one
     ACTORS_COL = ACTORS_COL # Keep the original name if only one actor col exists
else:
    print(f"Warning: No actor columns ({ACTORS_COL}, Actor 2, Actor 3) found as specified.")
    # Create a dummy 'Unknown' column if actors are crucial and missing
    df['Actors_Combined'] = 'Unknown'
    ACTORS_COL = 'Actors_Combined'

# Clean and prepare numerical features
numerical_features_to_impute = []

if YEAR_COL in df.columns:
    print(f"Processing '{YEAR_COL}' column...")
    # Extract 4 digits using regex. Use .iloc[:, 0] to get the Series.
    year_extracted = df[YEAR_COL].astype(str).str.extract(r'(\d{4})').iloc[:, 0]
    # Convert to numeric, coercing errors (non-digits, empty strings) to NaN
    df[YEAR_COL] = pd.to_numeric(year_extracted, errors='coerce')
    if df[YEAR_COL].isnull().any():
        numerical_features_to_impute.append(YEAR_COL)
    print(f"  - '{YEAR_COL}' NaNs after cleaning: {df[YEAR_COL].isnull().sum()}")


if DURATION_COL in df.columns:
    print(f"Processing '{DURATION_COL}' column...")
     # Extract digits using regex.
    duration_extracted = df[DURATION_COL].astype(str).str.extract(r'(\d+)').iloc[:, 0]
    # Convert to numeric, coercing errors to NaN (handles empty strings from replace, original NaNs, etc.)
    df[DURATION_COL] = pd.to_numeric(duration_extracted, errors='coerce')
    if df[DURATION_COL].isnull().any():
        numerical_features_to_impute.append(DURATION_COL)
    print(f"  - '{DURATION_COL}' NaNs after cleaning: {df[DURATION_COL].isnull().sum()}")


if VOTES_COL in df.columns:
    print(f"Processing '{VOTES_COL}' column...")
    # Remove commas first
    votes_cleaned = df[VOTES_COL].astype(str).str.replace(',', '', regex=False)
    # Convert to numeric, coercing errors (like from non-numeric strings after comma removal) to NaN
    df[VOTES_COL] = pd.to_numeric(votes_cleaned, errors='coerce')
    if df[VOTES_COL].isnull().any():
        numerical_features_to_impute.append(VOTES_COL)
    print(f"  - '{VOTES_COL}' NaNs after cleaning: {df[VOTES_COL].isnull().sum()}")


# Impute numerical features with median ONLY if NaNs exist after cleaning
print("\nImputing missing values in numerical columns (using median)...")
if not numerical_features_to_impute:
     print("  - No numerical columns required imputation after cleaning.")
else:
    for col in numerical_features_to_impute:
        if col in df.columns and df[col].isnull().any(): # Double check column exists and has NaNs
             median_val = df[col].median()
             df[col].fillna(median_val, inplace=True)
             print(f"  - Imputed NaNs in '{col}' with median ({median_val:.2f}). Remaining NaNs: {df[col].isnull().sum()}")
        elif col not in df.columns:
             print(f"  - Warning: Column '{col}' intended for imputation not found.")
        else:
              print(f"  - No NaNs to impute in '{col}' after cleaning.")


# Final check for numerical columns intended for modeling
print("\nChecking final numerical columns for NaNs before proceeding:")
final_numerical_cols = [col for col in [YEAR_COL, DURATION_COL, VOTES_COL] if col in df.columns]
print(df[final_numerical_cols].isnull().sum())
# If any NaNs remain here, something went wrong or the column wasn't processed
if df[final_numerical_cols].isnull().any().any():
    print("WARNING: NaNs still present in numerical features after processing! Check cleaning steps.")

# Ensure the list used later matches the columns actually processed and present
numerical_features_to_impute = final_numerical_cols # Update the list to reflect actual columns


print("\n--- Missing Values After Handling ---")
print(df.isnull().sum())

# Convert Rating to numeric if it's not already (it should be after dropna)
df[RATING_COL] = pd.to_numeric(df[RATING_COL])

# %%
# --- Feature Engineering & Transformation (using TF-IDF) ---

# TF-IDF (Term Frequency-Inverse Document Frequency) is suitable for text features
# like Genre, Director, Actors, especially when they might contain multiple items.
# It converts text into numerical vectors based on word importance.

# Initialize TF-IDF Vectorizers
# max_features limits the number of features created to prevent excessive memory usage/dimensionality
tfidf_genre = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_director = TfidfVectorizer(max_features=2500)
tfidf_actors = TfidfVectorizer(max_features=5000) # Actors might have a larger vocabulary

# Define the features (X) and target (y)
X_text_features = df[[GENRE_COL, DIRECTOR_COL, ACTORS_COL]]
X_numerical_features = df[numerical_features_to_impute] # Use the cleaned numerical columns
y = df[RATING_COL]

# Split data *before* applying TF-IDF to prevent data leakage from the test set
X_train, X_test, y_train, y_test = train_test_split(df.drop(RATING_COL, axis=1), # Drop target from features df
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Apply TF-IDF *separately* to Train and Test sets
# Fit and Transform on Training Data
X_train_genre_tfidf = tfidf_genre.fit_transform(X_train[GENRE_COL])
X_train_director_tfidf = tfidf_director.fit_transform(X_train[DIRECTOR_COL])
X_train_actors_tfidf = tfidf_actors.fit_transform(X_train[ACTORS_COL])

# Only Transform on Test Data (using the vectorizers fitted on train data)
X_test_genre_tfidf = tfidf_genre.transform(X_test[GENRE_COL])
X_test_director_tfidf = tfidf_director.transform(X_test[DIRECTOR_COL])
X_test_actors_tfidf = tfidf_actors.transform(X_test[ACTORS_COL])

# Combine TF-IDF features with numerical features
# Use hstack for sparse matrices (from TF-IDF) and convert numerical features to sparse or dense as needed.
# Ensure numerical features are handled correctly (e.g., scaling might be needed for some models, but often okay for tree-based models)

# Selecting numerical features from the split data
X_train_numerical = X_train[numerical_features_to_impute].astype(float)
X_test_numerical = X_test[numerical_features_to_impute].astype(float)

# Combine all features horizontally
# hstack requires inputs to be sparse matrices. Convert numerical dataframe to sparse.
from scipy.sparse import csr_matrix
X_train_combined = hstack([X_train_genre_tfidf, X_train_director_tfidf, X_train_actors_tfidf, csr_matrix(X_train_numerical)])
X_test_combined = hstack([X_test_genre_tfidf, X_test_director_tfidf, X_test_actors_tfidf, csr_matrix(X_test_numerical)])


print(f"\nShape of combined training features: {X_train_combined.shape}")
print(f"Shape of combined testing features: {X_test_combined.shape}")
print(f"Shape of training target: {y_train.shape}")
print(f"Shape of testing target: {y_test.shape}")


# %% [markdown]
# ## 4. Model Training
# Train several regression models on the prepared data.

# %%
# --- Initialize Models ---
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0), # Alpha controls regularization strength
    # "Lasso Regression": Lasso(alpha=0.1), # Lasso can perform feature selection
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15, min_samples_leaf=5), # n_jobs=-1 uses all CPU cores; Added some basic hyperparameters
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5, min_samples_leaf=5) # Added some basic hyperparameters
}

# --- Train Models ---
trained_models = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_combined, y_train)
    trained_models[name] = model
    print(f"{name} trained.")

# %% [markdown]
# ## 5. Model Evaluation
# Evaluate the performance of the trained models on the test set using common regression metrics.

# %%
# --- Evaluate Models ---
results = {}
for name, model in trained_models.items():
    print(f"\nEvaluating {name}...")
    y_pred = model.predict(X_test_combined)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  R-squared (R2): {r2:.4f}")

# --- Compare Results ---
results_df = pd.DataFrame(results).T # Transpose for better readability
print("\n--- Model Comparison ---")
print(results_df)

# Find the best model based on a chosen metric (e.g., R2 Score or RMSE)
best_model_name = results_df['R2'].idxmax() # Highest R2 is often preferred
# best_model_name = results_df['RMSE'].idxmin() # Lowest RMSE is also a good indicator
best_model = trained_models[best_model_name]
print(f"\nBest Model (based on R2 Score): {best_model_name}")

# %% [markdown]
# ## 6. Visualization (Optional)
# Visualize actual vs. predicted ratings for the best model.

# %%
plt.figure(figsize=(10, 6))
plt.scatter(y_test, best_model.predict(X_test_combined), alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2) # Perfect prediction line
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title(f"Actual vs. Predicted Ratings ({best_model_name})")
plt.grid(True)

# Construct the full path for saving the plot
plot_filename = os.path.join(OUTPUT_DIR, f"Actual_vs_Predicted_Ratings_{best_model_name.replace(' ', '_')}.png")
plt.savefig(plot_filename)
print(f"Plot saved to {plot_filename}")
plt.show() # Uncomment if you want the plot to pop up as well
plt.close() # Close the plot figure to free up memory

# %% [markdown]
# ## 7. Saving the Best Model (Optional)
# Save the model that performed best for future use (e.g., deployment or making predictions on new data without retraining).

# %%
# --- Save Model ---
# Define base file names
model_base_filename = f'best_movie_rating_model_{best_model_name.replace(" ", "_")}.joblib'
vectorizers_base_filename = 'tfidf_vectorizers.joblib'

# Construct full filepaths within the OUTPUT_DIR
model_filepath = os.path.join(OUTPUT_DIR, model_base_filename)
vectorizers_filepath = os.path.join(OUTPUT_DIR, vectorizers_base_filename)

try:
    # Save the trained model object
    joblib.dump(best_model, model_filepath)

    # Save the fitted vectorizers (essential for processing new data)
    vectorizers_to_save = { # Renamed to avoid conflict if 'vectorizers' used elsewhere, though unlikely here
        'genre': tfidf_genre,
        'director': tfidf_director,
        'actors': tfidf_actors,
        'numerical_cols': numerical_features_to_impute # Also save the list of numerical columns used
    }
    joblib.dump(vectorizers_to_save, vectorizers_filepath)

    print(f"\nBest model ({best_model_name}) saved to {model_filepath}")
    print(f"TF-IDF vectorizers saved to {vectorizers_filepath}")

except Exception as e:
    print(f"\nError saving model or vectorizers: {e}")


# %% [markdown]
# ## 8. Example Prediction with Saved Model (Optional)
# Show how to load the saved model and vectorizers to predict the rating for a hypothetical new movie.

# %%
# --- Load Model and Vectorizers ---
# NOTE: This section assumes 'best_model_name' from Section 5 and 'OUTPUT_DIR' from Section 2 are available.
# If running this section independently (e.g., new session), these might need to be redefined or loaded.

try:
    # Construct the full file paths for loading from the OUTPUT_DIR
    # These must match exactly how they were saved in Section 7.
    # 'best_model_name' should be available from the model evaluation part (Section 5).
    if 'best_model_name' not in locals() and 'best_model_name' not in globals():
        raise NameError("'best_model_name' is not defined. Run model evaluation first.")
    if 'OUTPUT_DIR' not in locals() and 'OUTPUT_DIR' not in globals():
         raise NameError("'OUTPUT_DIR' is not defined. Check script configuration.")


    load_model_base_filename = f'best_movie_rating_model_{best_model_name.replace(" ", "_")}.joblib'
    load_vectorizers_base_filename = 'tfidf_vectorizers.joblib'

    load_model_filepath = os.path.join(OUTPUT_DIR, load_model_base_filename)
    load_vectorizers_filepath = os.path.join(OUTPUT_DIR, load_vectorizers_base_filename)

    loaded_model = joblib.load(load_model_filepath)
    loaded_vectorizers_dict = joblib.load(load_vectorizers_filepath) # Use a distinct name
    print(f"\nModel loaded from: {load_model_filepath}")
    print(f"Vectorizers loaded from: {load_vectorizers_filepath}")


    # --- Prepare New Data Sample ---
    # Create a sample DataFrame with the same columns used for training
    # Use the placeholder 'Unknown' if data is missing
    new_movie_data = pd.DataFrame([{
        GENRE_COL: 'Action|Adventure|Sci-Fi',
        DIRECTOR_COL: 'Jane Doe',
        ACTORS_COL: 'Actor X, Actor Y, Actor Z', # Must match the combined ACTORS_COL
        YEAR_COL: 2024.0,
        DURATION_COL: 135.0,
        VOTES_COL: 150000.0
    }])

    # --- Preprocess New Data using Loaded Vectorizers ---
    new_genre_tfidf = loaded_vectorizers_dict['genre'].transform(new_movie_data[GENRE_COL])
    new_director_tfidf = loaded_vectorizers_dict['director'].transform(new_movie_data[DIRECTOR_COL])
    new_actors_tfidf = loaded_vectorizers_dict['actors'].transform(new_movie_data[ACTORS_COL])
    new_numerical = new_movie_data[loaded_vectorizers_dict['numerical_cols']].astype(float)

    # Combine features in the same order as during training
    new_combined_features = hstack([new_genre_tfidf, new_director_tfidf, new_actors_tfidf, csr_matrix(new_numerical)])

    # --- Make Prediction ---
    predicted_rating = loaded_model.predict(new_combined_features)

    print(f"\nPredicted Rating for the new movie: {predicted_rating[0]:.2f}")

except NameError as e:
    print(f"\nA NameError occurred: {e}. This often means a necessary variable (like 'best_model_name' or 'OUTPUT_DIR') was not available.")
    print("Ensure the required preceding sections of the script have run and set these variables.")
except FileNotFoundError:
    print(f"\nCould not load saved model/vectorizers. Check if the following files exist in the '{OUTPUT_DIR}' directory:")
    # Try to print expected paths for debugging, even if best_model_name was an issue
    try:
        if 'best_model_name' in locals() or 'best_model_name' in globals():
            expected_model_path = os.path.join(OUTPUT_DIR, f'best_movie_rating_model_{best_model_name.replace(" ", "_")}.joblib')
            print(f"  Model: {expected_model_path}")
        else:
            print("  Model: Could not determine expected model path because 'best_model_name' is missing.")
        expected_vec_path = os.path.join(OUTPUT_DIR, 'tfidf_vectorizers.joblib')
        print(f"  Vectorizers: {expected_vec_path}")
    except Exception: # General catch if even printing paths fails
         print("  Could not fully determine expected paths.")
    print("Ensure the saving step (Section 7) successfully created these files.")
except Exception as e:
    print(f"\nAn error occurred during prediction with loaded model: {e}")


# %% [markdown]
# ## Conclusion
#
# This project demonstrated the process of building a movie rating prediction model. We performed:
# 1.  **Data Loading & Cleaning:** Handled missing values and cleaned data types.
# 2.  **Feature Engineering:** Used TF-IDF to convert text features (Genre, Director, Actors) into numerical representations and included numerical features (Year, Duration, Votes).
# 3.  **Model Training:** Trained several regression models (Linear Regression, Ridge, Random Forest, Gradient Boosting).
# 4.  **Model Evaluation:** Compared models using MAE, MSE, RMSE, and R2 score, identifying the best performer on the test set.
# 5.  **(Optional) Model Saving & Prediction:** Showcased how to save the best model and its associated transformers (vectorizers) to make predictions on new, unseen data.
#
# **Potential Improvements:**
# *   **More Sophisticated Feature Engineering:** Extracting the number of actors, director's average past rating, genre combinations, etc.
# *   **Advanced Text Processing:** Using techniques like word embeddings (Word2Vec, GloVe) instead of TF-IDF.
# *   **Hyperparameter Tuning:** Using GridSearchCV or RandomizedSearchCV to find the optimal parameters for the best model.
# *   **Different Models:** Trying other regression algorithms like XGBoost, LightGBM, or Support Vector Regression (SVR).
# *   **Handling Categorical Features Differently:** Experimenting with One-Hot Encoding (carefully, due to high cardinality) or Target Encoding.
# *   **Using Pipelines:** Encapsulating preprocessing and modeling steps using `sklearn.pipeline.Pipeline` for cleaner code and preventing data leakage.

# %%
print("\n--- Project Execution Finished ---")