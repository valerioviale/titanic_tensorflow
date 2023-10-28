import numpy as np
import pandas as pd
import os
import tensorflow as tf
import tensorflow_decision_forests as tfdf

# Define file paths as constants
TRAIN_CSV_PATH = "train.csv"
TEST_CSV_PATH = "test.csv"
SUBMISSION_PATH = "submission.csv"

# Function to preprocess data
def preprocess_data(df):
    df = df.copy()
    # Your preprocessing code here
    return df

# Function to train a model
def train_model(train_ds):
    model = tfdf.keras.GradientBoostedTreesModel(
        verbose=0,
        features=[tfdf.keras.FeatureUsage(name=n) for n in input_features],
        exclude_non_specified_features=True,
        random_seed=1234
    )
    model.fit(train_ds)
    return model

# Function to make predictions
def make_predictions(model, serving_ds):
    proba_survive = model.predict(serving_ds, verbose=0)[:, 0]
    return pd.DataFrame({
        "PassengerId": serving_df["PassengerId"],
        "Survived": (proba_survive >= 0.5).astype(int)
    })

if __name__ == "__main__":
    print(f"Found TF-DF {tfdf.__version__}")
    
    # Read data from CSV files
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    serving_df = pd.read_csv(TEST_CSV_PATH)
    
    # Preprocess data
    preprocessed_train_df = preprocess_data(train_df)
    preprocessed_serving_df = preprocess_data(serving_df)
    
    input_features = list(preprocessed_train_df.columns)
    input_features.remove("Ticket")
    input_features.remove("PassengerId")
    input_features.remove("Survived")
    
    # Convert data to TensorFlow datasets
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(preprocessed_train_df, label="Survived")
    serving_ds = tfdf.keras.pd_dataframe_to_tf_dataset(preprocessed_serving_df)
    
    # Train the model
    trained_model = train_model(train_ds)
    
    # Make predictions
    kaggle_predictions = make_predictions(trained_model, serving_ds)
    
    # Export submission
    kaggle_predictions.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission exported to {SUBMISSION_PATH}")
