import numpy as np
import pandas as pd
import os

import tensorflow as tf
import tensorflow_decision_forests as tfdf

print(f"Found TF-DF {tfdf.__version__}")

# Load the data
train_df = pd.read_csv("train.csv")
serving_df = pd.read_csv("test.csv")

# Preprocessing function
def preprocess(df):
    df = df.copy()

    # Normalize names
    def normalize_name(x):
        return " ".join([v.strip(",()[].\"'") for v in x.split(" ")])
    
    # Extract ticket number
    def ticket_number(x):
        return x.split()[-1]

    # Extract ticket item
    def ticket_item(x):
        items = x.split()
        if len(items) == 1:
            return "NONE"
        return "_".join(items[:-1])

    df["Name"] = df["Name"].apply(normalize_name)
    df["Ticket_number"] = df["Ticket"].apply(ticket_number)
    df["Ticket_item"] = df["Ticket"].apply(ticket_item)

    return df

# Preprocess the data
preprocessed_train_df = preprocess(train_df)
preprocessed_serving_df = preprocess(serving_df)

# Define input features
input_features = list(preprocessed_train_df.columns)
input_features.remove("Ticket")
input_features.remove("PassengerId")
input_features.remove("Survived")

# Tokenize names
def tokenize_names(features, labels=None):
    features["Name"] = tf.strings.split(features["Name"])
    return features, labels

# Convert data to TensorFlow datasets
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    preprocessed_train_df, label="Survived").map(tokenize_names)
serving_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    preprocessed_serving_df).map(tokenize_names)

# Create and train the model
model = tfdf.keras.GradientBoostedTreesModel(
    verbose=0,
    features=[tfdf.keras.FeatureUsage(name=n) for n in input_features],
    exclude_non_specified_features=True,
    random_seed=1234
)
model.fit(train_ds)

# Model evaluation
self_evaluation = model.make_inspector().evaluation()
print(f"Accuracy: {self_evaluation.accuracy} Loss: {self_evaluation.loss}")

# Function to convert predictions to Kaggle format
def prediction_to_kaggle_format(model, threshold=0.5):
    proba_survive = model.predict(serving_ds, verbose=0)[:, 0]
    return pd.DataFrame({
        "PassengerId": serving_df["PassengerId"],
        "Survived": (proba_survive >= threshold).astype(int)
    })

# Make predictions and save to submission.csv
kaggle_predictions = prediction_to_kaggle_format(model)
kaggle_predictions.to_csv("submission.csv", index=False)
print("Submission exported to submission.csv")
