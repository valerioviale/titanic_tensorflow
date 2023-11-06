# Import necessary libraries
import numpy as np
import pandas as pd
import os

import tensorflow as tf
import tensorflow_decision_forests as tfdf

# Print the version of TensorFlow Decision Forests
print(f"Found TF-DF {tfdf.__version__}")

# Read the training and test datasets
train_df = pd.read_csv("train.csv")
serving_df = pd.read_csv("test.csv")

# Display the first 10 rows of the training dataset
train_df.head(10)

# Define a preprocessing function to clean and modify the dataset
def preprocess(df):
    df = df.copy()

    # Normalize the "Name" column by removing special characters
    def normalize_name(x):
        return " ".join([v.strip(",()[].\"'") for v in x.split(" ")])
    
    # Extract the last part of the "Ticket" as "Ticket_number"
    def ticket_number(x):
        return x.split(" ")[-1]
    
    # Extract the items from the "Ticket" and join them with "_"
    def ticket_item(x):
        items = x.split(" ")
        if len(items) == 1:
            return "NONE"
        return "_".join(items[0:-1])

    # Apply the defined functions to preprocess the dataset
    df["Name"] = df["Name"].apply(normalize_name)
    df["Ticket_number"] = df["Ticket"].apply(ticket_number)
    df["Ticket_item"] = df["Ticket"].apply(ticket_item)
    
    return df

# Preprocess the training and test datasets
preprocessed_train_df = preprocess(train_df)
preprocessed_serving_df = preprocess(serving_df)

# Display the first 5 rows of the preprocessed training dataset
preprocessed_train_df.head(5)

# Define input features and remove unnecessary columns
input_features = list(preprocessed_train_df.columns)
input_features.remove("Ticket")
input_features.remove("PassengerId")
input_features.remove("Survived")

# Print the input features
print(f"Input features: {input_features}")

# Tokenize the "Name" column for TensorFlow Decision Forests
def tokenize_names(features, labels=None):
    """Divide the names into tokens. TF-DF can consume text tokens natively."""
    features["Name"] = tf.strings.split(features["Name"])
    return features, labels

# Convert the preprocessed data to TensorFlow datasets
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(preprocessed_train_df, label="Survived").map(tokenize_names)
serving_ds = tfdf.keras.pd_dataframe_to_tf_dataset(preprocessed_serving_df).map(tokenize_names)

# Create a Gradient Boosted Trees model with specified settings and train it
model = tfdf.keras.GradientBoostedTreesModel(
    verbose=0,
    features=[tfdf.keras.FeatureUsage(name=n) for n in input_features],
    exclude_non_specified_features=True,
    random_seed=1234,
)
model.fit(train_ds)

# Evaluate the model's performance on the training data
self_evaluation = model.make_inspector().evaluation()
print(f"Accuracy: {self_evaluation.accuracy} Loss: {self_evaluation.loss}")

# Create another Gradient Boosted Trees model with different settings and train it
model = tfdf.keras.GradientBoostedTreesModel(
    verbose=0,
    features=[tfdf.keras.FeatureUsage(name=n) for n in input_features],
    exclude_non_specified_features=True,
    min_examples=1,
    categorical_algorithm="RANDOM",
    shrinkage=0.05,
    split_axis="SPARSE_OBLIQUE",
    sparse_oblique_normalization="MIN_MAX",
    sparse_oblique_num_projections_exponent=2.0,
    num_trees=2000,
    random_seed=1234,
)
model.fit(train_ds)

# Evaluate the second model's performance on the training data
self_evaluation = model.make_inspector().evaluation()
print(f"Accuracy: {self_evaluation.accuracy} Loss: {self_evaluation.loss}")

# Display a summary of the model
model.summary()

# Define a function to convert model predictions to Kaggle submission format
def prediction_to_kaggle_format(model, threshold=0.5):
    proba_survive = model.predict(serving_ds, verbose=0)[:, 0]
    return pd.DataFrame({
        "PassengerId": serving_df["PassengerId"],
        "Survived": (proba_survive >= threshold).astype(int)
    })

# Define a function to create a Kaggle submission file
def make_submission(kaggle_predictions):
    path = "submission.csv"
    kaggle_predictions.to_csv(path, index=False)
    print(f"Submission exported to {path}")

# Generate Kaggle predictions using the model and create a submission file
kaggle_predictions = prediction_to_kaggle_format(model)
make_submission(kaggle_predictions)

# Read and display the contents of the "submission.csv" file
with open("submission.csv", "r") as file:
    contents = file.read()

print(contents)

# Initialize a RandomSearch tuner for hyperparameter tuning
tuner = tfdf.tuner.RandomSearch(num_trials=1000)

# Define search space for hyperparameters
tuner.choice("learning_rate", [0.01, 0.1, 0.2, 0.3, 0.5])
tuner.choice("max_depth", [3, 4, 5, 6, 8, 10])
tuner.choice("num_trees", [100, 200, 500, 1000, 2000])

# Tune the model using the tuner
tuned_model = tfdf.keras.GradientBoostedTreesModel(tuner=tuner)
tuned_model.fit(train_ds, verbose=0)

# Evaluate the performance of the tuned model
tuned_self_evaluation = tuned_model.make_inspector().evaluation()
print(f"Accuracy: {tuned_self_evaluation.accuracy} Loss: {tuned_self_evaluation.loss}")

# Generate Kaggle predictions using the tuned model and create a submission file
kaggle_predictions = prediction_to_kaggle_format(tuned_model)
make_submission(kaggle_predictions)