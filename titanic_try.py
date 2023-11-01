import numpy as np
import pandas as pd
import os

import tensorflow as tf
import tensorflow_decision_forests as tfdf
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

print(f"Found TF-DF {tfdf.__version__}")

train_df = pd.read_csv("train.csv")
serving_df = pd.read_csv("test.csv")

def preprocess(df):
  df = df.copy()

  def normalize_name(x):
      return " ".join([v.strip(",()[].\"'") for v in x.split(" ")])

  def ticket_number(x):
      return x.split(" ")[-1]

  def ticket_item(x):
      items = x.split(" ")
      if len(items) == 1:
          return "NONE"
      return "_".join(items[0:-1])

  df["Name"] = df["Name"].apply(normalize_name)
  df["Ticket_number"] = df["Ticket"].apply(ticket_number)
  df["Ticket_item"] = df["Ticket"].apply(ticket_item)
  return df

preprocessed_train_df = preprocess(train_df)
preprocessed_serving_df = preprocess(serving_df)

input_features = list(preprocessed_train_df.columns)
input_features.remove("Ticket")
input_features.remove("PassengerId")
input_features.remove("Survived")

def tokenize_names(features, labels=None):
  features["Name"] = tf.strings.split(features["Name"])
  return features, labels

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
  preprocessed_train_df, label="Survived").map(tokenize_names)
serving_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
  preprocessed_serving_df).map(tokenize_names)

# Define the model
model = tfdf.keras.GradientBoostedTreesModel(
  verbose=0,
  features=[tfdf.keras.FeatureUsage(name=n) for n in input_features],
  exclude_non_specified_features=True,
  random_seed=1234,
)

# Define the hyperparameters and their values to tune
param_grid = [
  {'min_examples': [2, 5, 7, 10]},
  {'categorical_algorithm': ["CART", "RANDOM"]},
  {'shrinkage': [0.02, 0.05, 0.10, 0.15]},
  {'num_candidate_attributes_ratio': [0.2, 0.5, 0.9, 1.0]},
  {'split_axis': ["AXIS_ALIGNED", "SPARSE_OBLIQUE"]},
  {'sparse_oblique_normalization': ["NONE", "STANDARD_DEVIATION", "MIN_MAX"]},
  {'sparse_oblique_weights': ["BINARY", "CONTINUOUS"]},
  {'sparse_oblique_num_projections_exponent': [1.0, 1.5]}
]

# Use Grid Search for hyperparameter tuning
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(train_ds, verbose=0)

# Print the best parameters
print(grid_search.best_params_)

# Retrain the model with the best parameters
best_model = tfdf.keras.GradientBoostedTreesModel(**grid_search.best_params_)
best_model.fit(train_ds)


