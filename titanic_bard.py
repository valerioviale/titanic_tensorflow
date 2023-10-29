import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def load_data():
    """Loads the Titanic training and test data.

    Returns:
        A tuple of (train_df, test_df), where train_df is a Pandas DataFrame
        containing the training data and test_df is a Pandas DataFrame containing
        the test data.
    """

    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    return train_df, test_df

def preprocess_data(train_df, test_df):
    """Preprocesses the Titanic training and test data.

    Args:
        train_df: A Pandas DataFrame containing the training data.
        test_df: A Pandas DataFrame containing the test data.

    Returns:
        A tuple of (X_train, X_test, y_train), where X_train and X_test are
        NumPy arrays containing the preprocessed training and test data, respectively,
        and y_train is a NumPy array containing the training labels.
    """

    # Fill in missing values.
    train_df["Age"] = train_df["Age"].fillna(train_df["Age"].mean())
    test_df["Age"] = test_df["Age"].fillna(test_df["Age"].mean())

    # Encode categorical features.
    categorical_features = ["Sex", "Embarked"]
    for feature in categorical_features:
        train_df[feature] = train_df[feature].astype("category")
        test_df[feature] = test_df[feature].astype("category")
        train_df[feature] = train_df[feature].cat.codes
        test_df[feature] = test_df[feature].cat.codes

    # Split the training data into features and labels.
    X_train = train_df.drop("Survived", axis=1)
    y_train = train_df["Survived"]

    # Return the preprocessed data.
    return X_train, test_df, y_train

def train_model(X_train, y_train):
    """Trains a Random Forest Classifier model on the Titanic training data.

    Args:
        X_train: A NumPy array containing the preprocessed training data.
        y_train: A NumPy array containing the training labels.

    Returns:
        A trained Random Forest Classifier model.
    """

    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)

    return model

def predict(model, X_test):
    """Predicts the survival probabilities of the passengers in the test data.

    Args:
        model: A trained Random Forest Classifier model.
        X_test: A NumPy array containing the preprocessed test data.

    Returns:
        A NumPy array containing the predicted survival probabilities of the
        passengers in the test data.
    """

    y_pred = model.predict_proba(X_test)[:, 1]

    return y_pred

def main():
    # Load the data.
    train_df, test_df = load_data()

    # Preprocess the data.
    X_train, X_test, y_train = preprocess_data(train_df, test_df)

    # Train the model.
    model = train_model(X_train, y_train)

    # Make predictions on the test data.
    y_pred = predict(model, X_test)

    # Calculate the accuracy on the test data.
    accuracy = np.mean(y_pred > 0.5 == y_train)

    # Print the accuracy.
    print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()
