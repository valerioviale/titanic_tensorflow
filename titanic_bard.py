import optuna
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd

def preprocess(df):
    # TODO: Implement your preprocessing steps here.
    return df

def train_model(train_ds, hyperparameters):
    model = tfdf.keras.GradientBoostedTreesModel(**hyperparameters)
    model.fit(train_ds)
    return model

def evaluate_model(model, test_ds):
    predictions = model.predict(test_ds, verbose=0)[:, 0]
    accuracy = (predictions >= 0.5).astype(int) == test_ds["Survived"]
    return accuracy.mean()

def tune_hyperparameters(train_ds, test_ds):
    def objective(trial):
        trial.suggest_int("num_trees", 100, 1000, log=True)
        trial.suggest_int("max_depth", 3, 8, log=True)
        trial.suggest_float("shrinkage", 0.01, 0.1)

        model = train_model(train_ds, trial.params)
        accuracy = evaluate_model(model, test_ds)
        return accuracy

    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    return study.best_trial.params

def main():
    # Load the data.
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    # Preprocess the data.
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(preprocess(train_df))
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(preprocess(test_df))

    # Tune the hyperparameters.
    hyperparameters = tune_hyperparameters(train_ds, test_ds)

    # Train the model with the best hyperparameters.
    model = train_model(train_ds, hyperparameters)

    # Evaluate the model on the test set.
    accuracy = evaluate_model(model, test_ds)
    print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()
    
    # Write results to submission.csv
    submission_df = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": (model.predict(test_ds, verbose=0)[:, 0] >= 0.5).astype(int)
    })
    submission_df.to_csv("submission.csv", index=False)
    print("Results written to submission.csv")
