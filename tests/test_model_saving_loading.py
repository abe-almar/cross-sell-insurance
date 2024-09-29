import pickle
from src.model_training import build_full_pipeline, tune_model
from src.data_processing import get_preprocessing_pipeline
from src.custom_transformers import RemoveCollinearFeatures
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


def test_model_saving_loading(sample_training_data):
    """Test that the model is saved and loaded successfully using Pickle."""

    X_train, y_train = sample_training_data

    # Define the columns for preprocessing
    numeric_columns = ["Age", "Annual_Premium"]
    categorical_columns = [
        "Gender",
        "Region_Code",
        "Vehicle_Age",
        "Vehicle_Damage",
        "Policy_Sales_Channel",
    ]
    binary_columns = ["Previously_Insured", "Driving_License"]

    # Build the pipeline
    processing_pipeline = get_preprocessing_pipeline(
        numeric_columns, categorical_columns, binary_columns
    )
    full_pipeline = build_full_pipeline(
        processing_pipeline,
        RemoveCollinearFeatures(threshold=0.9),
        pca_n_components=0.95,
    )

    # Define a small parameter grid for testing purposes
    param_grid = {
        "classifier__n_estimators": [10, 20],  # Keep it small for unit testing
        "classifier__max_depth": [2, 4],
    }

    # Train the model and save it
    model_save_path = "test_model.pkl"
    best_estimator, _, _ = tune_model(
        full_pipeline,
        X_train,
        y_train,
        param_grid,
        cv_splits=2,
        save_model_path=model_save_path,
    )

    # Load the model
    with open(model_save_path, "rb") as f:
        loaded_model = pickle.load(f)

    # Ensure the loaded model is the same and can make predictions
    assert loaded_model.named_steps["classifier"].fit is not None
    assert loaded_model.predict(X_train).shape == y_train.shape
