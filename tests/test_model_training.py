from app.src.model_training import build_full_pipeline, tune_model
from app.src.data_processing import get_preprocessing_pipeline
from app.src.custom_transformers import RemoveCollinearFeatures
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


def test_model_training(sample_training_data):
    """Test that the model pipeline trains succesfully without errors.

    Args:
        sample_training_data (_type_): Sample training data for the pipeline
    """

    X_train, y_train = sample_training_data
    # Define the columns for preprocessing
    numeric_columns = ["Age", "Annual_Premium"]
    binary_columns = ["Previously_Insured", "Driving_License"]
    categorical_columns = [
        "Gender",
        "Region_Code",
        "Vehicle_Age",
        "Vehicle_Damage",
        "Policy_Sales_Channel",
    ]
    # Define the param grid
    param_grid = {
        "classifier__n_estimators": [10, 20],  # Keep it small for unit testing
        "classifier__max_depth": [2, 4],
    }
    # build the pipeline

    processing_pipeline = get_preprocessing_pipeline(
        numeric_columns, categorical_columns, binary_columns
    )
    full_pipeline = build_full_pipeline(
        processing_pipeline,
        RemoveCollinearFeatures(threshold=0.9),
        pca_n_components=0.95,
    )

    # Train the model using grid search with cross-validation handled internally
    best_estimator, _, _ = tune_model(
        full_pipeline, X_train, y_train, param_grid, cv_splits=2
    )

    # ensure pipeline is fitted
    assert best_estimator.named_steps["classifier"].fit is not None
