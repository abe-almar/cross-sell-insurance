# src/model_training.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import pickle


def build_full_pipeline(processing_pipeline, remove_collinear, pca_n_components):
    """Builds the full pipeline, including feature processing, collinearity removal, PCA, and RandomForestClassifier."""

    return Pipeline(
        steps=[
            ("feature_processing", processing_pipeline),
            ("remove_collinear", remove_collinear),
            ("pca", PCA(n_components=pca_n_components)),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )


def tune_model(
    pipeline, X_train, y_train, param_grid, cv_splits=5, save_model_path=None
):
    """
    Trains a model using a given pipeline and performs grid search for hyperparameter tuning.

    Args:
        pipeline (Pipeline): The preprocessing and model pipeline.
        X_train (pd.DataFrame): The training features.
        y_train (np.array): The training labels.
        param_grid (dict): The hyperparameter grid for tuning.
        cv_splits (int): Number of cross-validation splits (default: 5).
        save_model_path(str): Path to save the best model using Pickle (default: None)

    Returns:
        The best estimator from the grid search.
    """
    # Create a StratifiedKFold cross-validator with specified splits
    cv = StratifiedKFold(n_splits=cv_splits)

    # Set up GridSearchCV with the provided pipeline and param_grid
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="accuracy")

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Save the model using Pickle, if save_model_path is provided
    if save_model_path:
        with open(save_model_path, "wb") as f:
            pickle.dump(grid_search.best_estimator_, f)
        print(f"Model saved to {save_model_path}")

    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        grid_search.best_score_,
    )
