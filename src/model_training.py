# src/model_training.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def build_full_pipeline(processing_pipeline, remove_collinear, pca_n_components):
    """Builds the full pipeline, including feature processing, collinearity removal, PCA, and RandomForestClassifier."""
    
    return Pipeline(steps=[
        ('feature_processing', processing_pipeline),
        ('remove_collinear', remove_collinear),
        ('pca', PCA(n_components=pca_n_components)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

def tune_model(pipeline, X_train, y_train, param_grid):
    """Tunes the model using GridSearchCV to find the best hyperparameters."""
    
    grid_search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
