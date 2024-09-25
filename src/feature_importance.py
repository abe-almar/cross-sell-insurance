# src/feature_importance.py
from sklearn.ensemble import RandomForestClassifier


def extract_feature_importance(
    pipeline, X_train, y_train, numeric_columns, categorical_columns, binary_columns
):
    """Extracts feature importance after the transformation pipeline."""

    # Get column names after one-hot encoding and other transformations
    cat_features = (
        pipeline.named_steps["feature_processing"]
        .transformers_[1][1]["onehot"]
        .get_feature_names_out(categorical_columns)
    )
    all_features = numeric_columns + list(cat_features) + binary_columns

    # Transform the data before PCA and collinearity removal for feature importance extraction
    pre_pca_data = pipeline.named_steps["feature_processing"].transform(X_train)

    # Train a standalone RandomForestClassifier for feature importance on pre-PCA data
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(pre_pca_data, y_train)

    importances = rf.feature_importances_
    feature_importance = sorted(
        zip(all_features, importances), key=lambda x: x[1], reverse=True
    )
    return feature_importance
 #   for feature, importance in feature_importance:
  #      print(f"Feature: {feature}, Importance: {importance}")
