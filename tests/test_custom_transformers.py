from app.src.custom_transformers import RemoveCollinearFeatures
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


def test_remove_collinear_features(correlated_data):
    """Test that the RemoveCollinearFeatures transformer is removing highly collinear features

    Args:
        correlated_data (np.array): Sample input data with high collinearity between columns.
    """
    tranformer = RemoveCollinearFeatures(threshold=0.9)
    transformed_data = tranformer.fit_transform(correlated_data)

    # ensure that only one feature remains after removing collinear features
    assert transformed_data.shape[1] == 1
