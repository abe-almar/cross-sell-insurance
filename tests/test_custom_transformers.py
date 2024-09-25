import pytest
import numpy as np
from src.custom_transformers import RemoveCollinearFeatures


@pytest.fixture
def correlated_data():
    """Sample input data with high collinearity between columns."""
    return np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])


def test_remove_collinear_features(correlated_data):
    """Test that the RemoveCollinearFeatures transformer is removing highly collinear features

    Args:
        correlated_data (np.array): Sample input data with high collinearity between columns.
    """
    tranformer = RemoveCollinearFeatures(threshold=0.9)
    transformed_data = tranformer.fit_transform(correlated_data)

    # ensure that only one feature remains after removing collinear features
    assert transformed_data.shape[1] == 1
