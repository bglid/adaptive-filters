import numpy as np
import pytest

from adaptive_filter.algorithms import lms


# creating an expected output to test output
@pytest.mark.parametrize(
    "mu,e_n,x_n, expected_result",
    [
        (0.5, 2.0, np.array([1.0, -1.0]), np.array([1.0, -1.0])),
        (1.0, 1.0, np.array([5.0, 2.0]), np.array([5.0, 2.0])),
        # (-1.0, 1.0, np.array([0.5, 0.25]), np.array([-0.5, -0.25])),  # mu cannot be negative!
    ],
)
def test_update_step(mu, e_n, x_n, expected_result):
    # creating filter object
    filter = lms.LMS(mu=mu, n=len(x_n))
    output = filter.update_step(e_n, x_n)
    assert isinstance(output, np.ndarray)
    np.testing.assert_allclose(output, expected_result, rtol=1e-8)
