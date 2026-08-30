import numpy as np
import pytest

from adaptive_filter.filter_models.fd_filter_model import FrequencyDomainAF


def test_filter_returns_array(monkeypatch: pytest.MonkeyPatch):
    model = FrequencyDomainAF(mu=0.1, filter_order=1, block_size=2)

    # basically this replaces update_step with a lambda that returns
    # -> an array of 0.0, to test that filter() returns the array correctly.
    # Doesn't update algo, checks ndarray and shape
    monkeypatch.setattr(
        model,
        "update_step",
        lambda e_n, x_n: np.array([0.0], dtype=np.float64),
    )

    d = np.linspace(1, 5, 5)
    x = np.linspace(0.5, 2.5, 5)

    result = model.filter(d, x)

    assert isinstance(result, np.ndarray)
    assert result.shape == d.shape
