import numpy as np

from adaptive_filter.filter_models.fd_filter_model import FrequencyDomainAF


def fd_test_filter():
    model = FrequencyDomainAF(mu=0.1, filter_order=1, block_size=2)
    # overriding update step
    # model.update_step = lambda e_n, x_n: np.array([0.0])

    d = np.linspace(1, 5, 6)
    x = np.linspace(0.5, 2.5, 5)
    clean = np.linspace(1, 5, 8)
    if d.shape[0] < x.shape[0]:
        x = x[: d.shape[0]]
        assert x.shape[0] == d.shape[0]
    if x.shape[0] < d.shape[0]:
        d = d[: x.shape[0]]
        assert d.shape[0] == x.shape[0]
    if d.shape[0] < clean.shape[0]:
        clean = clean[: d.shape[0]]
        assert clean.shape[0] == d.shape[0]
    if clean.shape[0] < d.shape[0]:
        d = d[: clean.shape[0]]
        assert d.shape[0] == clean.shape[0]
        x = x[: clean.shape[0]]
        assert x.shape[0] == clean.shape[0]
    # checking the signal shapes
    assert d.shape == x.shape
    assert d.shape == clean.shape

    results = model.filter(d, x)
    assert isinstance(results, np.ndarray)
