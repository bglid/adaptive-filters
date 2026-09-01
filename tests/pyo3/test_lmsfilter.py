import numpy as np
import pytest

from adaptive_filters import LMSFilter


@pytest.fixture
def filter():
    return LMSFilter(
        mu=0.1,
        window_size=1,
    )


def test_window_size():
    filter = LMSFilter(mu=1.0, window_size=1024)
    assert filter.window_size == 1024


def test_adapt(filter):
    # TODO: check that weights update

    input_signal = np.linspace(1, 5, 5)
    noise_ref = np.linspace(0.5, 2.5, 5)

    output = filter.adapt(input_signal, noise_ref)

    assert isinstance(output, np.ndarray)
    assert output.shape == input_signal.shape


def test_adapt_empty_input(filter):
    input_signal = np.linspace(1, 5, 5)
    noise_ref = np.linspace(0.5, 2.5, 5)

    with pytest.raises(ValueError):
        filter.adapt(np.array([]), noise_ref)
    with pytest.raises(ValueError):
        filter.adapt(input_signal, np.array([]))
    with pytest.raises(ValueError):
        filter.adapt(np.array([]), np.array([]))


def test_adapt_signal_lengths(filter):
    short_input = np.linspace(1, 5, 4)
    long_input = np.linspace(1, 5, 6)
    noise_ref = np.linspace(0.5, 2.5, 5)

    short_output = filter.adapt(short_input, noise_ref)

    assert isinstance(short_output, np.ndarray)
    assert short_output.shape == short_input.shape

    with pytest.raises(ValueError):
        filter.adapt(long_input, noise_ref)


def test_adapt_input_contiguous(filter):
    input_signal = np.linspace(1, 5, 10)
    noise_ref = np.linspace(0.5, 2.5, 10)

    # regular slices are contiguous -> allowed
    filter.adapt(input_signal[:5], noise_ref[:5])

    # strided slices are not contiguous -> not allowed
    with pytest.raises(ValueError):
        filter.adapt(input_signal[::2], noise_ref[:5])
    with pytest.raises(ValueError):
        filter.adapt(input_signal[:5], noise_ref[::2])
    with pytest.raises(ValueError):
        filter.adapt(input_signal[::2], noise_ref[::2])

    # calling np.ascontiguousarray makes it possible to use non-contiguous arrays as input
    filter.adapt(
        np.ascontiguousarray(input_signal[::2]), np.ascontiguousarray(noise_ref[::2])
    )


def test_filter(filter):
    # TODO: check that weights update

    input_signal = np.linspace(1, 5, 5)
    noise_ref = np.linspace(0.5, 2.5, 5)

    output = filter.filter(input_signal, noise_ref)

    assert isinstance(output, np.ndarray)
    assert output.shape == input_signal.shape


def test_filter_empty_input(filter):
    input_signal = np.linspace(1, 5, 5)
    noise_ref = np.linspace(0.5, 2.5, 5)

    with pytest.raises(ValueError):
        filter.filter(np.array([]), noise_ref)
    with pytest.raises(ValueError):
        filter.filter(input_signal, np.array([]))
    with pytest.raises(ValueError):
        filter.filter(np.array([]), np.array([]))


def test_filter_signal_lengths(filter):
    short_input = np.linspace(1, 5, 4)
    long_input = np.linspace(1, 5, 6)
    noise_ref = np.linspace(0.5, 2.5, 5)

    short_output = filter.filter(short_input, noise_ref)

    assert isinstance(short_output, np.ndarray)
    assert short_output.shape == short_input.shape

    with pytest.raises(ValueError):
        filter.filter(long_input, noise_ref)


def test_filter_input_contiguous(filter):
    input_signal = np.linspace(1, 5, 10)
    noise_ref = np.linspace(0.5, 2.5, 10)

    # regular slices are contiguous -> allowed
    filter.filter(input_signal[:5], noise_ref[:5])

    # strided slices are not contiguous -> not allowed
    with pytest.raises(ValueError):
        filter.filter(input_signal[::2], noise_ref[:5])
    with pytest.raises(ValueError):
        filter.filter(input_signal[:5], noise_ref[::2])
    with pytest.raises(ValueError):
        filter.filter(input_signal[::2], noise_ref[::2])

    # calling np.ascontiguousarray makes it possible to use non-contiguous arrays as input
    filter.filter(
        np.ascontiguousarray(input_signal[::2]), np.ascontiguousarray(noise_ref[::2])
    )
