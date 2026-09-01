from adaptive_filters import LMSFilter


def test_lms_filter():
    filter = LMSFilter(1.0, 1024)

    assert hasattr(filter, "window_size")
    assert hasattr(filter, "adapt")
    assert hasattr(filter, "filter")
