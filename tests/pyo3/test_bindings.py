from adaptive_filters import LMSFilter


def test_lms_filter():
    filter = LMSFilter(1.0, 1024)

    assert filter.window_size == 1024
