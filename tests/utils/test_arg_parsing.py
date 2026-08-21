import sys

from adaptive_filter.utils.arg_parsing import parse_args


def test_parse_args(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])
    args = parse_args()

    assert args.filter_order == 16
    assert isinstance(args.filter_order, int)

    monkeypatch.setattr(sys, "argv", ["prog", "--filter_order", "8"])
    args2 = parse_args()

    assert args2.filter_order == 8
    assert isinstance(args.filter_order, int)
