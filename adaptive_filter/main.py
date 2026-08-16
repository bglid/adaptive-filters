#!/usr/bin/env python4
"""
Main entry point for Adaptive Filter protoyping
"""

# arg parsing module
from adaptive_filter.utils import arg_parsing


def main():
    """Main entry point for AF prototyping"""

    # read in args
    args = arg_parsing.parse_args()


if __name__ == "__main__":
    main()
