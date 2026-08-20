#!/usr/bin/env python3
from types import ModuleType
from typing import Any, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

"""
PLOTTING UTILS FOR PRESENTING DIFFERNT ALGORITHMS
"""


class PlotSuite:
    def __init__(self, algorithm: str, num_samples: int) -> None:
        """
        Args:
            algorithm (str): String description of the Plot being used
            num_samples (int): Time in terms of number of samples

        """
        self.algorithm = algorithm
        self.N = num_samples

    # plot to measure time domain of signal
    def signal_plot(
        self,
        signal: np.ndarray[Any, np.dtype[Any]],
        description: str,
        subplot: bool = False,
    ) -> None | ModuleType:
        """
        Summary: Plot to trace a given signal in the time domain

        Args:
            signal (np.ndarray): Vector of output signal y_hat
            description (str): String description of plot, used in title
            subplot (bool): Bool check for if returning as a part of a larger plot.

        Returns:
            Union[None, ModuleType]: Returns plot when subplot set to True.

        """
        # indexing x to get points over time
        plt.plot(signal, c="darkviolet", label=f"{self.algorithm}")
        plt.title(f"{description} of {self.algorithm} algorithm")
        plt.ylabel("Amplitude")  # rotate this
        plt.xlabel(f"Time in {self.N} samples")
        plt.ylim([-4, 4])
        plt.legend()
        plt.grid()

        # if its not a subplot, show results
        if subplot is False:
            plt.show()
            return None
        else:
            return plt

    # plot to measure error changes over time
    def error_plot(
        self,
        results: np.ndarray[Any, np.dtype[Any]],
        error_metric: str,
        subplot: bool = False,
    ) -> None | ModuleType:
        """
        Summary: Plot to trace the error over time.

        Args:
            results (np.ndarray): Vector of error results
            error_metric (str): String description of error_metric, used in title
            subplot (bool): Bool check for if returning as a part of a larger plot.

        Returns:
            Union[None, ModuleType]: Returns plot when subplot set to True.

        """
        x_axis = np.arange(0, len(results))
        plt.title(f"{error_metric} of {self.algorithm}")
        plt.plot(x_axis, results, c="crimson", label=f"{error_metric}")
        plt.title(error_metric)
        plt.ylabel(error_metric)  # rotate this
        # plt.xlabel(f"Time in {self.N} samples")
        plt.legend()
        plt.grid()

        # if its not a subplot, show results
        if subplot is False:
            plt.show()
            return None
        else:
            return plt

    # plot to run all three and return them as subplots in one, if chosen
    def full_plot_suite(
        self,
        signal: np.ndarray[Any, np.dtype[Any]],
        description: str,
        results: np.ndarray[Any, np.dtype[Any]],
        error_metric: str,
    ) -> None:
        """Summary: Plots the entire suite of plots available as subplots.

        Args:
            signal (np.ndarray): Vector of output signal y_hat
            description (str): String description of plot, used in title
            results (np.ndarray): Vector of error results
            error_metric (str): String description of error_metric, used in title

        """
        # creating each individual plot as a subplot
        plt.figure(figsize=(14, 8))
        plt.title(f"Results of {self.algorithm} algorithm")
        plt.subplot(3, 1, 1)
        signal_plot = self.signal_plot(signal, description, subplot=True)
        plt.subplot(3, 1, 2)
        error_plot = self.error_plot(results, error_metric, subplot=True)

        # show the full plot
        plt.tight_layout()
        plt.show()
