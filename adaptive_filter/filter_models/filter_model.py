# Class that contains filter model used by most adaptive filters

import numpy as np
from numpy.typing import NDArray


class FilterModel:
    def __init__(self, mu: float, filter_order: int) -> None:
        # consider adding p: order
        self.mu = mu  # step_rate
        self.N = filter_order  # filter window size
        # Algorithm type, defined by subclass algorithm
        self.algorithm = ""

    def noise_estimate(self, x_n: NDArray[np.float64]) -> np.float64:
        """Predict the noise estimate, given vector X[n], noise reference. Uses formula W^T[n]X[n].

        Args:
            x_n (NDArray[np.float64]): vector[n] of array X, the noise estimate

        Returns:
            np.float64: Predicted noise estimate output of the FIR filter and the noise reference
        """
        return np.dot(self.W, x_n)

    def error(self, d_n: float, noise_estimate: float) -> float:
        """Calculate the error, e[n] = d[n] - y[n], y[n] is output of W^T[n]X[n].

        Args:
            d_n (float): Desired sample at point n of array D, noisy input
            noise_estimate (float): The noise estimate product (y[n])

        Returns:
            float: error of (noisy) desired input[n] - noise estimate. Ideally, this should be the clean signal
        """
        return d_n - noise_estimate

    def update_step(self, e_n: float, x_n: NDArray[np.float64]) -> NDArray[np.float64]:
        """Update weights of W[n + 1], given the learning algorithm chosen.

        Args:
            e_n (float): Error sample at point n
            x_n (NDArray[np.float64]): Input vector n

        Returns:
            NDArray[np.float64]: Update step to self.W
        """
        return np.zeros(len(x_n))

    def filter(
        self,
        d: NDArray[np.float64],
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Iterate Adaptive filter alorithm and updates for length of input signal X.

        Args:
            d (NDArray[np.float64]):
                "Desired Signal", which in the ANC use-case is the noisy input signal.
            x (NDArray[np.float64]):
                Input reference matrix X, which in the ANC case is the noise reference.

        Returns:
            NDArray[np.float64]: "Clean output" The error signal of d - y.

        Raises:
            ValueError: If Signal dims are not compatible (1D)
        """

        # initializing our weights given X
        self.W = np.random.normal(0.0, 0.5, self.N)
        self.W *= 0.001  # setting weights close to zero

        # turning D and X into np arrays, if not already
        d = np.asarray(d).ravel()
        if d.ndim != 1:
            raise ValueError(f"Expected desired signal to be 1D, got shape: {d.shape}")

        x = np.asarray(x).ravel()
        if x.ndim != 1:
            raise ValueError(f"Expected input signal to be 1D, got shape: {x.shape}")

        # truncating x if it's longer
        if d.shape[0] < x.shape[0]:
            x = x[: d.shape[0]]

        # initializing the arrays to hold error and noise estimate
        noise_estimate = np.zeros(len(x))
        error = np.zeros(len(x))

        # creating a ciruclar buffer for the filter taps
        circ_buffer = np.zeros(self.N, dtype=float)

        for sample in range(len(x)):
            # using a circular buffer style window technique:
            circ_buffer = np.roll(circ_buffer, 1)
            # writer-pointer to add the most recent sample into the N buffer window
            circ_buffer[0] = x[sample]

            # getting the prediction y (noise estimate)
            noise_estimate[sample] = self.noise_estimate(circ_buffer)
            # getting the error e[sample] = d[sample] - y[sample]
            error[sample] = self.error(
                d_n=d[sample], noise_estimate=noise_estimate[sample]
            )

            # updating the weights
            self.W += self.update_step(e_n=error[sample], x_n=circ_buffer)

        return error
