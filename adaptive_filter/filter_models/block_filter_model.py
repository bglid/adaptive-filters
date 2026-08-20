# Class that contains filter model used by most adaptive filters
from typing import Any

import time
from collections import deque

import numpy as np

# from numpy.core.shape_base import block
from numpy.typing import NDArray


# SIMILAR to FilterModel, but with block-based processing logic added
class BlockFilterModel:
    def __init__(
        self,
        mu: float,
        filter_order: int,
        block_size: int,
    ) -> None:
        self.mu = mu
        self.N = filter_order
        # FFT length or P order for APA
        self.block_size = block_size
        self.hop_size = block_size // 2
        self.half_bins = self.block_size // 2 + 1
        self.eps = 1e-8
        # Algorithm type, defined by subclass algorithm
        self.algorithm = ""
        # # initializing weights
        # self.W = np.zeros(self.half_bins, dtype=np.complex128)

    def noise_estimate(self, x_n: NDArray[np.float64]) -> np.float64:
        """Predicts the noise estimate, given vector X[n], noise reference. Uses formula W^T[n]X[n]

        Args:
            x_n (NDArray[np.float64]): vector[n] of array X, the noise estimate

        Returns:
            np.float64: Predicted noise estimate output of the FIR filter and the noise reference
        """
        return np.dot(self.W, x_n)

    def error(self, d_n: float, noise_estimate: float) -> float:
        """Calculates the error, e[n] = d[n] - y[n], y[n] is output of W^T[n]X[n]

        Args:
            d_n (float): Desired sample at point n of array D, noisy input
            noise_estimate (float): The noise estimate product (y[n])

        Returns:
            float: error of (noisy) desired input[n] - noise estimate. Ideally, this should be the clean signal
        """
        return d_n - noise_estimate

    def update_step(
        self,
        e_n: float,
        x_n: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Updates weights of W[n + 1], given the learning algorithm chosen

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
        """Iterates Adaptive filter alorithm and updates for length of input signal X

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

        if d.shape[0] < x.shape[0]:
            x = x[: d.shape[0]]
            # assert x.shape == d.shape  # Double check

        # getting the number of samples from x len
        num_samples = len(x)

        # initializing the arrays to hold error and noise estimate
        noise_estimate = np.zeros(num_samples)
        error = np.zeros(num_samples)

        # creating a ciruclar buffer for the filter taps
        circ_buffer = np.zeros(self.N, dtype=float)
        # for buffering blocks
        x_buffer = deque(maxlen=self.block_size)
        error_buffer = deque(maxlen=self.block_size)

        for sample in range(num_samples):
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
            # Buffering blocks
            x_buffer.append(circ_buffer.copy())
            error_buffer.append(error[sample])

            # APA update
            if len(x_buffer) == (self.block_size):
                if self.algorithm not in ("FDLMS", "FDNLMS"):
                    x_block = np.stack(x_buffer, axis=1)
                    e_block = np.array(error_buffer)
                    self.W += self.update_step(e_n=e_block, x_n=x_block)

        return error


# for frequency domain block based processing
class FrequencyDomainAF(BlockFilterModel):
    def __init__(
        self,
        mu: float,
        filter_order: int,
        block_size: int,
    ) -> None:
        self.mu = mu
        self.N = filter_order
        self.block_size = block_size
        self.hop_size = block_size // 2
        # Init block base
        super().__init__(mu=mu, filter_order=filter_order, block_size=block_size)

        # Overriding with FD specs
        self.half_bins = self.block_size // 2 + 1
        self.eps = 1e-8
        self.algorithm = "FDAF"
        self.W = np.zeros(self.half_bins, dtype=np.complex128)

    # Setting the update step to include conj
    def update_step(self, e_f: NDArray[Any], x_f: NDArray[Any]) -> NDArray[Any]:
        """Update for FDAF.

        Args:
            e_f (NDArray[np.complex128]): Block error in the frequency domain.
            x_f (NDArray[np.complex128]): Block noise estimate in the frequency domain.

        Returns:
            NDArray[np.float64]: The weight update vector for FDAF.
        """
        return self.mu * np.multiply(np.conj(x_f), e_f)

    # FD specs
    def filter(
        self,
        d: NDArray[np.float64],
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Iterates Adaptive filter alorithm and updates for length of input signal X

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
        # turning D and X into np arrays, if not already
        d = np.asarray(d).ravel()
        if d.ndim != 1:
            raise ValueError(f"Expected desired signal to be 1D, got shape: {d.shape}")

        x = np.asarray(x).ravel()
        if x.ndim != 1:
            raise ValueError(f"Expected input signal to be 1D, got shape: {x.shape}")

        if d.shape[0] < x.shape[0]:
            x = x[: d.shape[0]]
            # assert x.shape == d.shape  # Double check

        # pad amount
        P = 1 << int(np.ceil(np.log2(self.N + self.hop_size - 1)))
        # padding the impulse response
        self.H = np.fft.rfft(np.pad(self.W, (0, P - self.N)), n=P)

        # getting the number of samples from x len
        num_samples = len(x)

        # initializing the arrays to hold error and noise estimate
        noise_estimate = np.zeros(num_samples)
        error = np.zeros(num_samples)

        # handling odd or '1' sample leftovers
        for sample in range(0, num_samples - self.hop_size + 1, self.hop_size):
            block = x[sample : sample + P]
            if len(block) < P:
                block = np.pad(block, (0, P - len(block)))

            # converting our block of input to the F domain
            x_f = np.fft.rfft(block)

            # getting our output via "convolution" in F domain
            y_f = self.H * x_f
            y_time = np.fft.irfft(y_f, n=P)

            # getting the overlapping
            valid = y_time[self.N - 1 : self.N - 1 + self.hop_size]
            noise_estimate[sample : sample + self.hop_size] = np.real(valid)

            # getting the overlapping error
            error[sample : sample + self.hop_size] = (
                d[sample : sample + self.hop_size]
                - noise_estimate[sample : sample + self.hop_size]
            )

            # now update H
            e_f = np.fft.rfft(
                np.pad(error[sample : sample + self.hop_size], (0, P - self.hop_size)),
                n=P,
            )

            self.H += self.update_step(e_f, x_f)

            # checking for leftover
            leftover_sample = (num_samples // self.hop_size) * self.hop_size
            if leftover_sample < num_samples:
                block = x[leftover_sample : leftover_sample + P]
                block = np.pad(block, (0, P - len(block)))

                # converting our block of input to the F domain
                x_f = np.fft.rfft(block)

                # getting our output via "convolution" in F domain
                y_f = self.H * x_f
                y_time = np.fft.irfft(y_f, n=P)

                # getting the overlapping
                valid = y_time[
                    self.N - 1 : self.N - 1 + (num_samples - leftover_sample)
                ]
                noise_estimate[leftover_sample:] = np.real(valid)

                # getting the overlapping error
                error[leftover_sample:] = d[leftover_sample:] - valid

                # now update H
                e_f = np.fft.rfft(
                    np.pad(valid, (0, P - len(valid))),
                    n=P,
                )
                self.H += self.update_step(e_f, x_f)

        return error
