import numpy as np
from numpy.typing import NDArray

from adaptive_filter.filter_models.fd_filter_model import FrequencyDomainAF


class FDNLMS(FrequencyDomainAF):
    def __init__(self, mu: float, n: int, block_size: int) -> None:
        # initializing BlockFilterModel
        super().__init__(mu=mu, filter_order=n, block_size=block_size)
        self.algorithm = "FD_NLMS"
        self.eps = 1e-6
        self.p = 0.0

    # updating the update step for FD_NLMS algorithm
    def update_step(
        self, e_n: NDArray[np.complex128], x_n: NDArray[np.complex128]
    ) -> NDArray[np.float64]:
        """Update for FDAF: FD NLMS algorithm.

        Args:
            e_n (NDArray[np.complex128]): Block error in the frequency domain.
            x_n (NDArray[np.complex128]): Block noise estimate in the frequency domain.

        Returns:
            NDArray[np.float64]: The weight update vector for FDAF.
        """
        # getting the normalizing denom
        self.p = self.eps + np.sum(np.abs(x_n) ** 2)
        return self.mu * np.conj(x_n) * e_n / self.p
