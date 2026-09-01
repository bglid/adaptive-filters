use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use numpy::{PyArray1, PyReadonlyArray1};

use crate::algorithms::LeastMeanSquares;
use crate::filters::LMSFilter as RustLMSFilter;
use crate::types::{InputSignal, NoiseReference};

fn input_signal_from_pyarray<'a>(
    input_signal: &'a PyReadonlyArray1<f64>,
) -> PyResult<InputSignal<'a>> {
    // Because we're using slices in Rust, the input NumPy arrays need to be contiguous.
    // Strided slices like x[::2] or x[:, 0] are not allowed, and need to be made contiguous first.
    let input_signal = input_signal.as_slice().map_err(|_e| {
        PyValueError::new_err(
            "input_signal must be a contiguous NumPy array; use numpy.ascontiguousarray().",
        )
    })?;

    let input_signal =
        InputSignal::new(input_signal).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(input_signal)
}
fn noise_ref_from_pyarray<'a>(
    noise_ref: &'a PyReadonlyArray1<f64>,
) -> PyResult<NoiseReference<'a>> {
    // Because we're using slices in Rust, the input NumPy arrays need to be contiguous.
    // Strided slices like x[::2] or x[:, 0] are not allowed, and need to be made contiguous first.
    let noise_ref = noise_ref.as_slice().map_err(|_e| {
        PyValueError::new_err(
            "noise_ref must be a contiguous NumPy array; use numpy.ascontiguousarray().",
        )
    })?;
    let noise_ref =
        NoiseReference::new(noise_ref).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(noise_ref)
}

#[pyclass]
pub struct LMSFilter(RustLMSFilter);
#[pymethods]
#[allow(
    clippy::needless_pass_by_value,
    reason = "PyArrays must be passed by value"
)]
impl LMSFilter {
    #[new]
    fn new(mu: f64, window_size: usize) -> PyResult<Self> {
        match RustLMSFilter::new(LeastMeanSquares { mu }, window_size) {
            Some(filter) => Ok(Self(filter)),
            None => Err(PyValueError::new_err("window_size cannot be zero")),
        }
    }

    #[getter]
    fn window_size(&self) -> usize {
        self.0.window_size()
    }

    // TODO: weights() (+ check before/after in tests)

    fn adapt<'py>(
        &mut self,
        py: Python<'py>,
        input_signal: PyReadonlyArray1<f64>,
        noise_ref: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let input_signal = input_signal_from_pyarray(&input_signal)?;
        let noise_ref = noise_ref_from_pyarray(&noise_ref)?;

        let output_signal = self
            .0
            .adapt(&input_signal, &noise_ref)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(PyArray1::from_vec(py, output_signal))
    }

    fn filter<'py>(
        &self,
        py: Python<'py>,
        input_signal: PyReadonlyArray1<f64>,
        noise_ref: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let input_signal = input_signal_from_pyarray(&input_signal)?;
        let noise_ref = noise_ref_from_pyarray(&noise_ref)?;

        let output_signal = self
            .0
            .filter(&input_signal, &noise_ref)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(PyArray1::from_vec(py, output_signal))
    }
}

#[pymodule]
mod adaptive_filters {
    #[pymodule_export]
    use super::LMSFilter;
}
