use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::algorithms::LeastMeanSquares;
use crate::filters::LMSFilter as RustLMSFilter;

#[pyclass]
pub struct LMSFilter(RustLMSFilter);
#[pymethods]
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
}
