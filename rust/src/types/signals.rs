use std::ops::Deref;

use crate::errors::{FilterError, FilterResult};

#[derive(Debug, Clone)]
pub struct InputSignal<'a>(&'a [f64]);
impl Deref for InputSignal<'_> {
    type Target = [f64];
    fn deref(&self) -> &Self::Target {
        self.0
    }
}
impl<'a> InputSignal<'a> {
    pub fn new(input_signal: &'a [f64]) -> FilterResult<Self> {
        if input_signal.is_empty() {
            Err(FilterError::EmptyInputArr)
        } else {
            Ok(InputSignal(input_signal))
        }
    }
    pub fn get_sample(&self, n: usize) -> Option<InputSample> {
        Some(InputSample(*self.get(n)?))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InputSample(pub f64);
impl Deref for InputSample {
    type Target = f64;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct NoiseReference<'a>(&'a [f64]);
impl Deref for NoiseReference<'_> {
    type Target = [f64];
    fn deref(&self) -> &Self::Target {
        self.0
    }
}
impl<'a> NoiseReference<'a> {
    pub fn new(noise_ref: &'a [f64]) -> FilterResult<Self> {
        if noise_ref.is_empty() {
            Err(FilterError::EmptyInputArr)
        } else {
            Ok(NoiseReference(noise_ref))
        }
    }

    pub fn get_sample(&self, n: usize) -> Option<NoiseSample> {
        Some(NoiseSample(*self.get(n)?))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NoiseSample(pub f64);
impl Deref for NoiseSample {
    type Target = f64;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NoiseEstimate(pub f64);
impl Deref for NoiseEstimate {
    type Target = f64;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct OutputSignal(Vec<f64>);
impl OutputSignal {
    pub fn new(input_signal: &InputSignal) -> Self {
        OutputSignal(Vec::with_capacity(input_signal.len()))
    }

    pub fn push(&mut self, error: OutputSample) {
        self.0.push(*error);
    }

    pub fn into_inner(self) -> Vec<f64> {
        self.0
    }
}
impl Deref for OutputSignal {
    type Target = Vec<f64>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(clippy::exhaustive_structs, reason = "Simple wrapper")]
pub struct OutputSample(pub f64);
impl Deref for OutputSample {
    type Target = f64;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
