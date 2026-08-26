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
pub struct InputSample(f64);
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
pub struct NoiseSample(f64);
impl Deref for NoiseSample {
    type Target = f64;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
