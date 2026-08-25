#![allow(unused, reason = "Used in tests for other modules")]
#![allow(
    clippy::unwrap_used,
    clippy::unwrap_in_result,
    reason = "Only used in tests"
)]

use std::num::NonZero;

use crate::filter_base::FilterWeights;
use crate::sample_buffer::SampleBuffer;

pub fn approx_equal(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps
}

pub fn all_approx_equal(a: &[f64], b: &[f64]) -> bool {
    if a.len() == b.len() {
        a.iter()
            .zip(b.iter())
            .all(|(x, y)| approx_equal(*x, *y, 1e-6))
    } else {
        false
    }
}

pub fn sample_buffer_from(arr: &[f64]) -> SampleBuffer {
    let weights = FilterWeights::zeros(NonZero::new(arr.len()).unwrap());
    let mut buffer = SampleBuffer::new(&weights);

    for val in arr {
        buffer.push(*val);
    }

    buffer
}
