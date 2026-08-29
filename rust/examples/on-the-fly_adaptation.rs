#![allow(clippy::unwrap_used, reason = "Examples")]

use adaptive_filters::algorithms::LeastMeanSquares;
use adaptive_filters::filters::LMSFilter;
use adaptive_filters::types::{InputSignal, NoiseReference};

fn main() {
    // Sample inputs
    let input_signal = InputSignal::new(&[1.0, -2.5, 3.0]).unwrap();
    let noise_ref = NoiseReference::new(&[2.0, -1.2, -3.8]).unwrap();

    let lms_config = LeastMeanSquares { mu: 1.0 }; // The parameters used by the LMS filter
    let window_size = 1024; // How many samples we process at a time
    // Initialize the filter
    let mut lms = LMSFilter::new(lms_config, window_size).unwrap();

    // Adapt the filter to the inputs and get the iteratively cleaned signal.
    let _cleaned_signal = lms.adapt(&input_signal, &noise_ref).unwrap();
}
