#![allow(clippy::unwrap_used, clippy::exhaustive_structs, reason = "Examples")]

use adaptive_filters::SampleBuffer;
use adaptive_filters::algorithms::Algorithm;
use adaptive_filters::filters::FilterBase;

// Create a struct to hold any required parameters or state
pub struct MyAlgorithm {
    pub alpha: f64,
}
// Implement the Algorithm trait so the algorithm can be used with FilterBase
impl Algorithm for MyAlgorithm {
    // This function is called every iteration during adaptation to update the weights
    fn update_step(&self, weights: &mut [f64], error: f64, noise_ref: &SampleBuffer) {
        for (w, x) in weights.iter_mut().zip(noise_ref.iter()) {
            *w += self.alpha * error * x;
        }
    }
}

fn main() {
    // Sample inputs
    let input_signal = [1.0, -2.5, 3.0];
    let noise_ref = [2.0, -1.2, -3.8];

    // Define the algorithm parameters
    let algorithm_cfg = MyAlgorithm { alpha: 1.0 };
    let window_size = 1024;

    // Instantiate the filter using FilterBase and our custom algorithm
    let mut filter = FilterBase::<MyAlgorithm>::new(algorithm_cfg, window_size).unwrap();

    // Adapt the filter using our algorithm's update rules
    let _output = filter.adapt(&input_signal, &noise_ref).unwrap();
}
