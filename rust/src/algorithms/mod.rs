use crate::types::{OutputSample, SampleBuffer};

mod lms;
pub use lms::LeastMeanSquares;

/// Trait used for implementing algorithms used in conjuction with `FilterBase`.
pub trait Algorithm {
    /// Updates the weights for the next time step based on the algorithm's update rules.
    /// This function is called every processing iteration by the filter during adapation.
    /// `error` is the cleaned sample from the current time step.
    /// `noise_ref` is the noise reference signal within the current processing window (the $k$ most recent samples).
    ///
    // DEV NOTE: Not 100 % sure about using newtypes here; It increases type safety internally,
    // but would require making them public...
    fn update_step(&self, weights: &mut [f64], error: OutputSample, noise_ref: &SampleBuffer);
}
