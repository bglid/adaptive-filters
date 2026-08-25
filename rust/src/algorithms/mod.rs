use crate::filters::SampleBuffer;

mod lms;
pub use lms::LeastMeanSquares;

pub trait Algorithm {
    // TODO: rename e_n and x_n
    fn update_step(&self, weights: &mut [f64], e_n: f64, x_n: &SampleBuffer);
}
