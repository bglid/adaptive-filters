use std::collections::VecDeque;
use std::num::NonZeroUsize;

// TODO: make f64 generic

// The only way to get a single slice from a VecDeque is by calling make_contiguous().
// Calling it for every sample could get expensive, so instead we use this trait
// to provide a common interface for element-wise operations (e.g. dot product).
#[allow(
    clippy::len_without_is_empty,
    reason = "Should only be used with fixed-size containers"
)]
pub trait SampleView {
    fn len(&self) -> usize;
    fn get(&self, idx: usize) -> Option<&f64>;
}

struct SampleBuffer {
    samples: VecDeque<f64>,
    capacity: NonZeroUsize,
}
impl SampleBuffer {
    pub fn new(capacity: NonZeroUsize) -> Self {
        SampleBuffer {
            samples: std::iter::repeat_n(0.0, capacity.into()).collect(),
            capacity,
        }
    }

    pub fn push(&mut self, sample: f64) {
        if self.samples.len() == self.capacity.into() {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }
}
impl SampleView for SampleBuffer {
    fn len(&self) -> usize {
        self.samples.len()
    }
    fn get(&self, idx: usize) -> Option<&f64> {
        self.samples.get(idx)
    }
}

#[derive(Debug)]
pub struct Filter<A: Algorithm> {
    algorithm: A,
    weights: Vec<f64>,
    window_size: NonZeroUsize,
}
impl<A: Algorithm> Filter<A> {
    pub fn new(algorithm: A, window_size: NonZeroUsize) -> Self {
        let weights = Vec::with_capacity(window_size.into());
        // TODO: init weights

        Filter {
            algorithm,
            weights,
            window_size,
        }
    }

    #[allow(clippy::missing_errors_doc, reason = "TODO")]
    pub fn filter(&mut self, d: &[f64], x: &[f64]) -> Result<Vec<f64>, &'static str> {
        // truncate x
        let n_samples = std::cmp::min(x.len(), d.len());

        let mut cleaned_signal = Vec::<f64>::with_capacity(n_samples);

        let mut noise_ref_buffer = SampleBuffer::new(self.window_size);

        #[allow(clippy::indexing_slicing, reason = "Bounds checked by n_samples")]
        for n in 0..n_samples {
            noise_ref_buffer.push(x[n]);

            let noise_estimate = self.estimate_noise(&noise_ref_buffer)?;
            let error = d[n] - noise_estimate;

            cleaned_signal[n] = error;

            self.algorithm
                .update_step(&mut self.weights, &[error], &noise_ref_buffer);
        }

        Ok(cleaned_signal)
    }

    fn estimate_noise<T>(&self, x_n: &T) -> Result<f64, &'static str>
    where
        T: SampleView,
    {
        // TODO: replace with generic dot product
        if self.weights.len() != x_n.len() {
            return Err("Dot product operands must have the same length");
        }

        let mut res: f64 = 0.0;

        #[allow(
            clippy::unwrap_used,
            clippy::indexing_slicing,
            reason = "s_n and weights have same length, so i is always in bounds"
        )]
        for i in 0..self.weights.len() {
            res += self.weights[i] * x_n.get(i).unwrap();
        }

        Ok(res)
    }
}

pub trait Algorithm {
    fn update_step<T>(&self, w: &mut [f64], e_n: &[f64], x_n: &T)
    where
        T: SampleView;
}
