use std::num::{NonZero, NonZeroUsize};

use crate::algorithms::Algorithm;
use crate::errors::{FilterError, FilterResult};
use crate::types::{FilterWeights, SampleBuffer};

// TODO: make f64 generic

pub struct FilterBase<A: Algorithm> {
    algorithm: A,
    weights: FilterWeights,
    window_size: NonZeroUsize,
}
impl<A: Algorithm> FilterBase<A> {
    pub fn new(algorithm: A, window_size: usize) -> Option<Self> {
        let window_size = NonZero::new(window_size)?;

        let weights = FilterWeights::new(window_size, 0.0, 0.5, 1e-4)?;

        Some(FilterBase {
            algorithm,
            weights,
            window_size,
        })
    }

    pub fn window_size(&self) -> usize {
        self.window_size.into()
    }

    #[allow(clippy::missing_errors_doc, reason = "TODO")]
    pub fn adapt(&mut self, input_signal: &[f64], noise_ref: &[f64]) -> FilterResult<Vec<f64>> {
        Self::check_signal_lengths(input_signal, noise_ref)?;

        let n_samples = input_signal.len();

        let mut cleaned_signal = Vec::<f64>::with_capacity(n_samples);
        let mut noise_ref_buffer = SampleBuffer::new(&self.weights);

        for n in 0..n_samples {
            // We set n_samples = input_signal.len() and called check_signal_lengths() (putting this in a comment so fmt doesn't split lines)
            #[allow(clippy::indexing_slicing, reason = "Bounds checked")]
            let error = self.process_sample(input_signal[n], noise_ref[n], &mut noise_ref_buffer);

            cleaned_signal.push(error);

            self.algorithm
                .update_step(&mut self.weights, error, &noise_ref_buffer);
        }

        Ok(cleaned_signal)
    }

    #[allow(clippy::missing_errors_doc, reason = "TODO")]
    pub fn filter(&self, input_signal: &[f64], noise_ref: &[f64]) -> FilterResult<Vec<f64>> {
        Self::check_signal_lengths(input_signal, noise_ref)?;

        let n_samples = input_signal.len();

        let mut cleaned_signal = Vec::<f64>::with_capacity(n_samples);
        let mut noise_ref_buffer = SampleBuffer::new(&self.weights);

        for n in 0..n_samples {
            // We set n_samples = input_signal.len() and called check_signal_lengths()
            #[allow(clippy::indexing_slicing, reason = "Bounds checked")]
            let error = self.process_sample(input_signal[n], noise_ref[n], &mut noise_ref_buffer);

            cleaned_signal.push(error);
        }

        Ok(cleaned_signal)
    }

    #[inline]
    fn process_sample(
        &self,
        input_sample: f64,
        noise_sample: f64,
        noise_ref_buffer: &mut SampleBuffer,
    ) -> f64 {
        noise_ref_buffer.push(noise_sample);

        let noise_estimate = self.estimate_noise(noise_ref_buffer);

        Self::error(input_sample, noise_estimate)
    }

    fn check_signal_lengths(input_signal: &[f64], noise_ref: &[f64]) -> FilterResult<()> {
        if noise_ref.is_empty() || input_signal.is_empty() {
            Err(FilterError::EmptyInputArr)
        } else if noise_ref.len() < input_signal.len() {
            Err(FilterError::NoiseRefTooShort {
                input_len: input_signal.len(),
                noise_len: noise_ref.len(),
            })
        } else {
            Ok(())
        }
    }

    #[inline]
    fn estimate_noise(&self, x_n: &SampleBuffer) -> f64 {
        // SampleBuffer is initiated with the same length as weights, therefore we don't need to check
        self.weights
            .iter()
            .zip(x_n.iter())
            .map(|(w, x)| w * x)
            .sum()
    }

    #[inline]
    fn error(input_sample: f64, noise_estimate: f64) -> f64 {
        input_sample - noise_estimate
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;

    use crate::test_utils::{all_approx_equal, approx_equal, sample_buffer_from};

    struct TestAlgorithm;
    impl Algorithm for TestAlgorithm {
        fn update_step(&self, weights: &mut [f64], e_n: f64, x_n: &SampleBuffer) {
            for (i, w) in weights.iter_mut().enumerate() {
                *w += e_n * x_n.get(i).unwrap();
            }
        }
    }

    fn testing_filter() -> FilterBase<TestAlgorithm> {
        let window_size = 3;
        let weights = [1.0, -2.0, 0.5];

        let mut filter = FilterBase::<TestAlgorithm>::new(TestAlgorithm {}, window_size).unwrap();
        for (i, val) in weights.iter().enumerate() {
            filter.weights[i] = *val;
        }
        filter
    }

    #[test]
    fn estimate_noise() {
        let filter = testing_filter();

        let x_n = sample_buffer_from(&[2.0, 3.0, 4.0]);

        let res = filter.estimate_noise(&x_n);
        assert!(approx_equal(res, -2.0, 1e-6));
    }

    #[test]
    fn error() {
        assert!(approx_equal(
            FilterBase::<TestAlgorithm>::error(5.0, 3.5),
            1.5,
            1e-6
        ));
    }

    #[test]
    fn adapt_weights_update() {
        let mut filter = testing_filter();

        let weights_before = filter.weights.clone();

        let input = [5.0, 3.5, 2.6, -8.4];
        let noise = [3.0, 2.8, -1.7, 2.24];

        filter.adapt(&input, &noise).unwrap();

        assert!(!all_approx_equal(
            filter.weights.iter(),
            weights_before.iter()
        ));
    }

    #[test]
    fn filter_weights_dont_update() {
        let filter = testing_filter();

        let weights_before = filter.weights.clone();

        let input = [5.0, 3.5, 2.6, -8.4];
        let noise = [3.0, 2.8, -1.7, 2.24];

        filter.filter(&input, &noise).unwrap();

        assert!(all_approx_equal(
            filter.weights.iter(),
            weights_before.iter()
        ));
    }

    #[test]
    fn adapt_weights_len_invariant() {
        let mut filter = testing_filter();

        let before = filter.weights.len();

        let input = [1.0, 2.0, 3.0];
        let noise = [4.0, 5.0, 6.0];

        filter.adapt(&input, &noise).unwrap();
        let after = filter.weights.len();

        assert_eq!(before, after);
    }

    #[test]
    fn reject_empty_input() {
        let mut filter = testing_filter();

        let input = [1.0, 2.0, 3.0];
        let noise = [4.0, 5.0, 6.0];

        filter.adapt(&input, &noise).unwrap();

        assert!(matches!(
            filter.adapt(&input, &[]),
            Err(FilterError::EmptyInputArr)
        ));
        assert!(matches!(
            filter.adapt(&[], &noise),
            Err(FilterError::EmptyInputArr)
        ));
        assert!(matches!(
            filter.adapt(&[], &[]),
            Err(FilterError::EmptyInputArr)
        ));

        filter.filter(&input, &noise).unwrap();

        assert!(matches!(
            filter.filter(&input, &[]),
            Err(FilterError::EmptyInputArr)
        ));
        assert!(matches!(
            filter.filter(&[], &noise),
            Err(FilterError::EmptyInputArr)
        ));
        assert!(matches!(
            filter.filter(&[], &[]),
            Err(FilterError::EmptyInputArr)
        ));
    }

    #[test]
    fn reject_shorter_noise_ref() {
        let mut filter = testing_filter();

        let input = [1.0, 2.0, 3.0];
        let noise = [4.0, 5.0];

        assert!(matches!(
            filter.adapt(&input, &noise),
            Err(FilterError::NoiseRefTooShort {
                input_len: 3,
                noise_len: 2
            })
        ));

        assert!(matches!(
            filter.filter(&input, &noise),
            Err(FilterError::NoiseRefTooShort {
                input_len: 3,
                noise_len: 2
            })
        ));
    }

    #[test]
    fn allow_longer_noise_ref() {
        let mut filter = testing_filter();

        let input = [1.0, 2.0];
        let noise = [4.0, 5.0, 6.0];

        filter.adapt(&input, &noise).unwrap();
        filter.filter(&input, &noise).unwrap();
    }
}
