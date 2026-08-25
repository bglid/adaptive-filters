use std::num::{NonZero, NonZeroUsize};
use std::ops::{Deref, DerefMut};

use rand_distr::{Distribution as _, Normal};

use crate::algorithms::Algorithm;
use crate::errors::{FilterError, FilterResult};
use crate::sample_buffer::SampleBuffer;

// TODO: make f64 generic

pub struct FilterBase<A: Algorithm> {
    algorithm: A,
    weights: FilterWeights,
    // window_size: NonZeroUsize,
}
impl<A: Algorithm> FilterBase<A> {
    #[allow(clippy::missing_panics_doc, reason = "Can't panic; See reason below")]
    pub fn new(algorithm: A, window_size: usize) -> Option<Self> {
        let window_size = NonZero::new(window_size)?;

        // The ? won't propagate a None because this function can only fail if std_dev is negative or infinity
        let weights = FilterWeights::new(window_size, 0.0, 0.5, 1e-4)?;

        Some(FilterBase {
            algorithm,
            weights,
            // window_size,
        })
    }

    // TODO: split API between fitting and applying the filter (fit() and process()/apply())
    #[allow(clippy::missing_errors_doc, reason = "TODO")]
    pub fn filter(
        &mut self,
        input_signal: &[f64],
        noise_reference: &[f64],
    ) -> FilterResult<Vec<f64>> {
        if noise_reference.is_empty() || input_signal.is_empty() {
            return Err(FilterError::EmptyInputArr);
        }

        if noise_reference.len() < input_signal.len() {
            return Err(FilterError::NoiseRefTooShort {
                input_len: input_signal.len(),
                noise_len: noise_reference.len(),
            });
        }

        let n_samples = input_signal.len();

        let mut cleaned_signal = Vec::<f64>::with_capacity(n_samples);

        // This ensures that estimate_noise() runs correctly because the buffer length matches the number of weights
        let mut noise_samples = SampleBuffer::new(&self.weights);

        #[allow(
            clippy::indexing_slicing,
            reason = "We set n_samples = input_signal.len() and checked that noise_reference.len() >= input_signal.len()"
        )]
        for n in 0..n_samples {
            noise_samples.push(noise_reference[n]);

            let noise_estimate = self.estimate_noise(&noise_samples);

            let error = Self::error(input_signal[n], noise_estimate);

            cleaned_signal.push(error);

            self.algorithm
                .update_step(&mut self.weights, error, &noise_samples);
        }

        Ok(cleaned_signal)
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

pub struct FilterWeights {
    weights: Box<[f64]>, // We use a boxed slice instead of a Vec to ensure length doesn't change
}
impl FilterWeights {
    pub fn new(
        window_size: NonZeroUsize,
        mean: f64,
        std_dev: f64,
        scaling_factor: f64,
    ) -> Option<Self> {
        let mut rng = rand::rng();

        let normal_dist = Normal::new(mean, std_dev).ok()?;

        let weights = normal_dist
            .sample_iter(&mut rng)
            .take(window_size.into())
            .map(|w| w * scaling_factor)
            .collect::<Vec<f64>>()
            .into_boxed_slice();

        Some(FilterWeights { weights })
    }

    // TODO: from_distribution() ?

    // mostly used for testing functions
    pub fn zeros(window_size: NonZeroUsize) -> Self {
        FilterWeights {
            weights: std::iter::repeat_n(0.0, window_size.into())
                .collect::<Vec<f64>>()
                .into_boxed_slice(),
        }
    }
}
impl Deref for FilterWeights {
    type Target = Box<[f64]>;
    fn deref(&self) -> &Self::Target {
        &self.weights
    }
}
impl DerefMut for FilterWeights {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.weights
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
        let mut filter = FilterBase::<TestAlgorithm>::new(TestAlgorithm {}, 3).unwrap();
        filter.weights = FilterWeights {
            weights: Box::new([1.0, -2.0, 0.5]),
        };
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
    fn filter_weights_update() {
        let mut filter = testing_filter();

        let weights_before = filter.weights.clone();

        let input = [5.0, 3.5, 2.6, -8.4];
        let noise = [3.0, 2.8, -1.7, 2.24];

        filter.filter(&input, &noise).unwrap();

        let same_weights = filter
            .weights
            .iter()
            .zip(weights_before.iter())
            .all(|(a, b)| approx_equal(*a, *b, 1e-6));
        assert!(!same_weights);
    }

    #[test]
    fn filter_weights_init() {
        const WINDOW_SIZE: usize = 1024;
        let weights =
            FilterWeights::new(NonZero::new(WINDOW_SIZE).unwrap(), 0.0, 0.5, 1e-4).unwrap();

        assert_eq!(WINDOW_SIZE, weights.len());
        assert!(!all_approx_equal(&weights, &[0.0; WINDOW_SIZE]));
    }

    #[test]
    fn filter_weights_zero() {
        const WINDOW_SIZE: usize = 1024;
        let weights = FilterWeights::zeros(NonZero::new(WINDOW_SIZE).unwrap());

        assert_eq!(WINDOW_SIZE, weights.len());
        assert!(all_approx_equal(&weights, &[0.0; WINDOW_SIZE]));
    }

    #[test]
    fn filter_reject_empty_input() {
        let mut filter = testing_filter();

        let input = [1.0, 2.0, 3.0];
        let noise = [4.0, 5.0, 6.0];

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
    fn filter_reject_shorter_noise_ref() {
        let mut filter = testing_filter();

        let input = [1.0, 2.0, 3.0];
        let noise = [4.0, 5.0];

        assert!(matches!(
            filter.filter(&input, &noise),
            Err(FilterError::NoiseRefTooShort {
                input_len: 3,
                noise_len: 2
            })
        ));
    }

    #[test]
    fn filter_allow_longer_noise_ref() {
        let mut filter = testing_filter();

        let input = [1.0, 2.0];
        let noise = [4.0, 5.0, 6.0];

        filter.filter(&input, &noise).unwrap();
    }

    #[test]
    fn filter_weight_len_invariant() {
        let mut filter = testing_filter();

        let before = filter.weights.len();

        let input = [1.0, 2.0, 3.0];
        let noise = [4.0, 5.0, 6.0];

        filter.filter(&input, &noise).unwrap();
        let after = filter.weights.len();

        assert_eq!(before, after);
    }
}
