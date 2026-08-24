use std::num::{NonZero, NonZeroUsize};

use rand_distr::{Distribution as _, Normal};

use crate::algorithms::Algorithm;
use crate::errors::{FilterError, FilterResult};
use crate::sample_buffer::{SampleBuffer, SampleView};

// TODO: make f64 generic

pub struct FilterBase<A: Algorithm> {
    algorithm: A,
    weights: Vec<f64>,
    window_size: NonZeroUsize,
}
impl<A: Algorithm> FilterBase<A> {
    #[allow(clippy::missing_panics_doc, reason = "Can't panic; See reason below")]
    pub fn new(algorithm: A, window_size: usize) -> Option<Self> {
        let window_size = NonZero::new(window_size)?;

        let weights = {
            let mut rng = rand::rng();

            #[allow(
                clippy::unwrap_used,
                reason = "Can only fail if std_dev is negative or infinity"
            )]
            let normal_dist = Normal::new(0.0, 0.5).unwrap();

            normal_dist
                .sample_iter(&mut rng)
                .take(window_size.into())
                .map(|w| w * 0.001) // Setting weights close to zero
                .collect()
        };

        Some(FilterBase {
            algorithm,
            weights,
            window_size,
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
        // TODO: replace window_size with weights.len() (then remove the check below)
        let mut noise_samples = SampleBuffer::new(self.window_size);

        // We make this check once here so that we don't have to do it for every sample,
        // relying on the invariant that weights.len() == buffer.len() == window size.
        // If this check fails, it means that this invariant isn't properly enforced elsewhere.
        if self.weights.len() != noise_samples.len() {
            return Err(FilterError::WeightSizeMismatch {
                weight_len: self.weights.len(),
                buffer_len: noise_samples.len(),
            });
        }

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

    // IMPORTANT: This function assumes that the invariant weights.len() == x_n.len() == window size is properly enforced.
    // Since none of these variables should change during the processing loop, we don't need to check it every iteration.
    // This avoids repeated unnecessary checks in the hot path.
    // It's up to the processing function to check that the lenghts are the same.
    #[inline]
    fn estimate_noise<T>(&self, x_n: &T) -> f64
    where
        T: SampleView,
    {
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

    use crate::test_utils::approx_equal;

    struct TestAlgorithm;
    impl Algorithm for TestAlgorithm {
        fn update_step<T>(&self, weights: &mut [f64], e_n: f64, x_n: &T)
        where
            T: SampleView,
        {
            for (i, w) in weights.iter_mut().enumerate() {
                *w += e_n * x_n.get(i).unwrap();
            }
        }
    }

    fn testing_filter() -> FilterBase<TestAlgorithm> {
        let mut filter = FilterBase::<TestAlgorithm>::new(TestAlgorithm {}, 3).unwrap();
        filter.weights = vec![1.0, -2.0, 0.5];
        filter
    }

    #[test]
    fn estimate_noise() {
        let filter = testing_filter();
        let x_n = SampleBuffer::from(&[2.0, 3.0, 4.0]).unwrap();

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
