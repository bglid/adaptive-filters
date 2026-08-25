use rand_distr::{Distribution as _, Normal};
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};

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
    use crate::test_utils::all_approx_equal;
    use std::num::NonZero;

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
}
