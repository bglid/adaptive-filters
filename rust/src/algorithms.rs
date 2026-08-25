use crate::filter_base::FilterBase;
use crate::sample_buffer::SampleBuffer;

pub trait Algorithm {
    // TODO: rename e_n and x_n
    fn update_step(&self, weights: &mut [f64], e_n: f64, x_n: &SampleBuffer);
}

pub type LMSFilter = FilterBase<LeastMeanSquares>;
pub struct LeastMeanSquares {
    mu: f64,
}
impl Algorithm for LeastMeanSquares {
    fn update_step(&self, weights: &mut [f64], e_n: f64, x_n: &SampleBuffer) {
        for (w, x) in weights.iter_mut().zip(x_n.iter()) {
            *w += self.mu * e_n * x;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;
    use crate::sample_buffer::SampleBuffer;
    use crate::test_utils::approx_equal;

    #[test]
    fn update_lms_1() {
        let lms = LeastMeanSquares { mu: 0.5 };
        let e_n = 2.0;
        let x_n = SampleBuffer::from(&[1.0, -1.0]).unwrap();
        let expected = [1.0, -1.0];
        let mut weights = [0.0, 0.0];

        lms.update_step(&mut weights, e_n, &x_n);

        let output_correct = weights
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| approx_equal(*a, *b, 1e-6));
        assert!(output_correct);
    }

    #[test]
    fn update_lms_2() {
        let lms = LeastMeanSquares { mu: 1.0 };
        let e_n = 1.0;
        let x_n = SampleBuffer::from(&[5.0, 2.0]).unwrap();
        let expected = [5.0, 2.0];
        let mut weights = [0.0, 0.0];

        lms.update_step(&mut weights, e_n, &x_n);

        let output_correct = weights
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| approx_equal(*a, *b, 1e-6));
        assert!(output_correct);
    }

    #[test]
    fn update_lms_3() {
        let lms = LeastMeanSquares { mu: -1.0 };
        let e_n = 1.0;
        let x_n = SampleBuffer::from(&[0.5, 0.25]).unwrap();
        let expected = [-0.5, -0.25];
        let mut weights = [0.0, 0.0];

        lms.update_step(&mut weights, e_n, &x_n);

        let output_correct = weights
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| approx_equal(*a, *b, 1e-6));
        assert!(output_correct);
    }
}
