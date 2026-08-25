use std::collections::VecDeque;
use std::num::{NonZero, NonZeroUsize};

use crate::filter_base::FilterWeights;

// Fixed-size ring buffer for processing samples.
// Functions must ensure that samples.len() is the same before and after function calls
// to enforce the invariant weights.len() == buffer.len() == window_size
pub struct SampleBuffer {
    samples: VecDeque<f64>,
    capacity: NonZeroUsize,
}
impl SampleBuffer {
    // We get the capacity directly from the weights to assure the invariant is enforced
    pub fn new(weights: &FilterWeights) -> Self {
        SampleBuffer {
            samples: std::iter::repeat_n(0.0, weights.len()).collect(),
            #[allow(clippy::unwrap_used, reason = "weights.len() is guaranteed non-zero")]
            capacity: NonZero::new(weights.len()).unwrap(),
        }
    }

    #[allow(
        unused,
        reason = "Used by some tests, and may be useful in the future, e.g. for block processing"
    )]
    // TODO: remove this because it creates a buffer of arbitrary length
    pub fn from(arr: &[f64]) -> Option<Self> {
        let capacity = NonZero::new(arr.len())?;
        let buff = SampleBuffer {
            samples: arr.iter().copied().collect(),
            capacity,
        };
        Some(buff)
    }

    pub fn push(&mut self, sample: f64) {
        if self.samples.len() == self.capacity.into() {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    pub fn get(&self, idx: usize) -> Option<&f64> {
        self.samples.get(idx)
    }

    pub fn len(&self) -> usize {
        self.capacity.into()
    }

    pub fn iter(&self) -> SampleIter<'_> {
        SampleIter {
            buffer: self,
            next_idx: 0,
        }
    }
}

pub struct SampleIter<'a> {
    buffer: &'a SampleBuffer,
    next_idx: usize,
}
impl<'a> Iterator for SampleIter<'a> {
    type Item = &'a f64;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.buffer.get(self.next_idx);
        self.next_idx += 1;
        item
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;
    use crate::test_utils::approx_equal;
    use std::num::NonZero;

    fn same_elements(buffer: &SampleBuffer, arr: &[f64]) -> bool {
        if buffer.len() != arr.len() {
            return false;
        }

        for (i, elem) in arr.iter().enumerate().take(buffer.len()) {
            if !approx_equal(*buffer.get(i).unwrap(), *elem, 1e-6) {
                return false;
            }
        }

        true
    }

    #[test]
    fn init_to_zero() {
        let weights = FilterWeights::new(NonZero::new(3).unwrap(), 0.0, 0.5, 1e-4).unwrap();
        let buffer = SampleBuffer::new(&weights);

        assert!(same_elements(&buffer, &[0_f64; 3]));
    }

    #[test]
    fn push() {
        let mut buffer = SampleBuffer::from(&[0.0; 3]).unwrap();

        buffer.push(1.0);
        assert_eq!(buffer.len(), 3);
        assert!(same_elements(&buffer, &[0.0, 0.0, 1.0]));

        buffer.push(2.0);
        assert_eq!(buffer.len(), 3);
        assert!(same_elements(&buffer, &[0.0, 1.0, 2.0]));
    }

    #[test]
    fn buffer_size_invariant() {
        let mut buffer = SampleBuffer::from(&[0.0; 3]).unwrap();

        buffer.push(1.0);
        buffer.push(2.0);
        buffer.push(3.0);

        assert_eq!(buffer.len(), 3);
        assert!(same_elements(&buffer, &[1.0, 2.0, 3.0]));

        buffer.push(4.0);

        assert_eq!(buffer.len(), 3);
        assert!(same_elements(&buffer, &[2.0, 3.0, 4.0]));
    }

    #[test]
    fn get() {
        let mut buffer = SampleBuffer::from(&[0.0; 3]).unwrap();

        buffer.push(1.0);
        buffer.push(2.0);
        buffer.push(3.0);

        assert_eq!(buffer.get(0), Some(&1.0));
        assert_eq!(buffer.get(1), Some(&2.0));
        assert_eq!(buffer.get(2), Some(&3.0));
        assert_eq!(buffer.get(3), None);
    }
}
