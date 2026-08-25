mod filter_base;
pub use filter_base::FilterBase;

mod weights;
pub use weights::FilterWeights; // TODO: decide whether this should be accessible outside the crate

mod sample_buffer;
pub use sample_buffer::SampleBuffer;

use crate::algorithms::LeastMeanSquares;
pub type LMSFilter = FilterBase<LeastMeanSquares>;
