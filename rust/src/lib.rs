pub mod algorithms;
mod error;
pub use error::{FilterError, FilterResult};
pub mod filters;

mod types;
// TODO: decide how and which types to publically export
pub use types::OutputSample;
pub use types::SampleBuffer; // has to be exported because it's used by the Algorithm trait, which is also public // has to be exported because it's used by the Algorithm trait, which is also public

mod test_utils;
