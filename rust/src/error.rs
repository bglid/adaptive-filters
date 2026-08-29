use std::error;
use std::fmt::Display;
use std::result;

pub type Result<T> = result::Result<T, Error>;

#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    EmptyInputArr,
    NoiseRefTooShort {
        input_len: usize,
        noise_len: usize,
    },
    WeightSizeMismatch {
        weight_len: usize,
        buffer_len: usize,
    },
}
impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::EmptyInputArr => write!(f, "empty input array."),
            Self::NoiseRefTooShort {
                input_len,
                noise_len,
            } => write!(
                f,
                "length of the noise reference ({noise_len}) must be equal to or greater than that of the input signal ({input_len})."
            ),
            Self::WeightSizeMismatch {
                weight_len,
                buffer_len,
            } => write!(
                f,
                "number of weights ({weight_len}) and samples ({buffer_len}) don't match."
            ),
        }
    }
}
impl error::Error for Error {}
