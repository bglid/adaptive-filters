use std::fmt::Display;

pub type FilterResult<T> = Result<T, FilterError>;

#[derive(Debug)]
#[non_exhaustive]
pub enum FilterError {
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
impl Display for FilterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::EmptyInputArr => write!(f, "Empty input array."),
            Self::NoiseRefTooShort {
                input_len,
                noise_len,
            } => write!(
                f,
                "Length of the noise reference ({noise_len}) must equal to or greater than that of the input signal ({input_len})."
            ),
            Self::WeightSizeMismatch {
                weight_len,
                buffer_len,
            } => write!(
                f,
                "Number of weights ({weight_len}) and samples ({buffer_len}) don't match."
            ),
        }
    }
}
