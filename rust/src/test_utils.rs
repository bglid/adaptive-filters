#[allow(unused, reason = "Used in tests for other modules")]
pub fn approx_equal(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps
}
