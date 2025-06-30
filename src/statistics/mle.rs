/// This function provides the Maximum Likelihood Estimator (MLE)
/// for Normal distribution, given a slice of f64 data.
/// 
/// · Arguments: data, with f64 values; these will be the observed values.
///              unbiased, let the user to select if the calculation will be performed on a sample basis or a poblational basis.
/// · Returns: a tuple, with elements mean and standard deviation.
/// 
/// · Example:
/// ```
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let (mean, stddev) = mle_normal(&data);
/// assert!((mean - 3.0).abs() < 1e-6);
/// ```
/// 
pub fn mle_normal(data: &[f64], unbiased: bool) -> (f64, f64) {
    let n_usize = data.len();
    assert!(
        n_usize > 0,
        "It seems the data you introduced is empty. The data must not be empty."
    );

    let n_f64 = n_usize as f64;

    // First, compute the mean
    let mean = data.iter().sum::<f64>() / n_f64;

    // Then, compute the denominator depending on bias setting
    let denom = if unbiased && n_usize > 1 {
        (n_usize - 1) as f64
    } else {
        n_f64
    };

    // Finally, compute the variance and stddev
    let variance = data
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / denom;

    (mean, variance.sqrt())
}

/// This function provides the Maximum Likelihood Estimator (MLE)
/// for Binomial distribution, given a slice of f64 data.
/// 
/// · Arguments: data, with f64 values; these will be the observed values.
///              unbiased, let the user to select if the calculation will be performed on a sample basis or a poblational basis.
/// · Returns: a tuple, with elements mean and standard deviation.
/// 
/// · Example:
/// ```
/// let data = vec![1.0, 1.0, 0.0, 1.0, 1.0];
/// let (mean, stddev) = mle_binomial(&data);
/// assert!((mean - 3.0).abs() < 1e-6);
/// ```
/// 
pub fn mle_binomial(data: &[u32], n_trials: u32) -> f64 {
    assert!(n_trials > 0, "Consider that the number of trials must be greater than 0.");
    assert!(!data.is_empty(), "Consider that the data must not be empty.");
    assert!(
        data.iter().all(|&x| x <= n_trials),
        "The number of successes must be greater than the number of trials."
    );

    let total_successes: u32 = data.iter().sum();
    let total_trials = data.len() as f64 * n_trials as f64;

    total_successes as f64 / total_trials
}

/// This function provides the Maximum Likelihood Estimator (MLE)
/// for a Bernoulli distribution, given a slice of binary `u8` data (0 or 1).
///
/// · Arguments:
///   - `data`: a slice of `u8` values; each must be either 0 or 1.
/// 
/// · Returns:
///   - The estimated success probability `p`, as an `f64`.
///
/// · Example:
/// ```
/// let data = vec![1, 0, 1, 1, 0];
/// let p = mle_bernoulli(&data);
/// assert!((p - 0.6).abs() < 1e-6);
/// ```
/// 
pub fn mle_bernoulli(data: &[u8]) -> f64 {
    assert!(
        !data.is_empty(),
        "The data must not be empty."
    );
    assert!(
        data.iter().all(|&x| x == 0 || x == 1),
        "Bernoulli data must contain only 0s and 1s."
    );

    let sum: u32 = data.iter().map(|&x| x as u32).sum();
    sum as f64 / data.len() as f64
}

/// This function provides the Maximum Likelihood Estimator (MLE)
/// for an Exponential distribution, given a slice of positive `f64` data.
///
/// · Arguments:
///   - `data`: a slice of `f64` values; all values must be strictly positive.
/// 
/// · Returns:
///   - The estimated rate parameter `lambda`, as an `f64`.
///
/// · Example:
/// ```
/// let data = vec![1.0, 2.0, 0.5, 1.5];
/// let lambda = mle_exponential(&data);
/// assert!((lambda - 1.0).abs() < 1e-6);  // Mean = 1, so λ = 1
/// ```
/// 
pub fn mle_exponential(data: &[f64]) -> f64 {
    assert!(
        !data.is_empty(),
        "The data must not be empty."
    );
    assert!(
        data.iter().all(|&x| x > 0.0),
        "All values must be strictly positive for Exponential MLE."
    );

    let sum: f64 = data.iter().sum();
    data.len() as f64 / sum
}

/// This function provides the Maximum Likelihood Estimator (MLE)
/// for a Poisson distribution, given a slice of non-negative integer counts.
///
/// · Arguments:
///   - `data`: a slice of `u32` values; each must be a non-negative count.
/// 
/// · Returns:
///   - The estimated rate parameter `lambda`, as an `f64`.
///
/// · Example:
/// ```
/// let data = vec![2, 3, 4, 1];
/// let lambda = mle_poisson(&data);
/// assert!((lambda - 2.5).abs() < 1e-6);
/// ```
///
pub fn mle_poisson(data: &[u32]) -> f64 {
    assert!(
        !data.is_empty(),
        "The data must not be empty."
    );

    let sum: u32 = data.iter().sum();
    sum as f64 / data.len() as f64
}

/// This function provides the Maximum Likelihood Estimator (MLE)
/// for a Geometric distribution, given a slice of u32 data (failures before first success).
///
/// · Arguments:
///   - `data`: a slice of `u32` values; each represents number of failures before first success.
///
/// · Returns:
///   - The estimated success probability `p`, as an `f64`.
///
/// · Example:
/// ```
/// let data = vec![0, 1, 2, 1];
/// let p = mle_geometric(&data);
/// assert!((p - 0.4).abs() < 1e-6);
/// ```
///
pub fn mle_geometric(data: &[u32]) -> f64 {
    assert!(
        !data.is_empty(),
        "The data must not be empty."
    );

    let mean: f64 = data.iter().map(|&x| x as f64).sum::<f64>() / data.len() as f64;
    1.0 / (1.0 + mean)
}

/// This function provides the Maximum Likelihood Estimator (MLE)
/// for a Uniform(0, b) distribution, given a slice of positive f64 data.
///
/// · Arguments:
///   - `data`: a slice of `f64` values; all values must be in [0, b].
///
/// · Returns:
///   - The estimated upper bound `b`, as an `f64`.
///
/// · Example:
/// ```
/// let data = vec![0.5, 1.0, 0.8];
/// let b = mle_uniform(&data);
/// assert!((b - 1.0).abs() < 1e-6);
/// ```
///
pub fn mle_uniform(data: &[f64]) -> f64 {
    assert!(
        !data.is_empty(),
        "The data must not be empty."
    );

    *data.iter().fold(data.first().unwrap(), |max, &x| if x > *max { &x } else { max })
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mle_normal_biased() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, stddev) = mle_normal(&data, false); // biased
        assert!((mean - 3.0).abs() < 1e-6);
        assert!((stddev - 1.414213562).abs() < 1e-6); // sqrt(2.0)
    }

    #[test]
    fn test_mle_normal_unbiased() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, stddev) = mle_normal(&data, true); // unbiased
        assert!((mean - 3.0).abs() < 1e-6);
        assert!((stddev - 1.581138830).abs() < 1e-6); // sqrt(2.5)
    }

    #[test]
    #[should_panic(expected = "empty")]
    fn test_mle_normal_empty_slice() {
        let data: Vec<f64> = vec![];
        let _ = mle_normal(&data, false);
    }

    #[test]
    fn test_mle_normal_single_element() {
        let data = vec![42.0];
        let (mean, stddev) = mle_normal(&data, false);
        assert_eq!(mean, 42.0);
        assert_eq!(stddev, 0.0);
    }
    #[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mle_bernoulli() {
        let data = vec![1, 0, 1, 1, 0];
        let p = mle_bernoulli(&data);
        assert!((p - 0.6).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "must not be empty")]
    fn test_mle_bernoulli_empty() {
        let data: Vec<u8> = vec![];
        let _ = mle_bernoulli(&data);
    }

    #[test]
    #[should_panic(expected = "must contain only 0s and 1s")]
    fn test_mle_bernoulli_invalid_data() {
        let data = vec![1, 2, 0];
        let _ = mle_bernoulli(&data);
    }
}
#[test]
fn test_mle_exponential() {
    let data = vec![1.0, 2.0, 0.5, 1.5];
    let lambda = mle_exponential(&data);
    assert!((lambda - 1.0).abs() < 1e-6);  // Mean = 1.0 → λ = 1.0
}

#[test]
#[should_panic(expected = "must not be empty")]
fn test_mle_exponential_empty() {
    let data: Vec<f64> = vec![];
    let _ = mle_exponential(&data);
}

#[test]
#[should_panic(expected = "strictly positive")]
fn test_mle_exponential_invalid_data() {
    let data = vec![1.0, 0.0, 2.0];
    let _ = mle_exponential(&data);
}
#[test]
fn test_mle_poisson() {
    let data = vec![2, 3, 4, 1];
    let lambda = mle_poisson(&data);
    assert!((lambda - 2.5).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "must not be empty")]
fn test_mle_poisson_empty() {
    let data: Vec<u32> = vec![];
    let _ = mle_poisson(&data);
}


}

#[test]
fn test_mle_geometric_basic() {
    let data = vec![0, 1, 2, 1]; // mean = 1.0 → p = 1 / (1 + 1) = 0.5
    let p = mle_geometric(&data);
    assert!((p - 0.5).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "must not be empty")]
fn test_mle_geometric_empty() {
    let data: Vec<u32> = vec![];
    let _ = mle_geometric(&data);
}

#[test]
fn test_mle_geometric_zero_only() {
    let data = vec![0, 0, 0, 0];
    let p = mle_geometric(&data);
    assert!((p - 1.0).abs() < 1e-6); // no failures → p = 1
}

#[test]
fn test_mle_uniform_basic() {
    let data = vec![0.5, 1.0, 0.8];
    let b = mle_uniform(&data);
    assert!((b - 1.0).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "must not be empty")]
fn test_mle_uniform_empty() {
    let data: Vec<f64> = vec![];
    let _ = mle_uniform(&data);
}

#[test]
fn test_mle_uniform_identical_values() {
    let data = vec![2.0, 2.0, 2.0];
    let b = mle_uniform(&data);
    assert!((b - 2.0).abs() < 1e-6);
}
