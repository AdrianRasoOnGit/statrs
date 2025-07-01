use crate::distribution::*;
use crate::statistics::statistics::Statistics;
use crate::statistics::traits::Distribution;

/// Trait for Maximum Likelihood Estimation (MLE)
///
/// Implement this trait for distributions that can estimate their parameters from data
/// using the method of Maximum Likelihood. The associated types define what the input
/// data looks like (`Data`) and what is returned on success (`Output`).
///
/// The `mle` method takes a reference to the input data and returns a result with either
/// the estimated distribution or a static error string.
pub trait Mle<'a> {
    type Data: 'a;
    type Output;

    fn mle(data: &Self::Data) -> Result<Self::Output, &'static str>;
}

/// Implements MLE for the Normal distribution.
///
/// Estimates the mean and standard deviation of a normal distribution given
/// a slice of `f64` values and a flag for unbiased variance.
///
/// # Parameters
/// - `data.0`: Observed samples (`&[f64]`).
/// - `data.1`: Boolean indicating unbiased (`true`) or biased (`false`) variance estimate.
///
/// # Errors
/// - Returns "empty input data" if the sample is empty.
/// - Returns "invalid parameters" if the variance is zero or not finite.
///
/// # Example
/// ```
/// use statrs::distribution::{Normal, Mle};
/// let data = [1.0, 2.0, 3.0];
/// let dist = Normal::mle(&(&data, true)).unwrap();
/// assert!((dist.mean() - 2.0).abs() < 1e-12);
/// ```
impl<'a> Mle<'a> for Normal {
    type Data = (&'a [f64], bool);
    type Output = Normal;

    fn mle(data: &Self::Data) -> Result<Self::Output, &'static str> {
        let (values, unbiased) = *data;
        if values.is_empty() {
            return Err("empty input data");
        }

        let mean = values.mean();
        let variance = if unbiased {
            values.variance()
        } else {
            values.population_variance()
        };

        if variance <= 0.0 || !variance.is_finite() {
            return Err("invalid parameters");
        }

        Normal::new(mean, variance.sqrt()).map_err(|_| "invalid parameters")
    }
}


/// Implements MLE for the Binomial distribution.
///
/// Estimates the success probability `p` from observed success counts and fixed trial count.
///
/// # Parameters
/// - `data.0`: Slice of observed successes (`&[u32]`).
/// - `data.1`: Number of trials per observation (`u64`).
///
/// # Errors
/// - Returns `"empty input data"` if the sample is empty.
/// - Returns `"invalid parameters"` if any count exceeds `n_trials` or `n_trials` is zero.
///
/// # Example
/// ```
/// use statrs::distribution::{Binomial, Mle};
/// let data = [2, 3, 4];
/// let dist = Binomial::mle(&(&data, 5)).unwrap();
/// assert_eq!(dist.n(), 5);
/// ```
impl<'a> Mle<'a> for Binomial {
    type Data = (&'a [u32], u64);
    type Output = Binomial;

    fn mle(data: &Self::Data) -> Result<Self::Output, &'static str> {
        let (values, n_trials) = *data;
        if values.is_empty() {
            return Err("empty input data");
        }
        if n_trials == 0 || values.iter().any(|&x| x as u64 > n_trials) {
            return Err("invalid parameters");
        }

        let total_successes: u64 = values.iter().map(|&x| x as u64).sum();
        let total_trials = (values.len() as u64) * n_trials;
        let p = total_successes as f64 / total_trials as f64;

        Binomial::new(p, n_trials).map_err(|_| "invalid parameters")
    }
}

/// Implements MLE for the Bernoulli distribution.
///
/// Estimates success probability `p` from a slice of 0s and 1s.
///
/// # Parameters
/// - `data`: Slice of `u8` values, must be only 0 or 1.
///
/// # Errors
/// - Returns `"invalid parameters"` if input is empty or contains values other than 0 or 1.
///
/// # Example
/// ```
/// use statrs::distribution::{Bernoulli, Mle};
/// let data = [1, 0, 1];
/// let dist = Bernoulli::mle(&data).unwrap();
/// assert!((dist.p() - 0.6667).abs() < 1e-4);
/// ```
impl<'a> Mle<'a> for Bernoulli {
    type Data = &'a [u8];
    type Output = Bernoulli;

    fn mle(data: &Self::Data) -> Result<Self::Output, &'static str> {
        if data.is_empty() || data.iter().any(|&x| x > 1) {
            return Err("invalid parameters");
        }

        let p = data.iter().map(|&x| x as f64).sum::<f64>() / data.len() as f64;
        Bernoulli::new(p).map_err(|_| "invalid parameters")
    }
}

/// Implements MLE for the Exponential distribution.
///
/// Estimates the rate `λ` as the reciprocal of the sample mean.
///
/// # Parameters
/// - `data`: Slice of positive `f64` values.
///
/// # Errors
/// - Returns `"invalid parameters"` if data is empty or contains non-positive values.
///
/// # Example
/// ```
/// use statrs::distribution::{Exp, Mle};
/// let data = [1.0, 2.0];
/// let dist = Exp::mle(&data).unwrap();
/// assert!((dist.rate() - 2.0 / 3.0).abs() < 1e-10);
/// ```
impl<'a> Mle<'a> for Exp {
    type Data = &'a [f64];
    type Output = Exp;

    fn mle(data: &Self::Data) -> Result<Self::Output, &'static str> {
        if data.is_empty() || data.iter().any(|&x| x <= 0.0) {
            return Err("invalid parameters");
        }

        let lambda = data.len() as f64 / data.iter().sum::<f64>();
        Exp::new(lambda).map_err(|_| "invalid parameters")
    }
}

/// Implements MLE for the Poisson distribution.
///
/// Estimates the rate `λ` as the sample mean.
///
/// # Parameters
/// - `data`: Slice of non-negative integers.
///
/// # Errors
/// - Returns `"empty input data"` if data is empty.
///
/// # Example
/// ```
/// use statrs::distribution::{Poisson, Mle};
/// let data = [1, 2, 3];
/// let dist = Poisson::mle(&data).unwrap();
/// assert!((dist.lambda() - 2.0).abs() < 1e-12);
/// ```
impl<'a> Mle<'a> for Poisson {
    type Data = &'a [u32];
    type Output = Poisson;

    fn mle(data: &Self::Data) -> Result<Self::Output, &'static str> {
        if data.is_empty() {
            return Err("empty input data");
        }

        let lambda = data.iter().map(|&x| x as f64).sum::<f64>() / data.len() as f64;
        Poisson::new(lambda).map_err(|_| "invalid parameters")
    }
}

/// Implements MLE for the Geometric distribution.
///
/// Estimates success probability `p` using `p = 1 / (1 + mean)`.
///
/// # Parameters
/// - `data`: Slice of `u32` values.
///
/// # Errors
/// - Returns `"empty input data"` if data is empty.
///
/// # Example
/// ```
/// use statrs::distribution::{Geometric, Mle};
/// let data = [0, 1, 2];
/// let dist = Geometric::mle(&data).unwrap();
/// assert!((dist.p() - 0.5).abs() < 1e-10);
/// ```
impl<'a> Mle<'a> for Geometric {
    type Data = &'a [u32];
    type Output = Geometric;

    fn mle(data: &Self::Data) -> Result<Self::Output, &'static str> {
        if data.is_empty() {
            return Err("empty input data");
        }

        let mean = data.iter().map(|&x| x as f64).sum::<f64>() / data.len() as f64;
        let p = 1.0 / (1.0 + mean);

        Geometric::new(p).map_err(|_| "invalid parameters")
    }
}

/// Implements MLE for the Uniform distribution.
///
/// Assumes support is `[0, θ]`, and estimates `θ` as the maximum of the data.
///
/// # Parameters
/// - `data`: Slice of positive `f64` values.
///
/// # Errors
/// - Returns `"empty input data"` if data is empty.
/// - Returns `"invalid parameters"` if max value is non-positive.
///
/// # Example
/// ```
/// use statrs::distribution::{Uniform, Mle};
/// let data = [1.0, 2.0, 3.0];
/// let dist = Uniform::mle(&data).unwrap();
/// assert_eq!(dist.min(), 0.0);
/// ```
impl<'a> Mle<'a> for Uniform {
    type Data = &'a [f64];
    type Output = Uniform;

    fn mle(data: &Self::Data) -> Result<Self::Output, &'static str> {
        if data.is_empty() {
            return Err("empty input data");
        }

        let &max = data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        if max <= 0.0 {
            return Err("invalid parameters");
        }

        Uniform::new(0.0, max).map_err(|_| "invalid parameters")
    }
}

/// Generic helper for MLE using the [`Mle`] trait.
///
/// Useful to avoid writing the distribution name twice. Especially handy
/// in generic code where the distribution type is a parameter.
///
/// # Example
/// ```
/// use statrs::distribution::{Normal, mle_for};
/// let data = [1.0, 2.0, 3.0];
/// let dist = mle_for::<Normal>(&(&data, true)).unwrap();
/// assert!((dist.mean() - 2.0).abs() < 1e-10);
/// ```
pub fn mle_for<'a, T: Mle<'a>>(data: &T::Data) -> Result<T::Output, &'static str> {
    T::mle(data)
}

/// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::{Bernoulli, Binomial, Exp, Geometric, Normal, Poisson, Uniform};

    #[test]
    fn test_mle_normal_unbiased() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let est = Normal::mle(&(&data, true)).unwrap();
        assert!((est.mean().unwrap() - 3.0).abs() < 1e-12);
        assert!((est.std_dev().unwrap() - 1.5811).abs() < 1e-4);
    }

    #[test]
    fn test_mle_normal_biased() {
        let data = [1.0, 2.0, 3.0];
        let est = Normal::mle(&(&data, false)).unwrap();
        assert!((est.mean().unwrap() - 2.0).abs() < 1e-12);
        assert!((est.std_dev().unwrap() - 0.8165).abs() < 1e-3);
    }

    #[test]
    fn test_mle_binomial() {
        let data = [4, 5, 6, 5, 4];
        let est = Binomial::mle(&(&data, 10)).unwrap();
        assert!((est.p() - 0.48).abs() < 1e-10);
        assert_eq!(est.n(), 10);
    }

    #[test]
    fn test_mle_bernoulli() {
        let data = [1u8, 0, 1, 1, 0, 1];
        let est = Bernoulli::mle(&&data[..]).unwrap();
        assert!((est.p() - 0.6667).abs() < 1e-4);
    }

    #[test]
    fn test_mle_exponential() {
        let data = [1.0, 2.0, 3.0];
        let est = Exp::mle(&&data[..]).unwrap();
        assert!((est.rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_mle_poisson() {
        let data = [2u32, 3, 4, 1, 0];
        let est = Poisson::mle(&&data[..]).unwrap();
        assert!((est.lambda() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_mle_geometric() {
        let data = [0u32, 1, 2, 1, 0];
        let est = Geometric::mle(&&data[..]).unwrap();
        assert!((est.p() - 0.5555555556).abs() < 1e-10);
    }


    #[test]
    fn test_mle_uniform() {
        let data = [1.0, 2.0, 3.0, 4.5];
        let est = Uniform::mle(&&data[..]).unwrap();
        assert_eq!(est.min(), 0.0);
        assert_eq!(est.max(), 4.5);
    }

    #[test]
    fn test_mle_for_generic_normal() {
        let data = [1.0, 2.0, 3.0];
        let dist = mle_for::<Normal>(&(&data, false)).unwrap();
        assert!((dist.mean().unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_error_empty_data() {
        let empty: &[f64] = &[];
        assert!(Normal::mle(&(empty, false)).is_err());
        assert!(Exp::mle(&empty).is_err());
        assert!(Poisson::mle(&&[][..]).is_err());
        assert!(Geometric::mle(&&[][..]).is_err());
        assert!(Uniform::mle(&&[][..]).is_err());
    }

    #[test]
    fn test_error_invalid_params() {
        let bern_data = [0u8, 1, 2];
        assert!(Bernoulli::mle(&&bern_data[..]).is_err());

        let exp_data = [1.0, -1.0];
        assert!(Exp::mle(&&exp_data[..]).is_err());

        let bin_data = [5u32, 6, 7];
        assert!(Binomial::mle(&(&bin_data, 5)).is_err());

        let uniform_data = [-1.0, -2.0];
        assert!(Uniform::mle(&&uniform_data[..]).is_err());
    }
}
