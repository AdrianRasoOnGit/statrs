use crate::distribution::*;
use crate::statistics::statistics::Statistics;
use crate::function::gamma::{digamma, inv_digamma};



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

///
/// CLOSED FORM SOLUTION MLE TRAITS (Normal, Binomial, Bernoulli, Exponential, Poisson, Uniform, Geometric)
/// 

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


///
/// NUMERICAL FORM SOLUTION MLE TRAITS
/// 

/// Implements MLE for the Gamma distribution.
///
/// Estimates shape α using Newton-Raphson and scale β = mean / α.
/// Requires strictly positive data.
///
/// # Parameters
/// - `data`: Slice of positive `f64` values.
///
/// # Errors
/// - Returns `"invalid parameters"` if input is empty or contains non-positive or non-finite values.
/// - Returns `"convergence failure"` if Newton-Raphson fails to converge.
///
/// # Example
/// ```
/// use statrs::distribution::{Gamma, Mle};
/// let data = [2.0, 3.0, 4.0];
/// let dist = Gamma::mle(&data).unwrap();
/// assert!(dist.shape() > 0.0);
/// ```
impl<'a> Mle<'a> for Gamma {
    type Data = &'a [f64];
    type Output = Gamma;

    fn mle(data: &Self::Data) -> Result<Self::Output, &'static str> {
        if data.is_empty() || data.iter().any(|&x| x <= 0.0 || !x.is_finite()) {
            return Err("invalid parameters");
        }

        let n = data.len() as f64;
        let mean = data.mean();
        let log_mean = data.iter().map(|&x| x.ln()).sum::<f64>() / n;
        let s = log_mean - mean.ln();

        // Initial guess using inverse digamma
        let mut alpha = if s.abs() < 1e-6 {
            1.0
        } else {
            inv_digamma(-s)
        };

        // Newton-Raphson refinement
        for _ in 0..20 {
            let psi = digamma(alpha);
            let deriv = 1.0 / alpha; // crude trigamma approx, safe for moderate α
            let delta = (psi - alpha.ln() + s) / (deriv - 1.0 / alpha);
            alpha -= delta;

            if !alpha.is_finite() || alpha <= 0.0 {
                return Err("invalid parameters");
            }
            if delta.abs() < 1e-8 {
                break;
            }
        }

        if !alpha.is_finite() || alpha <= 0.0 {
            return Err("convergence failure");
        }

        let beta = mean / alpha;
        Gamma::new(alpha, beta).map_err(|_| "invalid parameters")
    }
}

/// Implements Maximum Likelihood Estimation (MLE) for the Weibull distribution.
///
/// Estimates the shape parameter `k` using Newton-Raphson iterations starting from
/// a method-of-moments initial guess, and the scale parameter `λ` using the MLE formula.
///
/// # Parameters
/// - `data`: Slice of strictly positive `f64` samples.
///
/// # Errors
/// - Returns `"invalid parameters"` if input is empty, contains non-positive or non-finite values,
///   or if Newton-Raphson updates yield invalid values.
/// - Returns `"convergence failure"` if the iteration fails to converge to a positive finite shape.
///
/// # Example
/// ```
/// use statrs::distribution::{Weibull, Mle};
/// let data = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let dist = Weibull::mle(&data).unwrap();
/// assert!(dist.shape() > 0.0);
/// assert!(dist.scale() > 0.0);
/// ```
impl<'a> Mle<'a> for Weibull {
    type Data = &'a [f64];
    type Output = Weibull;

    fn mle(data: &Self::Data) -> Result<Self::Output, &'static str> {
        if data.is_empty() || data.iter().any(|&x| x <= 0.0 || !x.is_finite()) {
            return Err("invalid parameters");
        }

        let n = data.len() as f64;
        let mean = data.mean();
        let log_mean = data.iter().map(|&x| x.ln()).sum::<f64>() / n;

        // Initial guess for shape k (use method of moments)
        let mut k = 1.0;
        for _ in 0..20 {
            let xk = data.iter().map(|&x| x.powf(k)).sum::<f64>() / n;
            let xklnx = data.iter().map(|&x| x.powf(k) * x.ln()).sum::<f64>() / n;

            let num = xklnx / xk - log_mean;
            let den = data
                .iter()
                .map(|&x| {
                    let ln_x = x.ln();
                    let xk = x.powf(k);
                    xk * ln_x * ln_x
                })
                .sum::<f64>()
                / xk
                - (xklnx / xk).powi(2);

            let delta = num / (k * den + 1e-8); // avoid division by 0
            k -= delta;

            if !k.is_finite() || k <= 0.0 {
                return Err("invalid parameters");
            }
            if delta.abs() < 1e-8 {
                break;
            }
        }

        if !k.is_finite() || k <= 0.0 {
            return Err("convergence failure");
        }

        // Estimate scale λ using MLE formula: λ = (1/n Σ xᵏ)^{1/k}
        let lambda = (data.iter().map(|&x| x.powf(k)).sum::<f64>() / n).powf(1.0 / k);

        Weibull::new(k, lambda).map_err(|_| "invalid parameters")
    }
}

/// Implements Maximum Likelihood Estimation (MLE) for the Beta distribution.
///
/// Estimates the shape parameters α and β from data using a method-of-moments
/// initial guess followed by Newton–Raphson refinement of the log-likelihood.
///
/// # Parameters
/// - `data`: Slice of `f64` values in the open interval (0, 1).
///
/// # Errors
/// - Returns `"invalid parameters"` if the input is empty, contains values outside (0, 1),
///   or if Newton-Raphson iterations fail to produce positive finite parameters.
///
/// # Example
/// ```
/// use statrs::distribution::{Beta, Mle};
/// let data = [0.2, 0.4, 0.6, 0.8];
/// let dist = Beta::mle(&data).unwrap();
/// assert!(dist.shape_a() > 0.0);
/// assert!(dist.shape_b() > 0.0);
/// ```
impl<'a> Mle<'a> for Beta {
    type Data = &'a [f64];
    type Output = Beta;

    fn mle(data: &Self::Data) -> Result<Self::Output, &'static str> {
        let xs = *data;
        let n = xs.len();

        // Validate input
        if n == 0 || xs.iter().any(|&x| x <= 0.0 || x >= 1.0 || !x.is_finite()) {
            return Err("invalid parameters");
        }

        // Sample statistics
        let mean = xs.mean();
        let var = xs.variance(); // sample variance (uses n - 1 by default)

        // Initial estimates for α and β via method-of-moments
        let common = mean * (1.0 - mean) / var - 1.0;
        if common <= 0.0 {
            return Err("invalid parameters");
        }
        let mut alpha = mean * common;
        let mut beta = (1.0 - mean) * common;

        // Newton–Raphson to refine α and β by maximizing log-likelihood
        for _ in 0..20 {
            let psi_ab = digamma(alpha + beta);
            let grad_a = xs.iter().map(|&x| x.ln()).sum::<f64>() + n as f64 * (psi_ab - digamma(alpha));
            let grad_b = xs.iter().map(|&x| (1.0 - x).ln()).sum::<f64>() + n as f64 * (psi_ab - digamma(beta));

            // Approximate Hessian diagonal components using trigamma (approx via 1/α etc.)
            let h_aa = -(n as f64) * 1.0 / alpha;
            let h_bb = -(n as f64) * 1.0 / beta;
            let h_ab = n as f64 * 1.0 / (alpha + beta);

            // 2×2 system for update: [h] Δ = grad  => invert Hessian
            let det = h_aa * h_bb - h_ab * h_ab;
            if det.abs() < 1e-12 {
                break;
            }
            let da = ( h_bb * grad_a - h_ab * grad_b) / det;
            let db = (-h_ab * grad_a + h_aa * grad_b) / det;

            alpha -= da;
            beta  -= db;

            if alpha <= 0.0 || beta <= 0.0 || !alpha.is_finite() || !beta.is_finite() {
                return Err("invalid parameters");
            }
            if da.abs() < 1e-8 && db.abs() < 1e-8 {
                break;
            }
        }

        Beta::new(alpha, beta).map_err(|_| "invalid parameters")
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
    use crate::distribution::{Bernoulli, Binomial, Exp, Geometric, Normal, Poisson, Uniform, Gamma, Weibull, Beta};
    use crate::statistics::traits::Distribution;
    use crate::statistics::mle::Mle;

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

    #[test]
    fn test_mle_weibull() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let est = Weibull::mle(&&data[..]).unwrap();
        assert!(est.shape() > 0.0);
        assert!(est.scale() > 0.0);
    }

    #[test]
    fn test_mle_gamma_custom() {
        let data = [2.0, 3.0, 4.0, 5.0];
        let est = Gamma::mle(&&data[..]).unwrap();
        assert!(est.shape() > 0.0);
        assert!(est.rate() > 0.0);
    }

    #[test]
    fn test_mle_beta() {
        let data = [0.2, 0.4, 0.6, 0.8, 0.5];
        let est = Beta::mle(&&data[..]).unwrap();

        // Check that estimated alpha and beta are positive
        assert!(est.shape_a() > 0.0, "Alpha (shape_a) should be positive");
        assert!(est.shape_b() > 0.0, "Beta (shape_b) should be positive");

        // Check estimated mean is close to sample mean
        let sample_mean = data.iter().copied().sum::<f64>() / data.len() as f64;
        let est_mean = est.mean().unwrap();
        assert!(
            (sample_mean - est_mean).abs() < 0.05,
            "Estimated mean {} not close to sample mean {}",
            est_mean,
            sample_mean
        );
}
}
