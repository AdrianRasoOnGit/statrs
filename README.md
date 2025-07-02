# statrs (MLE fork!)

This fork of [`statrs`](https://github.com/statrs-dev/statrs) enhances Rust's statistical capabilities by introducing **closed-form and numerical form Maximum Likelihood Estimation (MLE)** support for several core distributions.

---

## What’s new?

This fork adds a modular trait-based system for parameter estimation via MLE:

- Normal::mle() – Estimate mean and standard deviation analytically
- Binomial::mle() – Estimate success probability from sample counts
- Bernoulli::mle() – Estimate success probability from binary outcomes
- Geometric::mle() – Estimate success probability from failure counts
- Poisson::mle() – Estimate rate parameter from count data
- Exponential::mle() – Estimate rate parameter from continuous data
- Uniform::mle() – Estimate bounds from sample extrema
- Gamma::mle() – Estimate shape and rate using digamma-based iteration
- Beta::mle() – Estimate shape parameters from proportions
- Weibull::mle() – Estimate shape and scale via iterative methods

---

## Usage

Add this repo as a local dependency in your `Cargo.toml`:

```toml
[dependencies]
statrs = { path = "../statrs-mle-fork" }
```

## Example

Here we provide a brief showcase of Normal::mle() and Binomial::mle() features:

use statrs::distribution::{Normal, Binomial, Mle};
<pre><code>```rustuse statrs::distribution::{Normal, Binomial, Mle};fn main() {// Estimate parameters of a Normal distributionlet data = [1.2, 2.3, 3.7, 4.1];let normal = Normal::mle(&(&data, true)).unwrap();println!("Estimated μ: {}, σ: {}", normal.mean(), normal.std_dev());// Estimate parameters of a Binomial distributionlet successes = [2, 3, 4]; // number of successeslet binomial = Binomial::mle(&(&successes, 5)).unwrap(); // 5 trialsprintln!("Estimated p: {}", binomial.p());}```</code></pre>

## Why this fork?

The motivation behind creating this fork comes from two main reasons: first, I have a deep interest in both these methods and the statrs crate itself, so implementing these capabilities in the fork has been an enjoyable and engaging activity. Secondly, this fork also responds to a feature request raised in issue #339, which called for the inclusion of Maximum Likelihood Estimators.

The original git can be found, naturally, in [the statrs-dev GitHub repository](https://github.com/statrs-dev/statrs), where the upstream development of the `statrs` crate continues. I hope you find this fork interesting; any suggestion is welcome!
