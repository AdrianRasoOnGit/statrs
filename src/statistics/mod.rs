//! Provides traits for statistical computation

pub use self::order_statistics::*;
pub use self::slice_statistics::*;
pub use self::statistics::*;
pub use self::traits::*;

// Contributions on Maximum Likelihood Estimators
pub mod mle;

mod iter_statistics;
mod order_statistics;
// TODO: fix later
mod slice_statistics;
#[allow(clippy::module_inception)]
mod statistics;
mod traits;
