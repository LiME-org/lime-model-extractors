//! Arrival-pattern inference and streaming extraction.
//!
//! This crate exposes certain-periodic, certain-sporadic, and
//! uncertain-sporadic model inference logic from LiME's Python reference
//! implementation. It works with discrete timestamps and release
//! windows and has no framework dependencies.
//!
//! # Example
//!
//! ```
//! use lime_model_extractors::{
//!     extractors::SporadicExtractor,
//!     infer_sporadic_model,
//!     time::{Duration, Instant},
//! };
//!
//! let releases: [Instant; 3] = [1000, 1500, 2000];
//! assert_eq!(infer_sporadic_model(releases), Ok(500 as Duration));
//!
//! let mut extractor = SporadicExtractor::new();
//! extractor.feed(releases).unwrap();
//! assert_eq!(extractor.current_model(), Some(500 as Duration));
//! ```

mod certain_periodic;
mod certain_sporadic;
mod periodic_core;
mod release_window;
mod sporadic_core;
mod uncertain_periodic;
mod uncertain_sporadic;

pub mod extractors {
    pub use crate::certain_periodic::PeriodicExtractor;
    pub use crate::certain_sporadic::{DeltaMaxExtractor, DeltaMinExtractor, SporadicExtractor};
    pub use crate::uncertain_periodic::{
        CertainFitPeriodicExtractor, PossibleFitPeriodicExtractor,
    };
    pub use crate::uncertain_sporadic::{
        DeltaMaxHiExtractor, DeltaMaxLoExtractor, DeltaMinHiExtractor, DeltaMinLoExtractor,
    };
}

pub mod time {
    pub type Instant = u64;
    pub type Duration = u64;

    pub use crate::release_window::ReleaseWindow;
}

pub use certain_periodic::{PeriodicConfig, PeriodicError, PeriodicModel, infer_periodic_model};
pub use certain_sporadic::{
    SporadicError, infer_delta_max, infer_delta_min, infer_sporadic_model, max_releases,
    min_releases,
};
pub use uncertain_periodic::{
    UncertainPeriodicError, infer_certain_fit_periodic_model, infer_possible_fit_periodic_model,
};
pub use uncertain_sporadic::{
    UncertainSporadicError, infer_delta_max_hi, infer_delta_max_lo, infer_delta_min_hi,
    infer_delta_min_lo,
};
