//! Arrival-pattern inference and streaming extraction.
//!
//! This crate exposes certain-periodic, certain-sporadic, and
//! uncertain-sporadic model inference logic from LiME's Python reference
//! implementation. It works with discrete timestamps and release
//! windows and has no framework dependencies.
//! It also infers resource-use and self-suspension models from per-job observations.
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

mod bos_suspension;
mod certain_periodic;
mod certain_sporadic;
mod periodic_core;
mod release_window;
mod resource_use;
mod sporadic_core;
mod suspension;
mod uncertain_periodic;
mod uncertain_sporadic;

pub mod extractors {
    pub use crate::bos_suspension::BosSuspensionModelExtractor;
    pub use crate::certain_periodic::PeriodicExtractor;
    pub use crate::certain_sporadic::{DeltaMaxExtractor, DeltaMinExtractor, SporadicExtractor};
    pub use crate::resource_use::{MaxResourceUseExtractor, MinResourceUseExtractor};
    pub use crate::suspension::{
        MaxSuspensionModelExtractor, MinSuspensionModelExtractor, SegmentedSuspensionModelExtractor,
    };
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

pub use bos_suspension::infer_bos_suspension_model;
pub use certain_periodic::{PeriodicConfig, PeriodicError, PeriodicModel, infer_periodic_model};
pub use certain_sporadic::{
    SporadicError, infer_delta_max, infer_delta_min, infer_sporadic_model, max_releases,
    min_releases,
};
pub use resource_use::{
    ResourceAmount, ResourceUseError, infer_max_resource_use, infer_min_resource_use,
};
pub use suspension::{
    BagOfSegmentsModel, BasicJobSuspensionBehavior, BasicSuspensionModel, ObservedSegment, Segment,
    SegmentedJobSuspensionBehavior, SegmentedSuspensionModel, SuspensionError,
    basic_from_segmented_model, basic_from_segmented_observation, infer_max_suspension_model,
    infer_min_suspension_model, infer_segmented_suspension_model,
};
pub use uncertain_periodic::{
    UncertainPeriodicError, infer_certain_fit_periodic_model, infer_possible_fit_periodic_model,
};
pub use uncertain_sporadic::{
    UncertainSporadicError, infer_delta_max_hi, infer_delta_max_lo, infer_delta_min_hi,
    infer_delta_min_lo,
};
