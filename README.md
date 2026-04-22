# Real-Time Model Inference

This crate provides Rust implementations of the real-time task model inference algorithms used by the [Linux Real-Time Model Extractor (LiME)](https://lime.mpi-sws.org).
For a clean Python implementation of the same algorithms, see [py-rt-model-inference](https://github.com/LiME-org/py-rt-model-inference).


## Implemented Algorithms

The crate implements the arrival-model inference algorithms used by LiME.

For _certain_ (i.e., exact) release times:

- `infer_sporadic_model`: infer the _minimum separation_ parameter of the sporadic task model from a sequence of releases.
- `infer_delta_min`: infer _delta-min_ curves from a sequence of releases, to be used with `max_releases` to derive _upper_ bounds on the _maximum_ number of releases in any interval of a given length.
- `infer_delta_max`: infer _delta-max_ curves from a sequence of releases, to be used with `min_releases` to derive _lower_ bounds on the _minimum_ number of releases in any interval of a given length.
- `infer_periodic_model`: infer periodic _(offset, period, jitter)_ models from a sequence of releases.

For _uncertain_ release windows:

- `infer_delta_min_hi` / `infer_delta_min_lo`: infer under-/over-approximations of delta-min from a sequence of release windows.
- `infer_delta_max_hi` / `infer_delta_max_lo`: infer under-/over-approximations of delta-max from a sequence of release windows.
- `infer_certain_fit_periodic_model`: infer a periodic model from a sequence of release windows that _fully covers_ every release window.
- `infer_possible_fit_periodic_model`: infer a periodic model from a sequence of release windows that _intersects_ every release window.

`Instant` and `Duration` are the crate's shared aliases for timestamps and interval lengths. `time::ReleaseWindow` represents an uncertain release timestamp as an inclusive interval `[lo, hi]`. For example, `ReleaseWindow::new(100, 120)` means the release happened sometime between `100` and `120`, while `ReleaseWindow::new(100, 100)` encodes an exact release time.

The above APIs are "one-shot" procedures that consume all given input in one go and return a model as a result. Additionally, the crate provides _streaming extractor_ variants of all these algorithms, which consume input incrementally and can be queried at any time to obtain the model inferred _so far_.

The available streaming extractors are:

- `SporadicExtractor`
- `DeltaMinExtractor`, `DeltaMinHiExtractor`, and `DeltaMinLoExtractor`
- `DeltaMaxExtractor`, `DeltaMaxHiExtractor`, and `DeltaMaxLoExtractor`
- `PeriodicExtractor`, `CertainFitPeriodicExtractor`, and `PossibleFitPeriodicExtractor`

### Streaming Extractor API

All streaming extractors follow the same basic pattern:

1. Construct an extractor with `new(...)` or `with_config(...)`.
2. Call `feed(...)` zero or more times with additional observations.
3. Call `current_model()` to retrieve the model inferred so far.

The exact constructor and return types depend on the extractor family:

- Sporadic extractor:
  `extractors::SporadicExtractor::new()`, `feed(...) -> Result<(), SporadicError>`, `current_model() -> Option<Duration>`
- Exact release curves:
  `extractors::DeltaMinExtractor::new(nmax)`, `extractors::DeltaMaxExtractor::new(nmax)`, `feed(...) -> Result<(), SporadicError>`, `current_model() -> Vec<Duration>`
- Uncertain release-window curves:
  `extractors::DeltaMinHiExtractor::new(nmax)`, `extractors::DeltaMinLoExtractor::new(nmax)`, `extractors::DeltaMaxHiExtractor::new(nmax)`, `extractors::DeltaMaxLoExtractor::new(nmax)`, `feed(...) -> Result<(), UncertainSporadicError>`, `current_model() -> Vec<Duration>`
- Periodic extractors:
  `extractors::PeriodicExtractor::new()`, `extractors::PeriodicExtractor::with_config(config)`, `extractors::CertainFitPeriodicExtractor::new()`, `extractors::PossibleFitPeriodicExtractor::new()`, `feed(...)`, `current_model() -> Option<PeriodicModel>`

The periodic extractors accept `PeriodicConfig`. The delta extractors accept an optional `nmax` bound. Exact-release extractors consume `Instant` release times; uncertain extractors consume `ReleaseWindow` values.


## Quick Start

Add the crate to your project:

```bash
cargo add lime-model-extractors
```

Infer a sporadic model from exact release times:

```rust
use lime_model_extractors::{
    extractors::SporadicExtractor, infer_sporadic_model,
    time::{Duration, Instant},
};

let releases: [Instant; 3] = [1000, 1500, 2000];
assert_eq!(infer_sporadic_model(releases), Ok(500 as Duration));

let mut extractor = SporadicExtractor::new();
extractor.feed(releases).unwrap();
assert_eq!(extractor.current_model(), Some(500 as Duration));
```

Infer a periodic model from exact release times:

```rust
use lime_model_extractors::{infer_periodic_model, PeriodicConfig, PeriodicModel};

let releases = [7, 20, 33, 46, 59];
let model = infer_periodic_model(releases, &PeriodicConfig::default()).unwrap();

assert_eq!(
    model,
    PeriodicModel {
        period: 13,
        offset: 7,
        jitter: 0,
    }
);
```

Infer a periodic model from uncertain release windows:

```rust
use lime_model_extractors::{
    infer_certain_fit_periodic_model, PeriodicConfig,
    time::ReleaseWindow,
};

let windows = [
    ReleaseWindow::new(2, 4),
    ReleaseWindow::new(12, 17),
    ReleaseWindow::new(23, 29),
    ReleaseWindow::new(34, 39),
];

let model = infer_certain_fit_periodic_model(windows, &PeriodicConfig::default()).unwrap();
assert!(model.period > 0);
```

Use a streaming periodic extractor incrementally:

```rust
use lime_model_extractors::{PeriodicConfig, extractors::PeriodicExtractor};

let mut extractor = PeriodicExtractor::with_config(PeriodicConfig::default()).unwrap();

extractor.feed([7, 20, 33]);
assert_eq!(extractor.current_model().map(|m| m.period), Some(13));

extractor.feed([46, 59]);
assert_eq!(extractor.current_model().map(|m| m.period), Some(13));
```


## Attribution

When using these algorithms for academic work, please cite the following two papers:

1. B. Brandenburg, C. Courtaud, F. Marković, and B. Ye, “LiME: The Linux Real-Time Task Model Extractor”, Proceedings of the 31st IEEE Real-Time and Embedded Technology and Applications Symposium (RTAS 2025), May 2025.
2. B. Ye, F. Marković, and B. Brandenburg, “Framework-Agnostic Model Inference for Intra-Thread Real-Time Tasks”, Proceedings of the 32nd IEEE Real-Time and Embedded Technology and Applications Symposium (RTAS 2026), May 2026.


## Feedback, Questions, Patches

Please use the project's issue tracker or contact [Björn Brandenburg](https://people.mpi-sws.org/~bbb/).


## Development

### Quick Start

Build the crate:

```bash
cargo build
```

Run the unit tests and doctests:

```bash
cargo test
```

### Reference Parity Tests

This repository includes parity tests against the Python reference implementation from `py-rt-model-inference`.

Clone the Python reference implementation and initialize its environment:

```bash
git clone --depth 1 https://gitlab.mpi-sws.org/LiME/py-rt-model-inference.git .ci/py-rt-model-inference
cd .ci/py-rt-model-inference
uv sync --dev
cd ../..
```

Run the parity tests:

```bash
cargo test --test periodic_traces_parity --test sporadic_traces_parity -- --nocapture
```

To run parity checks against all trace files included in the repository:

```bash
TEST_ALL_TRACES=1 \
cargo test --test periodic_traces_parity --test sporadic_traces_parity -- --nocapture
```

If `PY_RT_MODEL_INFERENCE_DIR` is unset, the tests look for `.ci/py-rt-model-inference`. Set the variable only if you want to use a different checkout location.

The parity drivers and Rust-side trace processing both support configurable parallelism:

- `PARITY_DRIVER_WORKERS=<n|auto>` controls the Python parity drivers.
- `RAYON_NUM_THREADS=<n>` controls Rust-side trace validation.

Example:

```bash
TEST_ALL_TRACES=1 PARITY_DRIVER_WORKERS=auto RAYON_NUM_THREADS=8 \
cargo test --test periodic_traces_parity --test sporadic_traces_parity -- --nocapture
```

### Formatting and Linting

Format the code:

```bash
cargo fmt
```

Run Clippy:

```bash
cargo clippy --all-targets --all-features -- -D warnings
```


## License

This library is free software and released under the MIT license.
