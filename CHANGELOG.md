# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-11-17

### Added

- Maximum and minimum cumulative resource-use models, including WCET(n) and BCET(n), with one-shot inference and streaming extraction.
- Minimum and maximum dynamic (basic) self-suspension models over consecutive jobs.
- Segmented and bag of segment vectors (BOS) self-suspension models, with one-shot inference and streaming extraction.
- Conversion of segmented observations and models to basic suspension counts and durations.
- Python parity checks and benchmarks for self-suspension models.

### Changed

- Updated crate dependencies to their latest releases.
- The crate now declares a minimum supported Rust version of 1.94.

### Fixed

- Periodic extraction no longer overflows when the sum of the period search bounds exceeds the duration range.

## [0.1.0] - 2026-04-22

Initial public release.
