#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use indicatif::{ProgressBar, ProgressStyle};
use lime_model_extractors::time::{Instant, ReleaseWindow};
use serde::de::DeserializeOwned;

pub const DEFAULT_NMAX: usize = 32;

pub type LoadedTrace = (Vec<Instant>, Vec<ReleaseWindow>);

pub fn python_reference_dir() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(path) = std::env::var_os("PY_RT_MODEL_INFERENCE_DIR") {
        let path = PathBuf::from(path);
        if path.is_absolute() {
            path
        } else {
            manifest_dir.join(path)
        }
    } else {
        manifest_dir.join(".ci/py-rt-model-inference")
    }
}

pub fn python_reference_python(reference_dir: &Path) -> Option<PathBuf> {
    ["python3", "python"]
        .iter()
        .map(|name| reference_dir.join(".venv/bin").join(name))
        .find(|path| path.exists())
}

pub fn python_reference_payload<T: DeserializeOwned>(
    python_path: &Path,
    reference_dir: &Path,
    driver_path: &Path,
    label: &str,
) -> T {
    let mut command = Command::new(python_path);
    command
        .current_dir(reference_dir)
        .arg(driver_path)
        .arg(reference_dir);

    if let Some(value) = std::env::var_os("TEST_ALL_TRACES") {
        command.env("TEST_ALL_TRACES", value);
    }

    let output = command.output().unwrap();

    assert!(
        output.status.success(),
        "python {label} command failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    serde_json::from_slice(&output.stdout).unwrap()
}

pub fn load_release_trace(
    trace_file: &Path,
    cutoff: Option<usize>,
) -> (Vec<u64>, Vec<ReleaseWindow>) {
    let content = fs::read_to_string(trace_file).unwrap();
    let mut releases = Vec::new();
    let mut windows = Vec::new();

    for line in content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .take(cutoff.unwrap_or(usize::MAX))
    {
        let mut cols = line.split(',');
        let release = cols.next().unwrap().parse::<Instant>().unwrap();
        let lo = cols.next().unwrap().parse::<Instant>().unwrap();
        let hi = cols.next().unwrap().parse::<Instant>().unwrap();
        releases.push(release);
        windows.push(ReleaseWindow::from((lo, hi)));
    }

    (releases, windows)
}

pub fn load_sporadic_traces() -> Vec<LoadedTrace> {
    load_trace_set("sporadic")
}

pub fn load_periodic_traces() -> Vec<LoadedTrace> {
    load_trace_set("periodic")
}

fn load_trace_set(kind: &str) -> Vec<LoadedTrace> {
    let trace_dir = trace_root_dir().join(kind);
    let trace_file_names = trace_file_names(&trace_dir);
    let mut traces = Vec::with_capacity(trace_file_names.len());

    for trace_file_name in trace_file_names {
        let trace_path = trace_dir.join(&trace_file_name);
        traces.push(load_release_trace(&trace_path, None));
    }

    traces
}

pub fn progress_bar(prefix: &str, len: usize) -> ProgressBar {
    let progress = ProgressBar::new(len as u64);
    let style = ProgressStyle::with_template("[{prefix}] {pos}/{len} {wide_msg}")
        .unwrap()
        .progress_chars("##-");
    progress.set_style(style);
    progress.set_prefix(prefix.to_string());
    progress
}

fn trace_root_dir() -> PathBuf {
    python_reference_dir().join("tests").join("traces")
}

fn trace_file_names(trace_dir: &Path) -> Vec<String> {
    let mut trace_file_names = fs::read_dir(trace_dir)
        .unwrap_or_else(|err| {
            panic!(
                "failed to list trace directory {}: {err}",
                trace_dir.display()
            )
        })
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.is_file())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("csv"))
        .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
        .collect::<Vec<_>>();

    trace_file_names.sort_unstable();
    assert!(
        !trace_file_names.is_empty(),
        "no trace files found in {}",
        trace_dir.display()
    );
    trace_file_names
}
