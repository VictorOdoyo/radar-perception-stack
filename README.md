# Radar Perception Stack

[![Engineering checks](https://github.com/VictorOdoyo/radar-perception-stack/actions/workflows/ci.yml/badge.svg)](https://github.com/VictorOdoyo/radar-perception-stack/actions/workflows/ci.yml)

An advanced electrical engineering laboratory that follows complex FMCW IQ through spectral processing, CFAR detection, array angle estimation, covariance propagation, global association and multi-object tracking.

![Range-Doppler detection](docs/gallery/single-target/figure.png)

## Run

```sh
uv sync --locked --extra dev
uv run radarlab list
uv run radarlab run --case single-target --output outputs/single
uv run radarlab run --case multi-object-tracking --output outputs/tracking
uv run radarlab verify outputs/tracking
```

Python 3.11 or newer is supported. A built wheel exposes the same `radarlab` command outside the source checkout.

## Investigate

- Coherent range/Doppler/angle IQ, sampled ambiguity bounds, stop-and-hop assumptions, normalized FFTs and physical noise gain.
- CA/OS-CFAR calibration, correlated windowed noise, guard cells, unavailable boundaries, peak interpolation and unresolved returns.
- Bartlett, MVDR and MUSIC, loaded spatial covariance, coherent-source smoothing, receiver calibration, grating lobes and angle bounds.
- Position and radial-velocity updates, Joseph covariance, gated global association, confirmation, coasting, deletion, rebirth and identity changes.
- Clutter, ghosts, ADC clipping, phase noise, nonlinear chirps, missing chirps and missed frames.
- Standalone waveform ambiguity, multi-PRF unwrapping, micro-Doppler, range migration, TDM compensation and robust ego-motion analysis.

Truth is used for synthesis and post-solve scoring. The tracker receives detections only. Failed or unavailable estimates retain explicit validity and cardinality penalties.

## Evidence and methods

The [computed gallery](docs/results.md) contains all 34 studies with raw arrays, CSV/JSON, plots, HTML reports, source provenance and SHA-256 manifests.

| Method | Read |
|---|---|
| Mixer signs, range coupling and frame geometry | [Signal model](docs/signal-model.md) |
| Detector calibration and window correlation | [Detection](docs/detection.md) |
| Array estimation and reference calibration | [Array methods](docs/array-estimation.md) |
| Detection-only tracking and identity scoring | [Tracking](docs/tracking.md) |
| Receiver and scene challenges | [Impairments](docs/impairments.md) |
| External raw IQ and detection data | [Records](docs/records.md) |
| Source freezing and replay | [Reproducibility](docs/reproducibility.md) |
| Independent checks and limits | [Validation](docs/validation.md) |
| Module map and MIT attribution | [Architecture](docs/architecture.md) |

These are synthetic numerical studies with illustrative sensor uncertainty. No measured radar calibration, automotive qualification or real-time hardware performance is claimed.

## Recorded data

```sh
uv run radarlab record-replay examples/detections.csv --output outputs/recorded
uv run radarlab verify outputs/recorded
uv run radarlab iq-audit PATH_TO_IQ_RECORD
uv run python examples/components.py
```

The example CSV is synthetic and includes an unavailable frame. Replay preserves supplied uncertainties and fingerprints original input bytes.

## Reproduce and develop

```sh
uv run radarlab batch --data-only --output outputs/catalog
uv run radarlab render outputs/catalog
uv run radarlab verify outputs/catalog
uv run python examples/build_gallery.py outputs/catalog --destination docs/gallery
uv run pytest -q
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv build
```

Publication requires a complete default catalog captured from one clean committed revision. CI checks Linux/Windows, Python 3.11/3.13, formatting, and an installed-wheel scientific report. See [contributing](CONTRIBUTING.md) and [release notes](CHANGELOG.md). MIT licensed.

## Multi-radar extension

[Localization and correlated-track fusion](docs/multisensor-fusion.md) and [statistical consistency](docs/fusion-consistency.md) add ten independent studies covering geometry, mounting error, timestamp alignment, correlation and contaminated measurements.
