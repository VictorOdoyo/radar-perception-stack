"""Validated reproducible run requests and lazy study dispatch."""

import json
from dataclasses import dataclass
from pathlib import Path

CASES = {
    "single-target": ["studies_frame", "nominal", "IQ to range Doppler angle detection"],
    "two-targets": ["studies_frame", "two", "Distinct radar returns"],
    "unresolved-targets": ["studies_frame", "close", "Close-target resolution failure"],
    "clutter-contamination": ["studies_frame", "cluttered", "Correlated slow-time clutter"],
    "clutter-subtraction": ["studies_frame", "filtered", "Moving-target clutter filtering"],
    "multipath-ghosts": ["studies_frame", "ghosts", "False object from a specular path"],
    "adc-saturation": ["studies_frame", "clipping", "Invalid clipped receiver challenge"],
    "receiver-calibration-error": ["studies_frame", "calibration", "Array gain and phase bias"],
    "oscillator-phase-noise": ["studies_frame", "oscillator", "Common oscillator instability"],
    "chirp-nonlinearity": ["studies_frame", "nonlinear", "Nonlinear beat curvature"],
    "low-snr-detection": ["studies_frame", "low_snr", "Low-SNR detection misses"],
    "chirp-dropout": ["studies_frame", "dropout", "Unavailable chirp integration loss"],
    "multi-object-tracking": ["studies_tracking", "nominal", "IQ to confirmed object tracks"],
    "tracking-short-gap": ["studies_tracking", "gap", "Prediction through unavailable frames"],
    "tracking-long-gap": ["studies_tracking", "long_gap", "Track deletion and rebirth"],
    "maneuver-model-mismatch": ["studies_tracking", "maneuver", "CV model against acceleration"],
    "cfar-false-alarm": ["studies_analysis", "cfar_rates", "Independent exponential CFAR calibration"],
    "windowed-cfar-calibration": ["studies_analysis", "window_cfar", "FFT-window correlation effects"],
    "os-cfar-contamination": ["studies_analysis", "os_robustness", "Contaminated reference cells"],
    "spatial-spectrum-comparison": ["studies_analysis", "spatial_spectra", "Bartlett MVDR and MUSIC"],
    "ego-motion-consensus": ["studies_analysis", "ego_consensus", "Stationary Doppler robust fit"],
    "dual-prf-unwrapping": ["studies_analysis", "dual_prf", "Bounded velocity ambiguity resolution"],
    "chirp-ambiguity": ["studies_analysis", "ambiguity_map", "Discrete chirp ambiguity"],
    "receiver-calibration-recovery": [
        "studies_analysis",
        "calibration_recovery",
        "Known-reference calibration recovery",
    ],
}


@dataclass(frozen=True)
class ExperimentConfig:
    case: str = "single-target"
    resolution_scale: int = 1
    seed: int = 42

    def __post_init__(self):
        if not isinstance(self.case, str) or self.case not in CASES:
            raise ValueError("unknown experiment case")
        if type(self.resolution_scale) is not int or not 1 <= self.resolution_scale <= 3:
            raise ValueError("resolution_scale must be an integer from 1 to 3")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")

    def to_dict(self):
        return {"case": self.case, "resolution_scale": self.resolution_scale, "seed": self.seed}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {"case", "resolution_scale", "seed"}:
            raise ValueError("request must be an object containing only case, resolution_scale and seed")
        return cls(**value)


def load_config(path):
    return ExperimentConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def run(config, progress=None):
    from importlib import import_module

    from .reporting import provenance, stamp
    from .results import validate

    if not isinstance(config, ExperimentConfig):
        raise ValueError("validated ExperimentConfig required")
    configuration = config.to_dict()
    source = provenance(configuration)
    module, function, _ = CASES[config.case]
    if progress is not None:
        progress("Solving " + config.case)
    result = getattr(import_module("radarlab." + module), function)(
        scale=config.resolution_scale, seed=config.seed
    )
    validate(result)
    if result["name"] != config.case:
        raise ValueError("study returned a different case name")
    return stamp(result, configuration, source)
