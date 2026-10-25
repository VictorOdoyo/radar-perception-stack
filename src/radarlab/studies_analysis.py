import numpy as np

from .results import curve, make


def cfar_rates(scale=1, seed=42):
    from .core import stream
    from .detection import ca_multiplier, wilson

    rng = stream(seed, "cfar-calibration")
    n = 48
    trials = 30000 * scale
    pfa = np.geomspace(0.001, 0.1, 9)
    rates = []
    lower = []
    upper = []
    for probability in pfa:
        noise = rng.gamma(n, 1 / n, trials)
        cut = rng.exponential(size=trials)
        count = int(np.sum(cut > ca_multiplier(n, float(probability)) * noise))
        rates.append(count / trials)
        lo, hi = wilson(count, trials)
        lower.append(lo)
        upper.append(hi)
    a = {"nominal_pfa": pfa, "observed_pfa": rates, "lower": lower, "upper": upper}
    return make(
        "cfar-false-alarm",
        {"seed": seed, "training_cells": n, "trials_per_probability": trials},
        {"maximum_absolute_error": float(max(abs(np.array(rates) - pfa)))},
        a,
        [
            curve(
                "Independent exponential CFAR calibration",
                "nominal_pfa",
                [
                    ("nominal_pfa", "Nominal"),
                    ("observed_pfa", "Monte Carlo"),
                    ("lower", "Wilson lower"),
                    ("upper", "Wilson upper"),
                ],
                "Requested false alarm probability",
                "Observed probability",
                xscale="log",
                yscale="log",
            )
        ],
        assumptions=[
            "Independent exponential CUT and gamma-distributed training mean; no FFT-window correlation."
        ],
    )


def window_cfar(scale=1, seed=42):
    from .channel import synthesize
    from .detection import ca_cfar, wilson
    from .pipeline import spectrum
    from .waveform import Radar

    p = Radar(receivers=2)
    trials = 12 * scale
    rows = []
    for kind in ("boxcar", "hann"):
        count = total = 0
        for k in range(trials):
            power = spectrum(synthesize([], p, 1, (seed + k) % 2**32), p, window_kind=kind)["power"]
            r = ca_cfar(power, false_alarm=0.01)
            count += int(sum(r["detections"].ravel()))
            total += int(sum(r["valid"].ravel()))
        lo, hi = wilson(count, total)
        rows.append(
            {
                "window": kind,
                "false_alarms": count,
                "valid_cells": total,
                "rate": count / total,
                "wilson_lower": lo,
                "wilson_upper": hi,
            }
        )
    a = {
        "window_index": np.arange(2),
        "observed_rate": [r["rate"] for r in rows],
        "nominal_rate": np.full(2, 0.01),
    }
    return make(
        "windowed-cfar-calibration",
        {"radar": p.__dict__, "frames": trials, "seed": seed, "nominal_pfa": 0.01},
        {"boxcar_rate": rows[0]["rate"], "hann_rate": rows[1]["rate"]},
        a,
        [
            curve(
                "Neighbor correlation changes calibration",
                "window_index",
                [("observed_rate", "Observed; 0 boxcar / 1 Hann"), ("nominal_rate", "Nominal")],
                "Window index",
                "False alarm fraction",
            )
        ],
        {"rates": rows},
        [
            "Wilson intervals summarize counts; overlapping CFAR windows correlate decisions and these are not independent-cell confidence guarantees."
        ],
    )


def os_robustness(scale=1, seed=42):
    from .core import stream
    from .detection import ca_multiplier, os_multiplier

    rng = stream(seed, "os-contamination")
    n = 32
    k = 24
    trials = 10000 * scale
    contamination = np.arange(0, 9)
    ca = []
    os = []
    for count in contamination:
        training = rng.exponential(size=(trials, n))
        training[:, :count] += 100
        cut = abs(5 + (rng.normal(size=trials) + 1j * rng.normal(size=trials)) / np.sqrt(2)) ** 2
        ca.append(float(np.mean(cut > ca_multiplier(n, 0.001) * training.mean(axis=1))))
        os.append(
            float(np.mean(cut > os_multiplier(n, k, 0.001) * np.partition(training, k - 1, axis=1)[:, k - 1]))
        )
    a = {"contaminated_cells": contamination, "ca_pd": ca, "os_pd": os}
    return make(
        "os-cfar-contamination",
        {"seed": seed, "training_cells": n, "rank": k, "trials": trials, "pfa": 0.001, "target_amplitude": 5},
        {"final_ca_pd": ca[-1], "final_os_pd": os[-1]},
        a,
        [
            curve(
                "Conditional detection under reference contamination",
                "contaminated_cells",
                [("ca_pd", "CA"), ("os_pd", "OS")],
                "Strong reference cells",
                "Detection fraction",
            )
        ],
        assumptions=[
            "Independent exponential reference cells; specified complex Gaussian CUT noise and deterministic target amplitude."
        ],
    )


def spatial_spectra(scale=1, seed=42):
    from .array import covariance, music, mvdr, steering
    from .core import stream
    from .processing import power_db

    rng = stream(seed, "spatial-sources")
    angles = np.linspace(-0.7, 0.7, 701 * scale)
    source_angles = [-0.2, 0.25]
    a = steering(source_angles)
    s = (rng.normal(size=(300, 2)) + 1j * rng.normal(size=(300, 2))) / np.sqrt(2)
    x = s @ a + 0.15 * (rng.normal(size=(300, 8)) + 1j * rng.normal(size=(300, 8))) / np.sqrt(2)
    r = covariance(x)
    conventional = np.einsum("ij,jk,ik->i", steering(angles).conj(), r, steering(angles)).real / 64
    capon = mvdr(r, angles)
    subspace = music(r, angles, 2)
    arrays = {
        "angle_deg": np.rad2deg(angles),
        "bartlett_db": power_db(conventional, max(conventional)),
        "mvdr_db": power_db(capon, max(capon)),
        "music_db": power_db(subspace, max(subspace)),
        "covariance": r,
        "snapshots": x,
    }
    return make(
        "spatial-spectrum-comparison",
        {
            "seed": seed,
            "source_angles_rad": source_angles,
            "snapshots": 300,
            "receivers": 8,
            "noise_component_scale": 0.15,
        },
        {"covariance_condition": float(np.linalg.cond(r))},
        arrays,
        [
            curve(
                "Same covariance, different spatial estimators",
                "angle_deg",
                [("bartlett_db", "Bartlett"), ("mvdr_db", "MVDR"), ("music_db", "MUSIC (2 sources assumed)")],
                "Azimuth (deg)",
                "Normalized spectrum (dB)",
            )
        ],
        assumptions=[
            "MUSIC source count is explicitly assumed; not estimated from ground truth inside a perception pipeline."
        ],
    )


def ego_consensus(scale=1, seed=42):
    from .core import stream
    from .egomotion import fit_velocity, robust_velocity

    angles = np.linspace(-0.7, 0.7, 60 * scale)
    truth = np.array([1.0, 12.0])
    design = -np.column_stack([np.sin(angles), np.cos(angles)])
    velocity = design @ truth + stream(seed, "ego-measurement").normal(0, 0.05, len(angles))
    velocity[::5] += 5
    least = fit_velocity(angles, velocity)
    robust = robust_velocity(angles, velocity, seed=seed)
    pred = design @ robust["velocity_m_s"]
    a = {
        "angle_deg": np.rad2deg(angles),
        "measured_m_s": velocity,
        "robust_m_s": pred,
        "truth_m_s": design @ truth,
        "inliers": robust["inliers"],
    }
    return make(
        "ego-motion-consensus",
        {"seed": seed, "ego_velocity_m_s": truth.tolist(), "outlier_step": 5, "noise_std_m_s": 0.05},
        {
            "least_squares_error_m_s": float(np.linalg.norm(least["velocity_m_s"] - truth)),
            "robust_error_m_s": float(np.linalg.norm(robust["velocity_m_s"] - truth)),
        },
        a,
        [
            curve(
                "Static-world Doppler consensus",
                "angle_deg",
                [
                    ("measured_m_s", "Measured incl moving objects"),
                    ("truth_m_s", "Static truth"),
                    ("robust_m_s", "Consensus fit"),
                ],
                "Azimuth (deg)",
                "Receding radial velocity (m/s)",
            )
        ],
    )
