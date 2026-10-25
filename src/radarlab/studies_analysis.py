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
