from dataclasses import dataclass

import numpy as np

C = 299792458.0


@dataclass(frozen=True)
class Radar:
    carrier_hz: float = 77e9
    bandwidth_hz: float = 300e6
    ramp_s: float = 40e-6
    repetition_s: float = 50e-6
    sample_hz: float = 6.4e6
    samples: int = 256
    chirps: int = 64
    receivers: int = 8
    spacing_wavelength: float = 0.5

    def __post_init__(self):
        from .core import integer, positive

        for key in (
            "carrier_hz",
            "bandwidth_hz",
            "ramp_s",
            "repetition_s",
            "sample_hz",
            "spacing_wavelength",
        ):
            positive(getattr(self, key), key)
        for key in ("samples", "chirps", "receivers"):
            integer(getattr(self, key), 2)
        if self.samples / self.sample_hz > self.ramp_s * (1 + 1e-12) or self.ramp_s > self.repetition_s:
            raise ValueError("ADC must fit the ramp and ramp must fit repetition")

    @property
    def slope(self):
        return self.bandwidth_hz / self.ramp_s

    @property
    def wavelength(self):
        return C / self.carrier_hz

    @property
    def spacing_m(self):
        return self.spacing_wavelength * self.wavelength


def coordinates(model=None):
    p = Radar() if model is None else model
    return {
        "fast_s": np.arange(p.samples) / p.sample_hz,
        "slow_s": np.arange(p.chirps) * p.repetition_s,
        "receiver_m": np.arange(p.receivers) * p.spacing_m,
    }


def chirp(time_s, model=None):
    from .core import finite

    p = Radar() if model is None else model
    t = finite(time_s, real=True)
    return np.exp(1j * np.pi * p.slope * t * t)


def dechirp(received, transmitted):
    from .core import finite

    rx, tx = np.broadcast_arrays(finite(received), finite(transmitted))
    return tx * np.conj(rx)
