from dataclasses import dataclass

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
