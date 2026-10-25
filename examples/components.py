"""Synthetic radar bench; all quantities use SI units."""

from radarlab.channel import synthesize
from radarlab.pipeline import detect
from radarlab.waveform import Radar, Target, limits


def main():
    radar = Radar()
    iq = synthesize([Target(range_m=30, receding_m_s=5, angle_rad=0.2)], radar, noise_power=1, seed=42)
    result = detect(iq, radar)
    print("Physical resolution and ambiguity:", limits(radar))
    for d in result["detections"]:
        print({key: d[key] for key in ("range_m", "receding_m_s", "angle_rad")})


if __name__ == "__main__":
    main()
