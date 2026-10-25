# Detection

Range and Doppler FFTs use periodic windows and coherent-gain normalization. Detection power comes from receiver zero. Array channels estimate direction after a spectral detection, avoiding an uncalibrated multi-channel noise-distribution assumption.

CA-CFAR uses alpha=N expm1(-log(Pfa)/N). This is exact for independent exponential CUT power and independent exponential training cells. OS-CFAR solves the exact exponential order-statistic tail product. Guard rectangles exclude the CUT and nearby target energy; boundary cells without full training support are explicitly unavailable.

Hann windows correlate adjacent spectral cells. The independent exponential study checks the analytic calibration, while the actual FFT study measures the effect of window correlation. Nominal Pfa is not a universal hardware calibration. Wilson count intervals do not remove correlation between overlapping CFAR decisions.

Nonmaximum suppression selects descending-power cells within a declared neighborhood. Log-parabolic interpolation refines isolated peaks; it does not resolve physically unresolved targets. The close-target scene retains missed objects, and the ghost path counts as a false physical object.

ADC saturation is a deliberately invalid sensor challenge. Its output is retained for diagnosis with a clipping mask; do not interpret a plausible peak as a calibrated measurement.
