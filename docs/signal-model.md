# Signal model

The complex mixer convention is transmitted chirp times conjugated received chirp. Range beat is positive. Radial velocity is positive when receding. The analytic stop-and-hop IF phase is 2 pi [(S tau + fd) t_fast + fd t_slow + receiver_position sin(azimuth)/lambda], with tau=2R/c and fd=2v/lambda.

The fast-time range coordinate is apparent range until Doppler is removed. The detector corrects beat frequency using its measured Doppler. Do not mix this convention with receive-times-conjugate-transmit examples without changing signs.

Default radar: 77 GHz, 300 MHz bandwidth, 40 us ramp, 50 us repetition, 6.4 MHz sampling, 256 ADC samples, 64 chirps and eight half-wavelength receivers. Only the positive fast-time Nyquist interval is used. Resolution follows observed bandwidth, not FFT zero padding.

Main scenes use a stop-and-hop frame model. Separate APIs demonstrate range migration, micro-Doppler, chirp curvature, leakage and TDM phase compensation. These optional effects are not all enabled in every scene. Signal amplitudes are normalized IF quantities; radar-equation and thermal-noise APIs are separate link-budget tools.

Primary references: [MathWorks FMCW range estimation](https://www.mathworks.com/help/phased/ug/fmcw-range-estimation.html), [range-Doppler response](https://www.mathworks.com/help/phased/ug/range-doppler-response.html), [TI mmWave training](https://www.ti.com/video/series/mmwave-training-series.html). No measured-radar equivalence is claimed.
