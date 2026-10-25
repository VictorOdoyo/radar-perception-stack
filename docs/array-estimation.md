# Array estimation

Azimuth is measured from forward-positive y toward lateral-positive x. A ULA steering phase increments by 2 pi d/lambda sin(azimuth). The front hemisphere is used, and adjacent-baseline phase estimation rejects spacing above half a wavelength.

Bartlett beam scanning is the detector's angle estimator. MVDR and MUSIC are standalone covariance benches. MVDR requires a positive definite loaded Hermitian covariance. MUSIC requires a declared source count and positive semidefinite covariance. The source-count assumption is printed in the published comparison.

Forward-backward smoothing can restore rank for coherent sources while reducing aperture. It is not applied automatically to every detection. Spatial FFT coordinates are uniform in sine angle, rather than uniform angle.

Receiver gain/phase errors and their known-reference correction are evaluated separately. A broadside reference identifies complex receiver response; the repository does not infer perfect calibration from unknown scene truth. Grating-lobe and single-source CRLB APIs expose geometry and observability limits.
