# Statistical consistency and adverse geometry
The new catalog compares common-error correlation, CI weight selection, radar baseline, range-only localization, multi-view Doppler, mounting error, asynchronous tracks, innovation whitening, common range bias and one contaminated range.

Monte Carlo consistency statistics average independent trials. Chi-square bounds use state dimension times independent trial count; time-correlated samples are not silently counted as independent evidence. Whitened innovation energy equals the Mahalanobis quadratic form, including off-diagonal covariance.

The correlation bench uses two unit-covariance estimates with a declared common-error coefficient. Known-correlation fusion has the appropriate covariance; naive fusion underestimates it. CI stays conservative without knowing that coefficient. The range-bias and robust-outlier studies intentionally violate ideal Gaussian assumptions and retain the resulting errors.

These are synthetic conditional experiments, not live radar networking, unknown-pose simultaneous calibration, guaranteed target association or measured sensor validation.
