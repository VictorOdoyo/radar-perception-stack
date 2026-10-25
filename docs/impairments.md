# Receiver and scene impairments

Named random streams generate circular complex thermal noise, correlated AR slow-time clutter, common receiver oscillator phase walk and chirp dropout. All seeds and coefficients are retained in study parameters.

Per-receiver complex gain/phase, widely linear IQ imbalance, ADC quantization, coherent leakage and impulsive interference have independent APIs. ADC output includes a component-wise clipping flag and LSB size. Whole-chirp loss and interference blanking retain explicit validity masks.

Clutter subtraction removes the slow-time mean and creates a zero-Doppler blind zone. It can suppress a stationary real target as well as clutter. Ghosts are extra declared paths with separate range/angle/amplitude; their detections are scored as false objects rather than new truth.

The common oscillator model is a simplified random phase process, not a fitted phase-noise mask. Chirp curvature is a declared IF phase curvature, not a complete PLL simulation. Some impairment combinations are standalone tools and are not jointly enabled by default.
