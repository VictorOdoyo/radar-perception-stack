# Validation scope

Checks compare analytic mixer phase and sampled FFT frequencies, coherent gain and noise gain, exact CFAR tail equations, Monte Carlo false alarms, steering geometry, covariance positivity, Joseph updates, CV covariance composition, radial Jacobians, global association and track lifecycle behavior.

End-to-end tests recover synthetic range/velocity/angle and track two objects from IQ. Challenge scenes retain unresolved returns, ghost detections, receiver angle bias, phase-noise coherence loss and track deletion/rebirth.

Statistical results are conditional on declared generative distributions. Narrow algebraic residuals verify equations, not hardware accuracy. MUSIC source count, ego stationary consensus, dual-PRF association and search bounds are assumptions, not free information available to every perception pipeline.

No measured radar dataset, RF front-end calibration, antenna chamber measurements, automotive certification or real-time deployment is claimed. CI runs Python 3.11/3.13 on Linux/Windows plus formatting and installed-wheel report checks.
