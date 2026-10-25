# Tracking and scoring

State order is x, y, vx, vy. The CV transition and continuous white-acceleration process covariance are exact over each actual timestamp interval. Measurements use polar-to-Cartesian covariance propagation with declared range and angle noise scales.

A global assignment solves gated position innovation costs with a separate missed-detection alternative for every track. Position correction uses Joseph covariance form. A separate nonlinear radial-velocity update uses its local Cartesian Jacobian. Unobserved tangential velocity has large initial uncertainty.

Births are tentative. Three hits within five observations confirm a track; confirmed tracks coast through misses and delete after five consecutive misses. Tentative tracks delete sooner. IDs are never recycled. Recorded invalid frames supply empty observations and preserve physical time.

The tracker receives extracted detections only. Truth is used to synthesize IQ and then to score completed output. Post-solve associations for plots retain original track IDs, and identity changes are counted. OSPA penalizes both location and cardinality error. Coasting predictions are not labeled fresh observations.

The measurement covariance is illustrative, not calibrated from a radar dataset. The separate RTS smoother requires a consistent saved linear-filter sequence. Main radar scenes do not claim calibrated NEES or safety performance.
