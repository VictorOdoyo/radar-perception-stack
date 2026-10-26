# Multi-radar localization and correlated-track fusion
The extension adds standalone Gaussian track-fusion and localization benches. It does not merge identities in the existing multi-object IQ pipeline.

All state estimates use x,y,vx,vy. Position is lateral x and forward y; polar azimuth remains atan2(x,y). A rigid pose rotates both position and velocity and transforms the complete covariance. Mounting uncertainty adds a Jacobian-projected translation/yaw covariance, including position-velocity cross terms. Tracks are propagated forward to the fusion timestamp before combination.

Independent fusion sums information matrices. If cross covariance is known, a full joint-covariance BLUE solve retains it. If it is unknown, covariance intersection combines weighted information matrices without treating duplicate information as independent. A bounded log-determinant search includes both endpoint candidates. These assumptions matter: naive independent fusion is overconfident for correlated sensor errors.

The range localizer solves weighted nonlinear least squares with an analytic Jacobian and rejects degenerate local geometry. Multi-view signed Doppler recovers two velocity components conditional on known target position and sensor poses. Range/bearing Fisher bounds are local planning diagnostics. The robust Huber localizer reports residuals and position, and makes no calibrated Gaussian covariance claim.

See [consistency diagnostics](fusion-consistency.md) and the [computed gallery](results.md). [MathWorks fusion reference](https://www.mathworks.com/help/pdf_doc/fusion/fusion_ref.pdf) provides covariance-intersection and track-consistency context.
