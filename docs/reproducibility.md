# Reproducible studies

ExperimentConfig accepts case, resolution_scale in 1–3, and unsigned 32-bit seed. Default studies use scale 1 and seed 42. A larger scale increases sample counts or observation cadence; it is not universally a numerical convergence sweep because coherent aperture can change.

The dispatcher freezes Git revision, dirty status, runtime versions and request SHA before solving. Resolved physical parameters receive a separate SHA. Arrays are finite numeric NPZ fields with no pickle objects. Complex IQ, spectra and covariance retain complex dtype.

Reports contain CSV/JSON, raw arrays, plots, HTML and a complete SHA-256 manifest. verify rejects modified or missing files. Catalog paths cannot escape their root. Rendering and replay preserve the original solve source.

```sh
radarlab batch --data-only --output outputs/catalog
radarlab render outputs/catalog
radarlab verify outputs/catalog
python examples/build_gallery.py outputs/catalog --destination docs/gallery
```
Publication requires every default case captured from one clean committed revision. Source and artifact commits need not be identical; the recorded source identifies the actual physical solve.
