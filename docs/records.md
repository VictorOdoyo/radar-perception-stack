# External records

Raw IQ uses a pickle-free NPZ schema with time_s, complex iq, boolean valid and Unicode JSON metadata containing schema and Radar geometry. IQ shape is frame, chirp, fast sample, receiver. Validity is per frame; arrays become immutable on import.

Polar detection CSV requires this exact SI header:
```text
time_s,range_m,receding_m_s,azimuth_rad,range_std_m,azimuth_std_rad,valid
```
Rows are nondecreasing in time. Multiple rows may belong to one frame. valid is exactly 0 or 1. Valid rows need positive range and uncertainty; an invalid placeholder preserves an unavailable frame's timestamp.

Run:
```sh
radarlab record-replay examples/detections.csv --output outputs/recorded
radarlab verify outputs/recorded
radarlab iq-audit PATH_TO_RAW_RECORD
```
The CSV example is synthetic. Record replay hashes the original input bytes before inference and preserves the supplied measurement uncertainty. It does not calibrate sensors or substitute target truth.
