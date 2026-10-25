import hashlib

from radarlab.cli import main
from radarlab.detections_io import COLUMNS, write_csv
from radarlab.reporting import load_archive, verify_report


def test_record_cli_hashes_input_and_preserves_measurement_uncertainty(tmp_path):
    rows = [
        dict(zip(COLUMNS, [k * 0.1, 30 + k * 0.2, 2.0, 0.0, 0.3, 0.02, True], strict=True)) for k in range(5)
    ]
    path = write_csv(rows, tmp_path / "detections.csv")
    out = tmp_path / "report"
    assert main(["record-replay", str(path), "--output", str(out)]) == 0
    assert verify_report(out)
    r = load_archive(out / "fields.npz")
    assert r["parameters"]["input_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert r["summary"]["final_confirmed_tracks"] == 1
