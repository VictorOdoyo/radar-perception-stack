import pytest

from radarlab import detections_io as m


def test_detection_csv_requires_si_headers_validity_and_time_order(tmp_path):
    p = tmp_path / "detections.csv"
    p.write_text(",".join(m.COLUMNS) + "\n0,30,2,0,.3,.02,1\n.1,0,0,0,0,0,0\n")
    rows = m.read_csv(p)
    assert rows[0]["valid"] and not rows[1]["valid"]
    p.write_text(",".join(m.COLUMNS) + "\n0,30,2,0,.3,.02,true\n")
    with pytest.raises(ValueError):
        m.read_csv(p)


def test_detection_csv_roundtrip_retains_uncertainty_and_canonical_bytes(tmp_path):
    rows = [dict(zip(m.COLUMNS, [0.0, 30.0, 2.0, 0.2, 0.3, 0.02, True], strict=True))]
    p = m.write_csv(rows, tmp_path / "detections.csv")
    assert b"\r\n" not in p.read_bytes() and m.read_csv(p) == rows
