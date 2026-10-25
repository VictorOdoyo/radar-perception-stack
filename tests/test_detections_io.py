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


def test_detection_replay_groups_same_frame_and_coasts_on_invalid_row():
    rows = []
    for k in range(5):
        rows.append(dict(zip(m.COLUMNS, [k * 0.1, 30 + k * 0.2, 2.0, 0.0, 0.3, 0.02, k != 4], strict=True)))
    h = m.replay(rows)
    assert len(h[-1]["tracks"]) == 1 and h[-1]["tracks"][0]["misses"] == 1


def test_external_report_retains_input_rows_and_track_table():
    rows = [
        dict(zip(m.COLUMNS, [k * 0.1, 30 + k * 0.2, 2.0, 0.0, 0.3, 0.02, True], strict=True))
        for k in range(5)
    ]
    r = m.report(rows)
    assert r["summary"]["final_confirmed_tracks"] == 1
    assert r["parameters"]["rows"] == rows and len(r["tables"]["tracks"]) == 3


def test_detection_csv_rejects_truncated_rows(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text(",".join(m.COLUMNS) + "\n0,30,2\n")
    with pytest.raises(ValueError):
        m.read_csv(p)
