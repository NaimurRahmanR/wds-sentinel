from __future__ import annotations

import hashlib
from pathlib import Path

from wds_sentinel.data.battledim import derive_onsets, load_scada_csv

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_final_2019_multimodal_files_match_published_md5s():
    expected = {
        "2019_SCADA_Pressures.csv": "5ea1e46d3f2f0a89a3f98d6fd39a851d",
        "2019_SCADA_Flows.csv": "28fc99fdcbf80fcd26079e7fe602d6dc",
        "2019_SCADA_Levels.csv": "e5a8050bd38729e4b7648b9d52abd5b1",
        "2019_Leakages.csv": "e1f0a43683813a90a9fec8562dde599b",
    }
    for name, digest in expected.items():
        assert _md5(DATA / name) == digest


def test_final_2019_inputs_are_aligned_and_have_expected_onset_structure():
    pressure = load_scada_csv(DATA / "2019_SCADA_Pressures.csv")
    flow = load_scada_csv(DATA / "2019_SCADA_Flows.csv")
    level = load_scada_csv(DATA / "2019_SCADA_Levels.csv")
    leak = load_scada_csv(DATA / "2019_Leakages.csv")

    assert pressure.shape == (105120, 33)
    assert flow.shape == (105120, 3)
    assert level.shape == (105120, 1)
    assert leak.shape == (105120, 23)
    assert pressure.index.equals(flow.index)
    assert pressure.index.equals(level.index)
    assert pressure.index.equals(leak.index)
    assert not pressure.isna().any().any()
    assert not flow.isna().any().any()
    assert not level.isna().any().any()

    onsets, carryovers = derive_onsets(leak)
    assert len(onsets) == 19
    assert sorted(carryovers) == ["p257", "p427", "p654", "p810"]
