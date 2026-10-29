"""
Tests for recents - the recent-files history of the GUI.

Run with:  python -m pytest test_recents.py -q
"""
import recents


def test_the_history_keeps_the_last_twenty_newest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(recents, "user_config_dir", lambda *a, **k: str(tmp_path))
    m = recents.RecentFilesManager()
    for i in range(25):
        m.add(str(tmp_path / f"capture_{i}_Raw_0.bin"))
    items = m.list()
    assert len(items) == recents.DEFAULT_LIMIT == 20
    assert items[0].endswith("capture_24_Raw_0.bin")
    assert items[-1].endswith("capture_5_Raw_0.bin")
    # reopening an older file moves it to the top instead of duplicating it
    m.add(str(tmp_path / "capture_10_Raw_0.bin"))
    assert m.list()[0].endswith("capture_10_Raw_0.bin") and len(m.list()) == 20
    # and the history survives a restart
    assert recents.RecentFilesManager().list() == m.list()
