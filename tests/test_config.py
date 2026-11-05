import json

from lsg.config import config_hash, data_root, load_config
from lsg.runinfo import write_metrics


def test_hash_stable_and_order_independent():
    a = {"x": 1, "y": {"b": 2, "a": 3}}
    b = {"y": {"a": 3, "b": 2}, "x": 1}
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash({"x": 2, "y": {"b": 2, "a": 3}})


def test_default_config_loads():
    cfg = load_config()
    assert cfg["frames"]["stride"] == 3
    assert cfg["visibility"]["tau_free"] == 0.15


def test_lsg_data_override(monkeypatch, tmp_path):
    monkeypatch.setenv("LSG_DATA", str(tmp_path))
    assert data_root({"data_root": "./elsewhere"}) == tmp_path
    monkeypatch.delenv("LSG_DATA")
    assert data_root({"data_root": "./elsewhere"}).name == "elsewhere"


def test_write_metrics(tmp_path):
    cfg = {"a": 1}
    p = write_metrics(tmp_path / "run1", {"m": 0.5}, cfg)
    d = json.loads(p.read_text())
    assert d["config_hash"] == config_hash(cfg)
    assert d["metrics"] == {"m": 0.5}
    assert "sha" in d["git"]
