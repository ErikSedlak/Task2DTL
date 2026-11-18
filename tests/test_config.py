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


def test_hub_offline_auto_only_when_all_repos_cached(tmp_path, monkeypatch):
    from lsg.config import hub_repos, setup_hub_offline

    cfg = load_config()
    repos = hub_repos(cfg)
    assert repos == ["timm/vit_small_patch14_dinov2.lvd142m", "timm/vit_base_patch32_clip_224.openai"]
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    snap = tmp_path / ("models--" + repos[0].replace("/", "--")) / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"x")
    assert setup_hub_offline(cfg) == "online"  # CLIP missing: a fresh machine must download
    import os

    assert "HF_HUB_OFFLINE" not in os.environ
    snap2 = tmp_path / ("models--" + repos[1].replace("/", "--")) / "snapshots" / "def"
    snap2.mkdir(parents=True)
    (snap2 / "open_clip_model.safetensors").write_bytes(b"x")
    assert setup_hub_offline(cfg) == "offline" and os.environ["HF_HUB_OFFLINE"] == "1"
    assert setup_hub_offline(cfg) == "env"  # an explicit variable wins
    monkeypatch.delenv("HF_HUB_OFFLINE")
    cfg["models"]["hub_offline"] = False
    assert setup_hub_offline(cfg) == "online" and "HF_HUB_OFFLINE" not in os.environ
