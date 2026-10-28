"""reference.library(load_f0=False) 的行为契约：
列表场景只读 meta.json，绝不触碰 ref_f0.npz（否则 845 首每次请求数百 MB IO）。"""
import json

from src.agent import reference


def _mk_ref(root, rid, name, with_f0=True):
    d = root / str(rid)
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(
        {"name": name, "artists": "某人", "id": rid, "query": name},
        ensure_ascii=False), encoding="utf-8")
    if with_f0:
        (d / "ref_f0.npz").write_bytes(b"not-a-real-npz")  # 损坏文件：被读就会露馅
    return d


def test_library_no_f0_load(tmp_path, monkeypatch):
    monkeypatch.setattr(reference, "REF_DIR", tmp_path)
    _mk_ref(tmp_path, 111, "歌A")
    _mk_ref(tmp_path, 222, "歌B", with_f0=False)   # 无 npz：require_cached_f0 排除

    items = reference.library(load_f0=False)
    assert [it["song"] for it in items] == ["歌A"]
    assert items[0]["f0"] is None
    assert items[0]["ref_id"] == 111
    # 损坏的 npz 还在：证明没有被读取（ref_f0 读到坏文件会删除重建）
    assert (tmp_path / "111" / "ref_f0.npz").exists()


def test_library_missing_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(reference, "REF_DIR", tmp_path / "nonexistent")
    assert reference.library(load_f0=False) == []
