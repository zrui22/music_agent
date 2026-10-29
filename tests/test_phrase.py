"""逐乐句诊断 phrase_diagnostics 的分支测试（合成对齐视图 + 合成 wav，无外部依赖）。"""
import wave

import numpy as np
import pytest

from src.pitchref.align import AlignedView
from src.agent.phrase import phrase_diagnostics, _vibrato_grid

STEP = 0.05
SR = 24000


def _mk_view(n=400, key_offset_semis=0.0, bounds=(0, 200, 400), seed=0):
    """合成 AlignedView：cover 恒 60 + 可选偏移，ref 恒 60，mapping 恒等。"""
    rng = np.random.default_rng(seed)
    ct = np.arange(n) * STEP
    cv = rng.random(n) > 0.1
    cm = np.where(cv, 60.0 + key_offset_semis + rng.normal(0, 0.05, n), 0.0)
    rm = np.full(n, 60.0)
    rv = np.ones(n, dtype=bool)
    mapping = np.arange(n)
    return AlignedView(ct, cm, cv, rm, rv, mapping, STEP,
                       np.array(bounds), [(0.5, n)])


def _mk_wav(path, dur_s=20.0, freq=261.6):
    t = np.arange(int(dur_s * SR)) / SR
    x = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(x.tobytes())


LRC = [(0.0, "第一句"), (10.0, "第二句")]


def test_split_by_lrc(tmp_path):
    wav = tmp_path / "a.wav"
    _mk_wav(wav)
    ph = phrase_diagnostics(_mk_view(), LRC, str(wav), global_slope=1.0)
    assert len(ph) == 2
    assert ph[0]["lyric"] == "第一句" and ph[1]["lyric"] == "第二句"
    assert ph[0]["end_s"] <= 10.0 < ph[1]["start_s"] or ph[1]["start_s"] >= 9.9
    assert all(0 <= p["score"] <= 100 for p in ph)


def test_pitch_off_direction(tmp_path):
    wav = tmp_path / "b.wav"
    _mk_wav(wav)
    view = _mk_view()
    view.cm[200:] += 0.5  # 第二句偏高 50 音分
    ph = phrase_diagnostics(view, LRC, str(wav), global_slope=1.0)
    assert ph[1]["score"] < ph[0]["score"]
    types = [i["type"] for i in ph[1]["issues"]]
    assert "pitch_off" in types
    assert ph[1]["signed_cents"] > 15  # 偏高


def test_octave_folds_signed(tmp_path):
    wav = tmp_path / "c.wav"
    _mk_wav(wav)
    view = _mk_view()
    view.cm[200:] -= 12.0  # 第二句低八度
    ph = phrase_diagnostics(view, LRC, str(wav), global_slope=1.0)
    second = ph[1]
    assert "octave_error" in [i["type"] for i in second["issues"]]
    # 折叠后 signed 应在八度内（接近 0），而不是 -1200
    assert abs(second["signed_cents"]) < 100


def test_fallback_bounds_without_lrc(tmp_path):
    wav = tmp_path / "d.wav"
    _mk_wav(wav)
    ph = phrase_diagnostics(_mk_view(), [], str(wav), global_slope=1.0)
    assert len(ph) == 2  # 两个气口段
    assert all(p["lyric"] == "" for p in ph)


def test_vibrato_grid_detects_5hz():
    n = 40  # 2s @50ms
    t = np.arange(n) * STEP
    midi = 60 + 0.5 * np.sin(2 * np.pi * 5.5 * t)  # ±50 音分 5.5Hz 颤音
    voiced = np.ones(n, dtype=bool)
    v = _vibrato_grid(midi, voiced, STEP)
    assert v is not None
    rate, extent = v
    assert 4.0 <= rate <= 7.5
    assert extent > 60


def test_vibrato_grid_rejects_flat():
    n = 40
    midi = np.full(n, 60.0)
    voiced = np.ones(n, dtype=bool)
    assert _vibrato_grid(midi, voiced, STEP) is None


def test_flat_long_note_vs_ref_vibrato(tmp_path):
    """原唱有颤音、cover 平直 -> flat_long_note；cover 也有 -> 不报。"""
    wav = tmp_path / "e.wav"
    _mk_wav(wav)
    view = _mk_view(n=200, bounds=(0, 200))
    view.cv[:] = True
    view.cm[:] = 60.0  # cover 完全平直
    t = np.arange(200) * STEP
    view.rm[:] = 60 + 0.5 * np.sin(2 * np.pi * 5.5 * t)  # ref 有颤音
    ph = phrase_diagnostics(view, [(0.0, "长音句")], str(wav), global_slope=1.0)
    assert ph and "flat_long_note" in [i["type"] for i in ph[0]["issues"]]
    # cover 也加颤音 -> 不再报平直
    view2 = _mk_view(n=200, bounds=(0, 200))
    view2.cv[:] = True
    view2.cm[:] = 60 + 0.5 * np.sin(2 * np.pi * 5.5 * t)
    view2.rm[:] = 60 + 0.5 * np.sin(2 * np.pi * 5.5 * t)
    ph2 = phrase_diagnostics(view2, [(0.0, "长音句")], str(wav), global_slope=1.0)
    assert ph2 and "flat_long_note" not in [i["type"] for i in ph2[0]["issues"]]


def test_wrong_note_detection(tmp_path):
    """ref 切成 8 个交替单音，cover 有一个音偏高 150 音分 -> wrong_note + 命中率<1。"""
    wav = tmp_path / "f.wav"
    _mk_wav(wav)
    view = _mk_view()
    view.cv[:] = True
    seg = 50  # 2.5s 一个音，60/64 交替
    for k in range(8):
        if k % 2 == 1:
            view.rm[k * seg:(k + 1) * seg] = 64.0
    view.cm[:] = view.rm
    view.cm[250:300] += 1.5  # 12.5-15s（第二句内）这个音高了一个半半音
    ph = phrase_diagnostics(view, LRC, str(wav), global_slope=1.0)
    second = ph[1]
    assert "wrong_note" in [i["type"] for i in second["issues"]]
    nh = second["notes"]
    assert nh is not None and nh["total"] >= 3
    assert nh["hit"] == nh["total"] - 1 and nh["worst"][0]["direction"] == "偏高"
    # 第一句全唱准 -> 无 wrong_note
    assert "wrong_note" not in [i["type"] for i in ph[0]["issues"]]


def test_late_entry(tmp_path):
    """第二句开头帧映射到 ref 10.8s -> 进句晚 -> rhythm_late_entry。"""
    wav = tmp_path / "g.wav"
    _mk_wav(wav)
    view = _mk_view()
    view.cv[:] = True
    for k in range(5):
        view.mapping[200 + k] = 216 + k  # 0.9s 的映射偏移
    ph = phrase_diagnostics(view, LRC, str(wav), global_slope=1.0)
    assert ph[1]["entry_late_s"] is not None and ph[1]["entry_late_s"] >= 0.35
    assert "rhythm_late_entry" in [i["type"] for i in ph[1]["issues"]]


def test_phrase_summary_aggregation():
    from src.agent.phrase import phrase_summary
    phrases = [{"score": 50, "issues": [{"type": "tail_fall"}, {"type": "pitch_off"}]},
               {"score": 60, "issues": [{"type": "tail_fall"}]},
               {"score": 90, "issues": []}]
    s = phrase_summary(phrases)
    assert s["total"] == 3 and abs(s["avg_score"] - 66.7) < 0.2
    assert s["issue_counts"]["tail_fall"] == 2
    assert "tail_fall" in s["systemic"] and "pitch_off" not in s["systemic"]
    assert phrase_summary([])["total"] == 0
