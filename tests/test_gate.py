"""歌唱门控 _singing_gate 的分支测试（纯 numpy，无需模型/音频文件）。"""
import numpy as np

from src.agent.diagnose import _singing_gate


def _mk(voiced, hop=0.01):
    voiced = np.asarray(voiced, dtype=bool)
    times = np.arange(len(voiced)) * hop
    return times, voiced


def test_silent_rejected():
    t, v = _mk(np.ones(1000))
    g = _singing_gate(t, v, rms_db=-60.0)
    assert g["sing"] is False and g["reason"] == "silent"


def test_noise_rejected():
    t, v = _mk(np.zeros(1000))  # 有声道但提不出音高
    g = _singing_gate(t, v, rms_db=-20.0)
    assert g["sing"] is False and g["reason"] == "no_voice"


def test_speech_like_rejected():
    # 发声占比够高，但最长连续发声只有 0.3s（说话式短促）
    v = np.zeros(1000, dtype=bool)
    for s in range(0, 1000, 100):
        v[s:s + 30] = True
    t, vv = _mk(v)
    g = _singing_gate(t, vv, rms_db=-20.0)
    assert g["sing"] is False and g["reason"] == "speech_like"
    assert g["longest_voiced_s"] < 0.6


def test_singing_passed():
    v = np.zeros(2000, dtype=bool)
    v[100:400] = True   # 3s 连续长音
    v[600:900] = True
    t, vv = _mk(v)
    g = _singing_gate(t, vv, rms_db=-18.0)
    assert g["sing"] is True and g["reason"] == "ok"
    assert g["longest_voiced_s"] >= 2.9
    assert 0.2 < g["voiced_ratio"] < 0.4
