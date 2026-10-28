"""哼唱匹配核心纯函数测试（pc 网格 / 音程序列 / 环距离 / 合成旋律检索）。"""
import numpy as np

from src.agent import matcher
from src.agent.matcher import (_pc_grid, _note_seq, _interval_bytes, _pc_dist,
                               _downsample_pc, humming_match)


def _f0(melody_midi, hop=0.01, gap=0.0):
    """melody_midi: 每秒音高序列（0=无声）。返回 (times, midi, voiced)。"""
    times, midi, voiced = [], [], []
    t = 0.0
    for m in melody_midi:
        n = int(1.0 / hop)
        for _ in range(n):
            times.append(t)
            midi.append(float(m))
            voiced.append(m > 0)
            t += hop
        t += gap
    return np.array(times), np.array(midi), np.array(voiced, dtype=bool)


def test_pc_dist():
    assert _pc_dist(0, 0) == 0
    assert _pc_dist(0, 11) == 1        # 环距离取短边
    assert _pc_dist(0, 6) == 6
    assert _pc_dist(-1, 5) == 1.0      # 无声帧代价
    assert _pc_dist(5, -1) == 1.0


def test_pc_grid_voiced_and_unvoiced():
    # step=0.05：t=0.12 落在第 3 格，grid 共 3 格
    times = np.array([0.0, 0.01, 0.05, 0.06, 0.12])
    midi = np.array([60.0, 60.2, 67.0, 0.0, 0.0])
    voiced = np.array([True, True, True, False, False])
    pc = _pc_grid(times, midi, voiced, step=0.05)
    assert len(pc) == 3
    assert pc[0] == 0                  # mean(60, 60.2)=60.1 -> 60 % 12
    assert pc[1] == 7                  # 67 % 12（0.06s 无声帧不计入）
    assert pc[2] == -1                 # 无声格


def test_note_seq_and_intervals():
    # 每个音符 3 格（>=min_run），无声格断开
    pc = np.array([0, 0, 0, -1, 2, 2, 2, -1, 5, 5, 5], dtype=np.int8)
    pcs, starts = _note_seq(pc)
    assert list(pcs) == [0, 2, 5]
    assert _interval_bytes(pcs) == bytes([2, 3])


def test_downsample_pc_majority():
    pc = np.array([3, 3, -1, -1, 7, 7], dtype=np.int8)
    out = _downsample_pc(pc, 2)
    assert list(out) == [3, -1, 7]


def test_humming_match_finds_true_song():
    """合成检索：query 是真歌中段（移调 +2 半音），应命中真歌。

    旋律用结构化音级（自然音阶 vs 全音阶，音级分布/轮廓明显不同）——
    随机均匀旋律的音级直方图无峰，会让调差估计和 DTW 都失去判别力，
    不符合真实曲库的数据假设。
    """
    # 真歌主题：C 大调自然音阶级进（音级集中 {0,2,4,5,7,9,11}）
    melody_a = np.array([60, 62, 64, 65, 67, 65, 64, 62,
                         60, 62, 64, 67, 69, 67, 64, 62])
    # 干扰曲：全音阶轮廓（音级集中 {1,3,6,8,10}，与真歌错开）
    melody_b = np.array([61, 63, 66, 68, 70, 73, 70, 68,
                         66, 63, 61, 63, 66, 61, 58, 56])
    # 原曲：前后拼其它旋律片段，中间是 melody_a
    pad_pre = np.array([55, 59, 62, 59, 55, 52])
    pad_post = np.array([67, 64, 60, 64, 67, 72])
    song_ref = np.concatenate([pad_pre, melody_a, pad_post])
    rt, rm, rv = _f0(song_ref)
    dt, dm, dv = _f0(melody_b)
    lib = [
        {"song": "真歌", "artists": "甲", "ref_id": 1, "dir": None,
         "f0": (rt, rm, rv)},
        {"song": "干扰", "artists": "乙", "ref_id": 2, "dir": None,
         "f0": (dt, dm, dv)},
    ]
    # 用户唱 melody_a 的中段 10 秒，整体高 2 个半音
    qt, qm, qv = _f0(melody_a[3:13] + 2)
    for mode in ("sdtw", "union"):
        cands = humming_match(qt, qm, qv, lib, mode=mode)
        assert cands, f"{mode} 应返回候选"
        assert cands[0]["song"] == "真歌", f"{mode} top1 应为真歌: {cands[:2]}"


def test_humming_match_empty_when_short():
    # 发声不足 2s（40 个 50ms 格）直接返回空
    t, m, v = _f0([60])
    assert humming_match(t, m, v, [{"song": "x", "artists": "", "ref_id": 1,
                                    "dir": None, "f0": (t, m, v)}]) == []
