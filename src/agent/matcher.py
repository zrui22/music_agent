"""哼唱匹配：用户唱段 F0 -> 缓存曲库检索最可能的歌曲

历代方案（留存对比）：
  v1 帧级 DTW：对"短 clip vs 长曲库"太宽松（错位幸运窗代价甚至低于真匹配）
  v2 pc 网格滑动互相关：50ms 格 + 12 音级 one-hot FFT，移调/伸缩枚举。
     弱点：刚性时序，业余演唱累积节奏漂移下命中率崩塌（合成抖动测试 0.22）
  v3 音程序列局部比对：pc 网格 -> 众数滤波 -> 折叠同音 run ->
     相邻音音程序列（mod 12，天然移调不变）-> rapidfuzz 局部比对扫全库。
     弱点：音符切分噪声大时判别力不足（真歌 ivl 分与随机歌拉不开差距）
  v4 自由起止子序列 DTW（默认）：100ms 粗网格 + numba JIT，全库扫描 ~秒级；
     DTW 天然吸收局部节奏漂移，合成抖动测试 0.85 vs 随机 0.077；
     直方图调差 ±1 半音三枚举；top-50 再用 50ms 细网格带状 DTW 精排。
"""
import numpy as np
from rapidfuzz import fuzz

from src.pitchref.align import estimate_key_shift

STEP = 0.05
STRETCHES = (0.75, 0.875, 1.0, 1.15, 1.3)  # 用户序列 -> 原曲时间轴 的拉伸比
PASS2_K = 50


def _pc_grid(times, midi, voiced, step: float = STEP) -> np.ndarray:
    """F0 -> pc 网格（int8，-1=无声）。"""
    if len(times) == 0:
        return np.full(0, -1, np.int8)
    n = int(times[-1] / step) + 1
    s = np.zeros(n)
    c = np.zeros(n)
    idx = np.clip((times[voiced] / step).astype(int), 0, n - 1)
    np.add.at(s, idx, midi[voiced])
    np.add.at(c, idx, 1)
    pc = np.full(n, -1, dtype=np.int8)
    hit = c > 0
    pc[hit] = (np.round(s[hit] / c[hit]).astype(int)) % 12
    return pc


def _note_seq(pc_grid: np.ndarray, min_run: int = 2):
    """pc 网格 -> (音符 pc 序列 np.int8, 各音符起始格 np.int32)。

    众数滤波(3格)去毛刺 -> 折叠同音 run -> 丢弃短于 min_run 的 run
    （装饰音/音高抖动），无声格断开 run。
    """
    g = pc_grid.astype(np.int16)
    n = len(g)
    if n < 3:
        return np.full(0, -1, np.int8), np.full(0, 0, np.int32)
    f = g.copy()  # 众数滤波：两侧相同则中间跟随
    same = (g[:-2] == g[2:]) & (g[:-2] >= 0)
    f[1:-1][same] = g[:-2][same]
    pcs, starts = [], []
    run_pc, run_start, run_len = -2, 0, 0  # -2 = 未开始
    def _flush(end):
        if run_pc >= 0 and run_len >= min_run:
            pcs.append(run_pc)
            starts.append(run_start)
    for i in range(n):
        p = int(f[i])
        if p == run_pc:
            run_len += 1
        else:
            _flush(i)
            run_pc, run_start, run_len = p, i, 1
    _flush(n)
    return np.array(pcs, np.int8), np.array(starts, np.int32)


def _interval_bytes(pcs: np.ndarray) -> bytes:
    """相邻音符音程序列 mod 12 -> bytes（移调不变表示）。"""
    if len(pcs) < 2:
        return b""
    return bytes((np.diff(pcs.astype(np.int16)) % 12).astype(np.uint8).tolist())


def _ivl_score(q_ivl: bytes, r_ivl: bytes):
    """局部比对得分 0-1 + 命中起始音符下标。"""
    if len(q_ivl) < 6 or len(r_ivl) < 6:
        return 0.0, 0
    aln = fuzz.partial_ratio_alignment(q_ivl, r_ivl)
    return aln.score / 100.0, aln.dest_start


def _slide_hit(u_pc: np.ndarray, r_pc: np.ndarray, stretch: float,
               soft: bool = False):
    """用户 pc 序列拉伸 stretch 后在原曲上滑动，返回 (最高命中率, 位置)。

    soft=True 时参考侧 one-hot 先做 [0.25,0.5,0.25] 平滑，给 ±1 格(50ms)
    的局部节奏抖动部分分——业余演唱的时值偏差是刚性与互相关的主要杀手。
    """
    n_u = int(round(len(u_pc) * stretch))
    if n_u < 20 or len(r_pc) < n_u:
        return 0.0, 0
    idx = np.clip((np.arange(n_u) / stretch).astype(int), 0, len(u_pc) - 1)
    u = u_pc[idx]
    m = u >= 0
    denom = int(m.sum())
    if denom < 20:
        return 0.0, 0
    L = 1 << (len(r_pc) + n_u - 1).bit_length()
    score = np.zeros(L)
    for c in range(12):
        rf1 = (r_pc == c).astype(np.float64)
        if soft:
            rf1 = np.convolve(rf1, [0.25, 0.5, 0.25], mode="same")
        rf = np.fft.rfft(rf1, L)
        uf = np.fft.rfft((u == c)[::-1].astype(np.float64), L)
        score += np.fft.irfft(rf * uf, L)
    valid = score[n_u - 1:len(r_pc)] / denom  # 完全重叠窗口
    if len(valid) == 0:
        return 0.0, 0
    p = int(np.argmax(valid))
    return float(valid[p]), p


def _shift_pc(u_pc: np.ndarray, ks: int) -> np.ndarray:
    out = u_pc.copy()
    hit = out >= 0
    out[hit] = (out[hit] + ks) % 12
    return out


def _best_hit(u_pc, r_pc, shifts, soft: bool = False,
              stretches: tuple = None) -> tuple:
    best, best_pos, best_ks, best_st = 0.0, 0, 0, 1.0
    for ks in shifts:
        u = _shift_pc(u_pc, ks)
        for st in (stretches or STRETCHES):
            h, p = _slide_hit(u, r_pc, st, soft=soft)
            if h > best:
                best, best_pos, best_ks, best_st = h, p, ks, st
    return best, best_pos, best_ks, best_st


def _pc_dist(a: int, b: int) -> float:
    """音级环距离（0-6 半音）；任一侧无声代价 1.0（略高于1半音错音，
    防止 DTW 靠"穿越无声帧"刷低代价）。"""
    if a < 0 or b < 0:
        return 1.0
    d = abs(a - b) % 12
    return float(min(d, 12 - d))


# ---- v4: numba 加速的自由起止子序列 DTW（pass1 全库扫描）----
try:
    from numba import njit as _njit
except ImportError:  # numba 不可用时退化为纯 numpy 慢速版（仅供调试）
    _njit = None


def _sdtw_py(q: np.ndarray, r: np.ndarray,
             gap_r: float = 0.7, gap_q: float = 1.5):
    """自由起止子序列 DTW（Needleman-Wunsch 空隙模型，纯 Python 参考版）。

    q 的每帧必须匹配（代价 d）或被跳过（代价 gap_q）；r 中段的跳过每帧
    代价 gap_r，前缀/后缀跳过免费（子序列定位）。按 q 长度归一，
    score = max(0, 1 - (cost/n)/1.5)。返回 (score, 起始格, 终止格)。
    代价：q无声帧 0.3；q有声对r无声 1.0；否则音级环距离。
    """
    n, m = len(q), len(r)
    if n < 10 or m < 10:
        return 0.0, 0, 0
    INF = float("inf")

    def dcost(qi, rj):
        if qi < 0:
            return 0.3
        if rj < 0:
            return 1.0
        d = abs(qi - rj) % 12
        return float(min(d, 12 - d))

    prev_c = np.full(m, INF)   # 已开始路径
    prev_s = np.zeros(m, dtype=np.int64)
    for i in range(n):
        cur_c = np.full(m, INF)
        cur_s = np.zeros(m, dtype=np.int64)
        qi = int(q[i])
        for j in range(m):
            d = dcost(qi, int(r[j]))
            if i == 0:
                # 首行：可在任意 j 起步（自由前缀），或整帧跳过
                best, bs = min(d, gap_q), j
            else:
                best, bs = INF, j
            if j > 0 and prev_c[j - 1] + d < best:
                best, bs = prev_c[j - 1] + d, prev_s[j - 1]
            if prev_c[j] + gap_q < best:
                best, bs = prev_c[j] + gap_q, prev_s[j]
            if j > 0 and cur_c[j - 1] + gap_r < best:
                best, bs = cur_c[j - 1] + gap_r, cur_s[j - 1]
            cur_c[j], cur_s[j] = best, bs
        prev_c, prev_s = cur_c, cur_s
    j = int(np.argmin(prev_c))
    cost = prev_c[j]
    if cost == INF:
        return 0.0, 0, 0
    s = max(0.0, 1.0 - (cost / n) / 1.5)
    return s, int(prev_s[j]), j


if _njit is not None:

    @_njit(cache=True)
    def _dtw_band_nb(u, r, band):
        """带状 DTW（起始自由/终点自由于末行 band 内），返回 (best_cost, best_len)。"""
        n, m = len(u), len(r)
        W = 2 * band + 1
        INF = 1e30
        prev = np.zeros(W)  # 起始自由：窗口内任意起点代价 0
        plen = np.zeros(W, dtype=np.int32)
        best_cost = INF
        best_len = 1
        for i in range(n):
            cur = np.full(W, INF)
            clen = np.zeros(W, dtype=np.int32)
            j0 = i - band
            if j0 < 0:
                j0 = 0
            j1 = i + band
            if j1 > m - 1:
                j1 = m - 1
            ui = u[i]
            for j in range(j0, j1 + 1):
                k = j - i + band
                rj = r[j]
                if ui < 0 or rj < 0:
                    d = 1.0
                else:
                    dd = abs(ui - rj) % 12
                    d = float(min(dd, 12 - dd))
                cc = prev[k]
                ll = plen[k]
                if k + 1 < W and prev[k + 1] < cc:
                    cc = prev[k + 1]
                    ll = plen[k + 1]
                if k - 1 >= 0 and cur[k - 1] < cc:
                    cc = cur[k - 1]
                    ll = clen[k - 1]
                if cc >= INF:
                    continue
                cur[k] = cc + d
                clen[k] = ll + 1
            if i == n - 1:
                for j in range(j0, j1 + 1):
                    k = j - i + band
                    if cur[k] < best_cost:
                        best_cost = cur[k]
                        best_len = clen[k]
                        if best_len < 1:
                            best_len = 1
            prev = cur
            plen = clen
        return best_cost, best_len


if _njit is not None:

    @_njit(cache=True)
    def _sdtw_nb(q, r, gap_r=0.7, gap_q=1.5):
        n, m = len(q), len(r)
        INF = 1e30
        prev_c = np.full(m, INF)
        prev_s = np.zeros(m, dtype=np.int64)
        for i in range(n):
            cur_c = np.full(m, INF)
            cur_s = np.zeros(m, dtype=np.int64)
            qi = q[i]
            for j in range(m):
                rj = r[j]
                if qi < 0:
                    d = 0.3
                elif rj < 0:
                    d = 1.0
                else:
                    dd = abs(qi - rj) % 12
                    d = float(min(dd, 12 - dd))
                if i == 0:
                    # 首行：可在任意 j 起步（自由前缀），或整帧跳过
                    best = d if d < gap_q else gap_q
                    bs = j
                else:
                    best = INF
                    bs = j
                if j > 0 and prev_c[j - 1] + d < best:
                    best = prev_c[j - 1] + d
                    bs = prev_s[j - 1]
                if prev_c[j] + gap_q < best:
                    best = prev_c[j] + gap_q
                    bs = prev_s[j]
                if j > 0 and cur_c[j - 1] + gap_r < best:
                    best = cur_c[j - 1] + gap_r
                    bs = cur_s[j - 1]
                cur_c[j] = best
                cur_s[j] = bs
            prev_c, prev_s = cur_c, cur_s
        best_c = INF
        best_j = 0
        for j in range(m):
            if prev_c[j] < best_c:
                best_c = prev_c[j]
                best_j = j
        if best_c >= INF:
            return 0.0, 0, 0
        s = 1.0 - (best_c / n) / 1.5
        return (s if s > 0.0 else 0.0), prev_s[best_j], best_j

    def _sdtw(q, r):
        """(score 0-1, 起始格, 终止格)；自由起止子序列 DTW（空隙模型）。"""
        if len(q) < 10 or len(r) < 10:
            return 0.0, 0, 0
        s, a, b = _sdtw_nb(q.astype(np.int8), r.astype(np.int8))
        return float(s), int(a), int(b)
else:
    _sdtw = _sdtw_py


def _downsample_pc(pc: np.ndarray, k: int = 2) -> np.ndarray:
    """pc 网格 k 倍降采样：窗口内众数（无声占多数则 -1）。"""
    n = len(pc) // k
    if n == 0:
        return pc[:0]
    w = pc[:n * k].reshape(n, k)
    voiced_cnt = (w >= 0).sum(1)
    counts = np.stack([(w == c).sum(1) for c in range(12)], axis=1)
    out = np.full(n, -1, np.int8)
    ok = voiced_cnt * 2 >= k  # 窗口内至少半数有声
    out[ok] = counts[ok].argmax(1).astype(np.int8)
    return out


def _dtw_window_score(u_pc: np.ndarray, r_pc: np.ndarray, pos: int,
                      stretch: float, band_frac: float = 0.15) -> float:
    """pass3 重排：在命中位置窗口内做带状 DTW，容忍局部节奏自由。

    返回 0-1 分：平均音级距离 0 → 1，≥1.5 半音 → 0。
    """
    n_u = int(round(len(u_pc) * stretch))
    if n_u < 20:
        return 0.0
    idx = np.clip((np.arange(n_u) / stretch).astype(int), 0, len(u_pc) - 1)
    u = u_pc[idx]
    margin = int(n_u * 0.05) + 2
    s = max(0, pos - margin)
    e = min(len(r_pc), pos + n_u + margin)
    r = r_pc[s:e]
    n, m_ = len(u), len(r)
    if m_ < 20:
        return 0.0
    band = max(int(n * band_frac), 15)
    if _njit is not None:
        best_cost, best_len = _dtw_band_nb(u.astype(np.int8), r.astype(np.int8),
                                           band)
        if best_cost >= 1e29:
            return 0.0
        mean_d = best_cost / max(int(best_len), 1)
        return max(0.0, 1.0 - mean_d / 1.5)
    W = 2 * band + 1
    INF = float("inf")
    prev = np.zeros(W)  # 起始自由：窗口内任意起点代价 0（命中位置本身是近似的）
    plen = np.zeros(W, dtype=np.int32)
    best_cost, best_len = INF, 1
    for i in range(n):
        cur = np.full(W, INF)
        clen = np.zeros(W, dtype=np.int32)
        j0 = max(0, i - band)
        j1 = min(m_ - 1, i + band)
        for j in range(j0, j1 + 1):
            k = j - i + band
            d = _pc_dist(int(u[i]), int(r[j]))
            c0, l0 = prev[k], plen[k]               # (i-1, j-1)
            c1 = prev[k + 1] if k + 1 < W else INF  # (i-1, j)
            l1 = plen[k + 1] if k + 1 < W else 0
            c2 = cur[k - 1] if k - 1 >= 0 else INF  # (i, j-1)
            l2 = clen[k - 1] if k - 1 >= 0 else 0
            cc, ll = min((c0, l0), (c1, l1), (c2, l2), key=lambda x: x[0])
            if cc == INF:
                continue
            cur[k] = cc + d
            clen[k] = ll + 1
        if i == n - 1:  # 终点可在最后一行 band 内任意处（窗口有冗余）
            for j in range(j0, j1 + 1):
                k = j - i + band
                if cur[k] < best_cost:
                    best_cost, best_len = cur[k], max(int(clen[k]), 1)
        prev, plen = cur, clen
    if best_cost == INF:
        return 0.0
    mean_d = best_cost / best_len
    return max(0.0, 1.0 - mean_d / 1.5)


def _item_pc(item: dict) -> np.ndarray:
    """参考项的 pc 网格（支持调用方预计算缓存）。"""
    r_pc = item.get("pc")
    if r_pc is None:
        r_t, r_m, r_v = item["f0"]
        r_pc = _pc_grid(r_t, r_m, r_v)
    return r_pc


def _item_ivl(item: dict):
    """参考项的 (音程序列 bytes, 音符起始格, 音符pc序列)（支持预计算缓存）。"""
    ivl = item.get("ivl")
    if ivl is None:
        pcs, starts = _note_seq(_item_pc(item))
        ivl = (_interval_bytes(pcs), starts, pcs)
        item["ivl"] = ivl
    return ivl


def _item_pc_coarse(item: dict) -> np.ndarray:
    """参考项的 100ms 粗网格 pc（支持预计算缓存）。"""
    c = item.get("pc_c")
    if c is None:
        c = _downsample_pc(_item_pc(item), 2)
        item["pc_c"] = c
    return c


def _ks_topk(c_midi, c_voiced, r_midi, r_voiced, k: int = 3) -> list:
    """直方图相关 top-k 调差（返回应施加于 query 的移调量列表，mod 12）。

    与 estimate_key_shift 同算法但返回前 k 个峰：直方图法在关系大小调/
    五度混淆时主峰可能错误，多候选由下游 DTW 自行裁决。
    """
    def hist12(midi, voiced):
        h = np.zeros(12)
        pc = np.mod(np.round(midi[voiced]).astype(int), 12)
        for p in pc:
            h[p] += 1
        return h / max(h.sum(), 1)

    hc, hr = hist12(c_midi, c_voiced), hist12(r_midi, r_voiced)
    if hc.std() == 0 or hr.std() == 0:
        return [0]
    scores = []
    for kk in range(-6, 7):
        s = np.corrcoef(np.roll(hr, kk), hc)[0, 1]
        scores.append((s, kk))
    scores.sort(key=lambda x: -x[0])
    return [(-kk) % 12 for _s, kk in scores[:k]]  # cover-kk≈ref → query 加 -kk


def _match_sdtw(u_pc, c_midi, c_voiced, lib: list, top_n: int) -> list:
    """v4：自由起止子序列 DTW 全库扫描（100ms 粗网格）+ 50ms 精排。

    pass1 对业余演唱的累积节奏漂移天然鲁棒（DTW 局部伸缩），
    直方图 top-3 调差候选补偿估计误差；pass2 在原网格上
    用 pass1 的精确起止位置做带状 DTW 精排。
    """
    q_c = _downsample_pc(u_pc, 2)
    if (q_c >= 0).sum() < 20:
        return []
    p1 = []
    for item in lib:
        r_c = _item_pc_coarse(item)
        if len(r_c) < len(q_c) // 2 + 10:
            continue
        try:
            shifts = _ks_topk(c_midi, c_voiced, *item["f0"][1:])
        except Exception:
            shifts = (0,)
        best_s, best_a, best_e, best_k = 0.0, 0, 0, 0
        for k_ in shifts:
            s, a, e = _sdtw(_shift_pc(q_c, k_), r_c)
            if s > best_s:
                best_s, best_a, best_e, best_k = s, a, e, k_
        if best_s > 0:
            p1.append((best_s, best_a, best_e, best_k, item))
    p1.sort(key=lambda x: -x[0])
    cands = []
    n_u = len(u_pc)
    for s1, a, e, ks, item in p1[:PASS2_K]:
        r_pc = _item_pc(item)
        start_f, end_f = a * 2, max(e * 2, a * 2 + 20)
        st = float(np.clip((end_f - start_f) / n_u, 0.6, 1.6))
        dtw = _dtw_window_score(_shift_pc(u_pc, ks), r_pc, start_f, st)
        cands.append({"song": item["song"], "artists": item["artists"],
                      "ref_id": item["ref_id"], "dir": item["dir"],
                      "hit": None, "ivl": None, "sdtw": round(s1, 3),
                      "dtw": round(dtw, 3),
                      "score": round(0.3 * s1 + 0.7 * dtw, 3),
                      "ref_pos_s": round(start_f * STEP, 1),
                      "key_shift": int(ks)})
    cands.sort(key=lambda x: -x["score"])
    return _finalize(cands, top_n)


def _match_union(u_pc, c_midi, c_voiced, lib: list, top_n: int,
                 per_k: int = 40, fast: bool = True) -> list:
    """v5 并集召回：三路互补 pass1（corr 软互相关 / sdtw 子序列DTW / ivl 音程）
    各取 top-per_k 按 ref_id 去重合并，pass2 软互相关 + 带状 DTW 精排。

    依据：三路方法在不同失败样本上排名互补（实测 corr#15/ivl#671 与
    ivl#15/corr#335 并存），单路 top-50 覆盖率不足。
    候选保留各路子分（p1_corr/p1_sdtw/p1_ivl）供离线调权。

    fast=True（默认，~12s/查询）：pass1a 在 100ms 粗网格上、3 伸缩比；
    pass1b 两层（top1 调差全库扫 -> top150 补 top3 调差）；
    pass2 调差 _ks_topk top-3 + 3 伸缩比 + numba 带状 DTW。
    fast=False 为穷举对照（~50s/查询）：细网格 pass1a、±1 三移调 pass1b、
    pass2 全 12 移调 × 5 伸缩比。
    """
    by_id = {}
    for it in lib:
        by_id.setdefault(it["ref_id"], it)
    q_c = _downsample_pc(u_pc, 2)
    # pass1a: corr（直方图调差 -> 单移调软互相关）
    p1a = []
    for item in lib:
        try:
            ks0 = estimate_key_shift(c_midi, c_voiced, *item["f0"][1:])
        except Exception:
            continue
        ks = (-ks0) % 12
        if fast:
            r_c = _item_pc_coarse(item)
            if len(r_c) < 30:
                continue
            h, _p, _k, _s = _best_hit(q_c, r_c, (ks,), soft=True,
                                      stretches=(0.875, 1.0, 1.15))
        else:
            r_pc = _item_pc(item)
            if len(r_pc) < 60:
                continue
            h, _p, _k, _s = _best_hit(u_pc, r_pc, (ks,), soft=True)
        p1a.append((h, item["ref_id"]))
    # pass1b: sdtw（粗网格子序列 DTW；fast=两层调差筛选）
    p1b = []
    if (q_c >= 0).sum() >= 20:
        tier1 = []
        for item in lib:
            r_c = _item_pc_coarse(item)
            if len(r_c) < len(q_c) // 2 + 10:
                continue
            try:
                if fast:
                    shifts = _ks_topk(c_midi, c_voiced, *item["f0"][1:], k=3)
                else:
                    ks0 = estimate_key_shift(c_midi, c_voiced, *item["f0"][1:])
                    ks = (-ks0) % 12
                    shifts = [ks, (ks - 1) % 12, (ks + 1) % 12]
            except Exception:
                shifts = [0]
            s0, _a, _e = _sdtw(_shift_pc(q_c, shifts[0]), r_c)
            tier1.append((s0, shifts, item))
        tier1.sort(key=lambda x: -x[0])
        n_deep = 150 if fast else len(tier1)
        for rank, (s0, shifts, item) in enumerate(tier1):
            bs = s0
            if rank < n_deep:
                r_c = _item_pc_coarse(item)
                for k_ in shifts[1:]:
                    s_, _a, _e = _sdtw(_shift_pc(q_c, k_), r_c)
                    if s_ > bs:
                        bs = s_
            p1b.append((bs, item["ref_id"]))
    # pass1c: ivl（音程序列局部比对）
    p1c = []
    q_pcs, _q_starts = _note_seq(u_pc)
    q_ivl = _interval_bytes(q_pcs)
    if len(q_ivl) >= 6:
        for item in lib:
            r_ivl, _rs, _rp = _item_ivl(item)
            if len(r_ivl) < 6:
                continue
            s, _np = _ivl_score(q_ivl, r_ivl)
            p1c.append((s, item["ref_id"]))
    # 并集
    sub = {}
    for lst, key in ((p1a, "p1_corr"), (p1b, "p1_sdtw"), (p1c, "p1_ivl")):
        lst.sort(key=lambda x: -x[0])
        for rank, (s, rid) in enumerate(lst[:per_k]):
            if s <= 0:
                continue
            d = sub.setdefault(rid, {})
            d[key] = (round(float(s), 3), rank + 1)
    # pass2: 并集候选软互相关 + 带状 DTW
    cands = []
    for rid, subs in sub.items():
        item = by_id[rid]
        r_pc = _item_pc(item)
        if fast:
            try:
                shifts = _ks_topk(c_midi, c_voiced, *item["f0"][1:], k=3)
            except Exception:
                shifts = [0]
            hit, pos, ks, st = _best_hit(u_pc, r_pc, shifts, soft=True,
                                         stretches=(0.875, 1.0, 1.15))
        else:
            hit, pos, ks, st = _best_hit(u_pc, r_pc, range(12), soft=True)
        dtw = _dtw_window_score(_shift_pc(u_pc, ks), r_pc, pos, st)
        c = {"song": item["song"], "artists": item["artists"],
             "ref_id": item["ref_id"], "dir": item["dir"],
             "hit": round(hit, 3), "dtw": round(dtw, 3),
             "ivl": subs.get("p1_ivl", [None])[0],
             "sdtw": subs.get("p1_sdtw", [None])[0],
             "p1_corr": subs.get("p1_corr", [None])[0],
             "ranks": {k: v[1] for k, v in subs.items()},
             # 权重来自 union100 离线网格调优（rerank_tune.py）：
             # dtw 是弱特征(单特征top1仅8.8%)，sdtw/hit 为主
             "score": round(0.4 * hit + 0.1 * dtw
                            + 0.5 * (subs.get("p1_sdtw", [0])[0] or 0), 3),
             "ref_pos_s": round(pos * STEP, 1), "key_shift": int(ks)}
        cands.append(c)
    cands.sort(key=lambda x: -x["score"])
    return _finalize(cands, top_n)


def humming_match(c_times, c_midi, c_voiced, lib: list,
                  top_n: int = 5, mode: str = "sdtw",
                  fast: bool = True) -> list:
    """返回候选 [{song, artists, ref_id, dir, hit, dtw, ivl, score, ref_pos_s,
    key_shift, margin, confidence}]；空列表=库空或用户发声不足。

    mode:
      union    三路 pass1 并集召回 + 精排（v5，召回率最高；fast=True ~12s）
      sdtw     子序列DTW全库扫描+精排（v4，默认；漂移鲁棒）
      corr     纯 pc 互相关（v2 刚性时序方案，留存对照）
      ivl      纯音程序列局部比对扫库（v3）
      ivl_dtw  音程比对粗排 top-50 -> 软互相关+带状DTW 精排（v3）
    fast 仅对 union 生效：False 为穷举对照模式（~50s，评估调权用）。
    """
    u_pc = _pc_grid(c_times, c_midi, c_voiced)
    if (u_pc >= 0).sum() < 40:
        return []
    if mode == "corr":
        return _match_corr(u_pc, c_midi, c_voiced, lib, top_n)
    if mode == "sdtw":
        return _match_sdtw(u_pc, c_midi, c_voiced, lib, top_n)
    if mode == "union":
        return _match_union(u_pc, c_midi, c_voiced, lib, top_n, fast=fast)
    q_pcs, q_starts = _note_seq(u_pc)
    q_ivl = _interval_bytes(q_pcs)
    if len(q_ivl) < 6:
        return []
    # pass 1：音程序列局部比对，全库粗排（C 实现，毫秒级/库）
    p1 = []
    for item in lib:
        r_ivl, r_starts, r_pcs = _item_ivl(item)
        if len(r_ivl) < 6:
            continue
        s, note_pos = _ivl_score(q_ivl, r_ivl)
        p1.append((s, note_pos, r_starts, item))
    p1.sort(key=lambda x: -x[0])
    cands = [{"song": it["song"], "artists": it["artists"], "ref_id": it["ref_id"],
              "dir": it["dir"], "ivl": round(s, 3),
              "ref_pos_s": round(float(r_starts[min(np_, len(r_starts)-1)]) * STEP, 1)
              if len(r_starts) else 0.0}
             for s, np_, r_starts, it in p1[:PASS2_K] if s > 0]
    if mode == "ivl" or not cands:
        for c in cands:
            c.update(hit=None, dtw=None, score=c["ivl"], key_shift=None)
        cands.sort(key=lambda x: -x["score"])
        return _finalize(cands, top_n)
    # pass 2：top-K 软互相关（全 12 移调枚举）拿命中率/位置/调差
    for c in cands:
        item = next(it for it in lib if it["ref_id"] == c["ref_id"])
        r_pc = _item_pc(item)
        hit, pos, ks, st = _best_hit(u_pc, r_pc, range(12), soft=True)
        c["hit"] = round(hit, 3)
        c["_pos"], c["_st"], c["_r_pc"], c["key_shift"] = pos, st, r_pc, int(ks)
    # pass 3：带状 DTW 精排（前 15）
    for c in cands[:15]:
        c["_dtw"] = _dtw_window_score(_shift_pc(u_pc, c["key_shift"]),
                                      c["_r_pc"], c["_pos"], c["_st"])
    for c in cands:
        c["dtw"] = round(c.pop("_dtw", 0.0), 3)
        c.pop("_pos"), c.pop("_st"), c.pop("_r_pc")
        c["score"] = round(0.55 * c["ivl"] + 0.30 * c["dtw"] + 0.15 * (c["hit"] or 0), 3)
    cands.sort(key=lambda x: -x["score"])
    return _finalize(cands, top_n)


def _finalize(cands: list, top_n: int) -> list:
    for i, c in enumerate(cands):
        margin = (c["score"] - cands[1]["score"]) if i == 0 and len(cands) > 1 else 0.0
        c["margin"] = round(margin, 3)
        conf = min(1.0, max(0.0,
                            (c["score"] - 0.30) / 0.40 * 0.75
                            + min(margin / 0.10, 1.0) * 0.25))
        c["confidence"] = round(conf, 3)
    return cands[:top_n]


def _match_corr(u_pc, c_midi, c_voiced, lib: list, top_n: int) -> list:
    """v2 留存对照：直方图移调 pass1 -> 全移调 pass2 -> DTW pass3。"""
    p1 = []
    for item in lib:
        r_pc = _item_pc(item)
        if len(r_pc) < 60:
            continue
        try:
            ks0 = estimate_key_shift(c_midi, c_voiced, *item["f0"][1:])
            ks = (-ks0) % 12  # cover - best_k ≈ ref → query 需加 -best_k
        except Exception:
            continue
        hit, pos, ks, _st = _best_hit(u_pc, r_pc, (ks,), soft=True)
        p1.append((hit, pos, ks, r_pc, item))
    p1.sort(key=lambda x: -x[0])
    cands = []
    for hit1, _, ks1, r_pc, item in p1[:PASS2_K]:
        hit, pos, ks, st = _best_hit(u_pc, r_pc, range(12), soft=True)
        dtw = _dtw_window_score(_shift_pc(u_pc, ks), r_pc, pos, st)
        cands.append({"song": item["song"], "artists": item["artists"],
                      "ref_id": item["ref_id"], "dir": item["dir"],
                      "hit": round(hit, 3), "dtw": round(dtw, 3), "ivl": None,
                      "score": round(0.45 * hit + 0.55 * dtw, 3),
                      "ref_pos_s": round(pos * STEP, 1), "key_shift": int(ks)})
    cands.sort(key=lambda x: -x["score"])
    return _finalize(cands, top_n)
