"""翻唱-原曲对齐与 clip 音准/节奏打分

流程：调性平移估计(12音级直方图) -> 按长静音切 cover 发声段 ->
每段独立子序列DTW(circular半音距) -> 拼接为统一对齐视图 ->
clip 内帧级音分误差 / 八度错误 / DTW路标节奏指标聚合。

为什么分段：K歌短版常跳过间奏、或把同一段旋律唱两遍；单条全局子序列 DTW
无法回跳，会在结构差异处"漂"出垃圾对齐（实测 cover 196s 被压进 ref 5-99s）。
每段独立对齐后，两遍副歌各自匹配最优原曲区间，结构差异天然免疫。
"""
import re

import numpy as np


def estimate_key_shift(cover_midi: np.ndarray, cover_voiced: np.ndarray,
                       ref_midi: np.ndarray, ref_voiced: np.ndarray) -> int:
    """12 音级直方图相关峰估计翻唱升降 key（半音数 -6..+6）。"""
    return estimate_key_shift_topk(cover_midi, cover_voiced, ref_midi, ref_voiced,
                                   topk=1)[0][0]


def estimate_key_shift_topk(cover_midi: np.ndarray, cover_voiced: np.ndarray,
                            ref_midi: np.ndarray, ref_voiced: np.ndarray,
                            topk: int = 3) -> list:
    """top-k 调差候选 [(ks, 相关分)] 按分降序，符号同 estimate_key_shift。

    单峰直方图在短唱段（15-30s）直方图稀疏，易混淆五度/关系大小调
    （matcher._ks_topk 同款失败模式），返回多个候选供对齐层投票。
    """
    def hist12(midi, voiced):
        h = np.zeros(12)
        pc = np.mod(np.round(midi[voiced]).astype(int), 12)
        for p in pc:
            h[p] += 1
        return h / max(h.sum(), 1)

    hc, hr = hist12(cover_midi, cover_voiced), hist12(ref_midi, ref_voiced)
    if hc.std() == 0 or hr.std() == 0:
        return [(0, 0.0)]
    scored = [(k, float(np.corrcoef(np.roll(hr, k), hc)[0, 1]))
              for k in range(-6, 7)]
    scored.sort(key=lambda x: -x[1])
    return scored[:topk]


_CREDIT_RE = re.compile(
    r"[：:]|录音|混音|制作|作词|作曲|编曲|监制|吉他|贝斯|鼓|键盘|和声|出品|发行|"
    r"词\s|曲\s|OP\b|SP\b", re.I)


def vocal_mask_from_lrc(lrc, ref_times, pre_s: float = 0.3,
                        post_s: float = 0.2, tail_s: float = 8.0) -> np.ndarray:
    """LRC 歌词行时间窗 -> ref 帧级人声掩码（True=保留）。

    伴奏/间奏段 Praat 常把器乐误判为 voiced（实测《十年》前奏 0-10s 纯器乐段
    voiced 占比 0.80-0.92），这些帧污染调差直方图，并在 DTW 里充当错误锚点。
    只保留歌词行窗 [行首-pre_s, 下一行首+post_s] 内的帧；含冒号/制作名单关键词
    的 credits 行不算歌词窗。无有效歌词行时返回全 True（不掩码）。
    """
    mask = np.ones(len(ref_times), dtype=bool)
    if not lrc or len(ref_times) == 0:
        return mask
    lines = sorted((float(t), str(x)) for t, x in lrc)
    lines = [(t, x) for t, x in lines if x.strip() and not _CREDIT_RE.search(x)]
    if not lines:
        return mask
    mask[:] = False
    rt = np.asarray(ref_times)
    starts = [t for t, _ in lines]
    for i, (t0, _) in enumerate(lines):
        t1 = starts[i + 1] if i + 1 < len(starts) else t0 + tail_s
        mask |= (rt >= t0 - pre_s) & (rt <= t1 + post_s)
    return mask


def _downsample(times, midi, voiced, step_s=0.05):
    """10ms 帧 -> 50ms 帧（中位数），DTW 规模缩 5 倍。"""
    if len(times) == 0:
        return times, midi, voiced
    t0, t1 = times[0], times[-1]
    edges = np.arange(t0, t1 + step_s, step_s)
    idx = np.digitize(times, edges)
    out_t, out_m, out_v = [], [], []
    for b in range(1, len(edges) + 1):
        m = idx == b
        if not m.any():
            continue
        vv = voiced[m]
        mm = midi[m][vv]
        out_t.append(edges[b - 1] + step_s / 2)
        out_m.append(np.median(mm) if len(mm) else 0.0)
        out_v.append(vv.mean() > 0.5)
    return np.array(out_t), np.array(out_m), np.array(out_v, dtype=bool)


def circular_dist(a, b):
    d = np.abs(a - b) % 12
    return np.minimum(d, 12 - d)


class SubseqDTW:
    """子序列 DTW：cover(已平移调) 片段 对 ref 任一连续区间。unvoiced 帧不参与代价。"""

    def __init__(self, cover_times, cover_midi, cover_voiced,
                 ref_times, ref_midi, ref_voiced, step_s=0.05):
        self.step = step_s
        self.ct, self.cm, self.cv = _downsample(cover_times, cover_midi, cover_voiced, step_s)
        self.rt, self.rm, self.rv = _downsample(ref_times, ref_midi, ref_voiced, step_s)

    def run(self, gap_pen=1.5, step_pen=0.15, ref_t0=None, ref_t1=None):
        """反传动线向量化 DTW：同一反对角线上的格子互不依赖，numpy 整行算。

        step_pen: 非对角步（横/纵走）的加性罚分（半音/帧）。circular 音级距离是
        弱特征，不加斜率先验时路径会在"哪都差不多"的区域蠕行（实测 cover 196s
        压进 ref 94s）。加性罚分线性惩罚无谓摆动，真实的速度差仍可通过少量
        非对角步表达。注意不能用乘性偏置：驻留同列时罚约每帧复合增长，
        长段会指数爆炸（实测 1.2^204≈1e16 污染整行）。

        ref_t0/ref_t1: 限制 ref 搜索窗（秒，重锚定用），返回的 mapping 仍是
        全曲 ref 网格下标。
        """
        cm, cv, rm, rv = self.cm, self.cv, self.rm, self.rv
        offset = 0
        if ref_t0 is not None and ref_t1 is not None:
            w = (self.rt >= ref_t0) & (self.rt <= ref_t1)
            if not w.any():
                return np.full(len(cm), -1), np.inf
            offset = int(np.argmax(w))
            rm, rv = rm[w], rv[w]
        n, m = len(cm), len(rm)
        if n == 0 or m == 0:
            return np.full(n, -1), np.inf
        INF = 1e18
        diff = np.abs(cm[:, None].astype(np.float32) - rm[None, :].astype(np.float32)) % 12
        cdist = np.minimum(diff, 12 - diff)
        C = np.where(cv[:, None] & rv[None, :], cdist, np.float32(gap_pen))
        D = np.full((n + 1, m + 1), INF, dtype=np.float32)
        D[0, :] = 0.0  # 子序列：ref 起点自由
        for k in range(2, n + m + 1):
            i_lo, i_hi = max(1, k - m), min(n, k - 1)
            if i_lo > i_hi:
                continue
            ii = np.arange(i_lo, i_hi + 1)
            jj = k - ii
            D[ii, jj] = C[ii - 1, jj - 1] + np.minimum(
                D[ii - 1, jj - 1],
                step_pen + np.minimum(D[ii - 1, jj], D[ii, jj - 1]))
        j_end = int(np.argmin(D[n, 1:])) + 1  # ref 终点自由
        best_cost = float(D[n, j_end]) / max(n, 1)
        mapping = np.full(n, -1)
        i, j = n, j_end
        while i > 0 and j > 0 and np.isfinite(D[i, j]) and D[i, j] < INF / 2:
            mapping[i - 1] = j - 1
            steps = [(D[i - 1, j - 1], i - 1, j - 1),
                     (D[i - 1, j] + step_pen, i - 1, j),
                     (D[i, j - 1] + step_pen, i, j - 1)]
            _, i, j = min(steps, key=lambda s: s[0])
        if offset:
            mapping[mapping >= 0] += offset
        return mapping, best_cost


class AlignedView:
    """多段对齐拼接后的统一视图。

    ct/cm/cv: cover 50ms 网格（全局时间，仅含对齐成功帧）
    mapping: 每帧 -> ref 50ms 网格 idx
    bounds: 各段在拼接数组里的分界下标（prefix sum）
    seg_costs: 每段 DTW 平均代价
    """

    def __init__(self, ct, cm, cv, rm, rv, mapping, step, bounds, seg_costs):
        self.ct, self.cm, self.cv = ct, cm, cv
        self.rm, self.rv = rm, rv
        self.mapping, self.step = mapping, step
        self.bounds, self.seg_costs = bounds, seg_costs


def segment_voiced(times, voiced, min_gap_s=1.5, min_dur_s=2.0, max_dur_s=None):
    """按长静音切 cover 发声段：[(t0,t1)]（全局时间）。短于 min_dur 的段丢弃。

    max_dur_s：超长段硬切上限。K歌混音里伴奏常被 Praat 判为 voiced，
    154s/177s 的"连续段"实际跨多次重复演唱，单调 DTW 无法回跳；
    硬切成 ≤30s 块后每块独立匹配原曲最优窗口，重复段天然免疫。
    """
    segs, start, last = [], None, None
    for t, v in zip(times, voiced):
        if v:
            if start is None:
                start = t
            elif t - last > min_gap_s:
                if last - start >= min_dur_s:
                    segs.append((start, last))
                start = t
            last = t
    if start is not None and last - start >= min_dur_s:
        segs.append((start, last))
    if max_dur_s:
        out = []
        for t0, t1 in segs:
            while t1 - t0 > max_dur_s:
                out.append((t0, t0 + max_dur_s))
                t0 += max_dur_s
            out.append((t0, t1))
        segs = out
    return segs


def align_multi(c_times, c_midi, c_voiced, r_times, r_midi, r_voiced,
                step_s=0.05, min_gap_s=0.8, max_cost=2.5):
    """短段子序列对齐（乐句级）。

    min_gap_s=0.8 在呼吸/乐句间隙处切开：每句独立匹配原曲最优位置，
    重复副歌/跳间奏的结构性差异天然免疫。max_cost 过滤对齐失败的段
    （circular 音级距离 >2.5 级/帧 视为误匹配）。
    """
    segs = segment_voiced(c_times, c_voiced, min_gap_s=min_gap_s, max_dur_s=30.0)
    if not segs:
        return None
    rm = rv = None
    cts, cms, cvs, maps, costs = [], [], [], [], []
    for t0, t1 in segs:
        m = (c_times >= t0) & (c_times <= t1)
        if m.sum() < 100:  # 10ms 帧 <1s 跳过
            continue
        dtw = SubseqDTW(c_times[m], c_midi[m], c_voiced[m],
                        r_times, r_midi, r_voiced, step_s)
        mapping, cost = dtw.run()
        if cost > max_cost:  # 对齐失败，弃段
            continue
        if rm is None:
            rm, rv = dtw.rm, dtw.rv
        ok = mapping >= 0
        if ok.sum() < 10:
            continue
        cts.append(dtw.ct[ok]); cms.append(dtw.cm[ok]); cvs.append(dtw.cv[ok])
        maps.append(mapping[ok]); costs.append((cost, int(ok.sum())))
    if not cts:
        return None
    bounds = np.cumsum([0] + [len(a) for a in maps])
    return AlignedView(np.concatenate(cts), np.concatenate(cms), np.concatenate(cvs),
                       rm, rv, np.concatenate(maps), step_s, bounds, costs)


def _wmedian(x, w):
    idx = np.argsort(x)
    x, w = np.asarray(x)[idx], np.asarray(w, dtype=float)[idx]
    cw = np.cumsum(w)
    return float(x[min(np.searchsorted(cw, 0.5 * cw[-1]), len(x) - 1)])


def reanchor_segments(view: AlignedView, c_times, c_midi, c_voiced,
                      r_times, r_midi, r_voiced, dev_thr_s=8.0, win_s=6.0,
                      improve=0.15, max_cost=2.5) -> AlignedView:
    """全局顺序一致性重锚定。

    逐段独立 DTW 无顺序约束：主歌/副歌同调、音级集合相似时，个别乐句会锁进
    副歌（用户按顺序唱时该落点物理上不可能）。用全体段落点拟合 cover→ref
    主趋势（斜率取全局路标斜率，截距取帧数加权中位数，抗离群），偏离
    >dev_thr 的段在 [趋势预测±win] 限制窗内重跑 SubseqDTW，代价改善
    >=improve 才接受——用户真乱序唱且唱得准时代价不会改善，不会被强制搬家。

    c_midi 必须是已平移调（与 view 生成时一致）的版本。
    """
    nseg = len(view.bounds) - 1
    if nseg < 3:
        return view
    rt, _, _ = _downsample(r_times, r_midi, r_voiced, view.step)
    if len(rt) == 0:
        return view
    seg_ids, c_mid, r_mid, w, costs = [], [], [], [], []
    for i in range(nseg):
        lo, hi = view.bounds[i], view.bounds[i + 1]
        mp = view.mapping[lo:hi]
        ok = mp[mp >= 0]
        if len(ok) < 10:
            continue
        seg_ids.append(i)
        c_mid.append(0.5 * (view.ct[lo] + view.ct[hi - 1]))
        r_mid.append(float(rt[int(np.median(ok))]))
        w.append(hi - lo)
        costs.append(view.seg_costs[i][0])
    if len(seg_ids) < 3:
        return view
    a = global_mapping_slope(view)
    if not (0.2 < a < 5):
        a = 1.0
    b = _wmedian(np.array(r_mid) - a * np.array(c_mid), w)
    resid = np.array(r_mid) - (a * np.array(c_mid) + b)
    outliers = np.abs(resid) > dev_thr_s
    if not outliers.any() or outliers.all():
        return view  # 无离群或全体离群（趋势无定义，多为整段唱错歌）

    c_times = np.asarray(c_times)
    new_ct, new_cm, new_cv, new_map, new_costs = [], [], [], [], []
    for i in range(nseg):
        lo, hi = view.bounds[i], view.bounds[i + 1]
        if i in set(np.array(seg_ids)[outliers]):
            t0, t1 = view.ct[lo], view.ct[hi - 1]
            pred0, pred1 = a * t0 + b - win_s, a * t1 + b + win_s
            m = (c_times >= t0 - 0.1) & (c_times <= t1 + 0.1)
            if m.sum() >= 100:
                dtw = SubseqDTW(c_times[m], c_midi[m], c_voiced[m],
                                r_times, r_midi, r_voiced, view.step)
                mp2, cost2 = dtw.run(ref_t0=pred0, ref_t1=pred1)
                ok2 = mp2 >= 0
                old_cost = view.seg_costs[i][0]
                if (ok2.sum() >= 10 and cost2 <= max_cost
                        and cost2 <= old_cost * (1 - improve)):
                    new_ct.append(dtw.ct[ok2]); new_cm.append(dtw.cm[ok2])
                    new_cv.append(dtw.cv[ok2]); new_map.append(mp2[ok2])
                    new_costs.append((cost2, int(ok2.sum())))
                    continue
        new_ct.append(view.ct[lo:hi]); new_cm.append(view.cm[lo:hi])
        new_cv.append(view.cv[lo:hi]); new_map.append(view.mapping[lo:hi])
        new_costs.append(view.seg_costs[i])
    bounds = np.cumsum([0] + [len(a_) for a_ in new_map])
    return AlignedView(np.concatenate(new_ct), np.concatenate(new_cm),
                       np.concatenate(new_cv), view.rm, view.rv,
                       np.concatenate(new_map), view.step, bounds, new_costs)


def align_best(c_times, c_midi, c_voiced, r_times, r_midi, r_voiced,
               lrc=None, topk=3, short_s=15.0, reanchor=True, **kw):
    """对齐主入口：LRC 人声窗掩码 -> 调差投票 -> 全局顺序重锚定。

    返回 (view, ks)；无法对齐返回 (None, 0)。
    - LRC 掩码：过滤伴奏段被 Praat 误判 voiced 的污染帧，调差直方图与 DTW
      同时生效（ref_f0 缓存文件不动，掩码在使用时现算）。
    - 调差投票：单峰直方图在短唱段易错五度/关系大小调，逐候选全量对齐，
      按 seg_costs 帧数加权总代价选最优。发声时长 <short_s 时直方图基本
      无信息（实测单句 4s 场景 top3 全错、正确调差相关分排第 5），改为
      全 13 候选投票——短段 DTW 快，代价曲面才是强信号（正确调差代价
      0.495 vs 错误最优 0.93）。
    - view.ambiguous: 最优与次优候选代价差 <0.1 时置 True（落点不可信，
      上游应提示而非自信展示）。
    """
    r_voiced = np.asarray(r_voiced)
    if lrc:
        r_voiced = r_voiced & vocal_mask_from_lrc(lrc, r_times)
    c_midi = np.asarray(c_midi, dtype=float)
    c_times = np.asarray(c_times)
    c_voiced = np.asarray(c_voiced)
    if len(c_times) > 1:
        frame_s = float(np.median(np.diff(c_times)))
        voiced_s = float(c_voiced.sum()) * frame_s
    else:
        voiced_s = 0.0
    k_eff = 13 if voiced_s < short_s else topk
    best, runner = None, None
    for ks, _s in estimate_key_shift_topk(c_midi, c_voiced, r_midi, r_voiced,
                                          topk=k_eff):
        view = align_multi(c_times, c_midi - ks, c_voiced,
                           r_times, r_midi, r_voiced, **kw)
        if view is None or len(view.ct) < 40:  # 2s 下限；更严阈值由调用方把控
            continue
        if reanchor:
            view = reanchor_segments(view, c_times, c_midi - ks, c_voiced,
                                     r_times, r_midi, r_voiced,
                                     max_cost=kw.get("max_cost", 2.5))
        n_tot = sum(n for _, n in view.seg_costs)
        cost = sum(c * n for c, n in view.seg_costs) / max(n_tot, 1)
        if best is None or cost < best[0]:
            runner = best
            best = (cost, ks, view)
        elif runner is None or cost < runner[0]:
            runner = (cost, ks, view)
    if best is None:
        return None, 0
    margin = (runner[0] - best[0]) if runner else float("inf")
    best[2].ambiguous = bool(margin < 0.1)
    best[2].align_margin = round(float(margin), 3)
    return best[2], best[1]


def clip_pitch_score(clip_start, clip_end, view: AlignedView, mapping=None):
    """clip 窗口内对齐帧 -> {median_abs_cents, in_tune_ratio, octave_off_ratio, n_frames, pitch_ref}。

    octave_off_ratio：原始(非循环)音差落在 [10,14] 半音的帧占比——整段高/低八度
    （含男唱女歌低八度、Praat 八度跳变噪声）会被循环距离漏掉，显式捕获。
    有效 50ms 帧 <8 返回 None。
    """
    if mapping is None:
        mapping = view.mapping
    ct = view.ct
    sel = (ct >= clip_start) & (ct < clip_end) & view.cv & (mapping >= 0)
    if sel.sum() < 8:
        return None
    cents_err, raw_err = [], []
    for gi in np.where(sel)[0]:
        rj = mapping[gi]
        if not view.rv[rj]:
            continue
        raw = float(view.cm[gi] - view.rm[rj])
        raw_err.append(raw)
        diff = abs(raw) % 12
        cents_err.append(min(diff, 12 - diff) * 100)
    if len(cents_err) < 8:
        return None
    cents_err = np.array(cents_err)
    raw_err = np.abs(np.array(raw_err))
    med = float(np.median(cents_err))
    ratio = float((cents_err <= 50).mean())
    octave = float(((raw_err >= 10) & (raw_err <= 14)).mean())
    return {"median_abs_cents": round(med, 1), "in_tune_ratio": round(ratio, 3),
            "octave_off_ratio": round(octave, 3), "n_frames": len(cents_err),
            "pitch_ref": round(float(np.clip(5 - med / 25, 0, 5)), 2)}


def _landmark_series(ct, sel_idx, rj, win_s=1.0):
    """frame 级 mapping -> 1s 路标（每窗 ref idx 中位数），抗 DTW 阶梯量化噪声。"""
    t0 = ct[sel_idx[0]]
    centers, meds = [], []
    w0 = t0
    while True:
        m = (ct[sel_idx] >= w0) & (ct[sel_idx] < w0 + win_s)
        if m.sum() >= 3:
            centers.append(float(np.mean(sel_idx[m])))
            meds.append(float(np.median(rj[m])))
        w0 += win_s
        if w0 > ct[sel_idx[-1]]:
            break
    return np.array(centers), np.array(meds)


def _longest_run(flags):
    """布尔数组最长连续 True 段的 (start, end) 切片下标；无则 None。"""
    best, cur = None, None
    for i, f in enumerate(flags):
        if f and cur is None:
            cur = i
        elif not f and cur is not None:
            if best is None or i - cur > best[1] - best[0]:
                best = (cur, i)
            cur = None
    if cur is not None and (best is None or len(flags) - cur > best[1] - best[0]):
        best = (cur, len(flags))
    return best


def clip_rhythm_score(clip_start, clip_end, view: AlignedView, mapping=None,
                      global_slope: float = None):
    """clip 节奏指标（1s 路标级）。

    - 跨段边界/对齐跳变处 mapping 会突变，按跳变切成 run，只用最长 run 拟合
    - tempo_ratio = 局部路标斜率 / 全局斜率；>1.05 拖拍 <0.95 赶拍
    - timing_jitter_ms = 路标拟合残差 std（ms，1s 尺度，不反映 sub-second 卡拍）
    - timing_dropout = clip 内 cover 发声但未对齐帧占比
    """
    if mapping is None:
        mapping = view.mapping
    ct = view.ct
    sel = (ct >= clip_start) & (ct < clip_end) & view.cv
    if sel.sum() < 20:
        return None
    sel_idx = np.where(sel)[0]
    rj_all = mapping[sel_idx].astype(float)
    dropout = float((rj_all < 0).mean())
    keep = rj_all >= 0
    sel_idx, rj = sel_idx[keep], rj_all[keep]
    if len(sel_idx) < 15:
        return None
    # 对齐跳变切 run（ref 跳 >0.75s 或 cover 网格不连续）
    jumps = np.where((np.abs(np.diff(rj)) > 15) | (np.diff(sel_idx) > 3))[0]
    runs, prev = [], 0
    for jp in jumps:
        runs.append((prev, jp + 1))
        prev = jp + 1
    runs.append((prev, len(sel_idx)))
    run = max(runs, key=lambda r: r[1] - r[0])
    if run[1] - run[0] < 15:
        return None
    s_idx, s_rj = sel_idx[run[0]:run[1]], rj[run[0]:run[1]]
    centers, meds = _landmark_series(ct, s_idx, s_rj)
    if len(centers) < 4:
        return None
    a, b = np.polyfit(centers, meds, 1)
    resid = meds - (a * centers + b)
    jitter_ms = float(resid.std() * view.step * 1000)
    tempo_ratio = float(a / global_slope) if global_slope and global_slope > 0 else 1.0
    score = rhythm_score_from_raw(jitter_ms, tempo_ratio, dropout)
    return {"tempo_ratio": round(tempo_ratio, 3),
            "timing_jitter_ms": round(jitter_ms, 1),
            "timing_dropout": round(dropout, 3),
            "rhythm_ref": score}


def rhythm_score_from_raw(jitter_ms: float, tempo_ratio: float,
                          dropout: float = 0.0) -> float:
    """jitter/tempo -> 0-5 节奏分（2026-10-27 按 K歌轨实测分布重标定）。

    旧公式 5-jitter/30-|tempo-1|*25 在 1s 路标尺度下过苛（jitter 中位
    140ms 即归零），且 tempo_ratio 在对齐崩溃时取到 0.099 这类荒谬值
    却照样重罚。新标定：
      jitter 主项：40ms->5 分，405ms->0 分（实测 p10=43.7 / p95=407.4）
      tempo 罚项：|ratio-1|<=0.12 免罚，超出部分 *8，封顶 2.5 分；
                  ratio 越出 [0.5,2] 视为对齐崩溃，只信 jitter
      dropout>0.5 时对齐不可靠，总分 8 折
    """
    sj = float(np.clip(5 - max(0.0, jitter_ms - 40) / 73, 0, 5))
    if 0.5 <= tempo_ratio <= 2.0:
        sj -= min(2.5, max(0.0, abs(tempo_ratio - 1) - 0.12) * 8)
    score = float(np.clip(sj, 0, 5))
    if dropout > 0.5:
        score *= 0.8
    return round(score, 2)


def global_mapping_slope(view: AlignedView) -> float:
    """全局速度比：各段 1s 路标斜率的中位数（段间跳变天然剔除）。"""
    slopes = []
    for i in range(len(view.bounds) - 1):
        lo, hi = view.bounds[i], view.bounds[i + 1]
        if hi - lo < 40:
            continue
        sel_idx = np.arange(lo, hi)
        centers, meds = _landmark_series(view.ct, sel_idx, view.mapping[sel_idx].astype(float))
        if len(centers) < 6:
            continue
        a, _ = np.polyfit(centers, meds, 1)
        if 0.2 < a < 5:
            slopes.append(a)
    return float(np.median(slopes)) if slopes else 1.0
