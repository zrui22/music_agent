"""_json_safe 回归测试：诊断响应中的 NaN/Inf 浮点必须被置为 None，
否则 Starlette json.dumps 抛 ValueError 导致 /api/diagnose 500（前端报"评测结果非json"）。"""
import math

import numpy as np

from src.server.app import _json_safe


def test_nan_inf_replaced_with_none():
    bad = []
    obj = {
        "cost": float("inf"),
        "scores": {"pitch": float("nan"), "ok": 3.5},
        "tried": [1.0, np.float64("-inf"), "x", None, True],
        "nested": [{"a": np.float32("nan")}],
    }
    out = _json_safe(obj, _bad=bad)
    assert out["cost"] is None
    assert out["scores"]["pitch"] is None
    assert out["scores"]["ok"] == 3.5
    assert out["tried"] == [1.0, None, "x", None, True]
    assert out["nested"][0]["a"] is None
    assert len(bad) == 4
    import json
    json.dumps(out)  # 必须可序列化


def test_finite_values_untouched():
    obj = {"a": 1.5, "b": [0.0, -2.75], "c": "s", "d": None, "e": 3}
    assert _json_safe(obj) == obj
