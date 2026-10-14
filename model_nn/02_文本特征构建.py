# -*- coding: utf-8 -*-
"""
文本标签特征构建（方案A：政策事件日历 -> 日度衰减特征）

逻辑：
  1. 读取 LLM 辅助标注的政策事件日历（data/out_site/文本标签_政策事件日历.csv）
     - direction: +1 利多债市（收益率下行） / -1 利空债市 / 0 中性
     - intensity: 1 一般 / 2 重要 / 3 重大
  2. 在交易日历上（复用 01 脚本的 data∩rate 对齐逻辑）计算日度特征：
     - txt_pos / txt_neg: 近60交易日利多/利空事件的强度衰减和（指数半衰10日）
     - txt_net: 净情绪 = pos - neg
     - txt_loose: 货币宽松类事件的衰减强度
     - txt_cnt20: 近20交易日事件数（事件密度）
     - txt_gap: 距最近事件的交易日数（封顶60，事件稀疏度）
  3. 输出 model_nn/data/text_features_daily.csv（date + 6列，供 01 脚本按日期对齐合并）

防泄漏：事件日 d 的特征包含 d 日事件（当日新闻盘后可知），标签为 d+5 日方向，无前视。
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
XGB_DATA = BASE / "model_new" / "data"
EVENT_CSV = BASE / "data" / "out_site" / "文本标签_政策事件日历.csv"
OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(exist_ok=True)

HALF_LIFE = 10.0   # 衰减半衰期（交易日）
MAX_GAP = 60       # gap 封顶


def build_trading_days():
    """复刻 01 脚本的日期构造：data.csv ∩ rate.csv 内连接 -> 交易日历"""
    df_data = pd.read_csv(XGB_DATA / "data.csv", usecols=["Unnamed: 0"])
    df_rate = pd.read_csv(XGB_DATA / "rate.csv")[["time", "rate"]]
    df_data["Unnamed: 0"] = pd.to_datetime(df_data["Unnamed: 0"])
    df_rate["time"] = pd.to_datetime(df_rate["time"])
    merged = pd.merge(df_data, df_rate, left_on="Unnamed: 0",
                      right_on="time", how="inner").sort_values("Unnamed: 0")
    return merged["Unnamed: 0"].reset_index(drop=True)


def main():
    days = build_trading_days()
    ev = pd.read_csv(EVENT_CSV)
    ev["date"] = pd.to_datetime(ev["date"])
    ev["weight"] = ev["intensity"].astype(float) * ev["direction"].astype(float)

    # 事件日期 -> 交易日索引（事件落在非交易日时归到其后的第一个交易日）
    day_arr = days.values
    ev = ev.sort_values("date").reset_index(drop=True)
    ev["day_idx"] = np.searchsorted(day_arr, ev["date"].values)

    n = len(days)
    w = np.exp(-np.log(2.0) / HALF_LIFE * np.arange(MAX_GAP))  # 半衰期指数权重

    pos = np.zeros(n); neg = np.zeros(n); loose = np.zeros(n)
    cnt20 = np.zeros(n, dtype=int); gap = np.full(n, MAX_GAP, dtype=int)

    ev_idx = ev["day_idx"].values
    is_loose = (ev["category"] == "货币宽松").values
    wt = ev["weight"].values
    j = 0  # 指针：<= 当前交易日的事件前缀
    for d in range(n):
        # 新纳入窗口起点前的事件（指针推进，保证同一天多事件都计入）
        while j < len(ev) and ev_idx[j] <= d:
            j += 1
        # 近60日窗口内的事件
        lo = np.searchsorted(ev_idx, d - MAX_GAP + 1, side="left")
        for k in range(lo, j):
            dist = d - ev_idx[k]
            decay = w[dist] if dist < MAX_GAP else 0.0
            contrib = wt[k] * decay
            if contrib >= 0:
                pos[d] += contrib
            else:
                neg[d] += -contrib
            if is_loose[k]:
                loose[d] += decay * ev["intensity"].values[k]
            gap[d] = min(gap[d], dist)
        # 近20日事件数
        lo20 = np.searchsorted(ev_idx, d - 19, side="left")
        cnt20[d] = j - lo20

    out = pd.DataFrame({
        "date": days.dt.strftime("%Y-%m-%d"),
        "txt_pos": np.round(pos, 4),
        "txt_neg": np.round(neg, 4),
        "txt_net": np.round(pos - neg, 4),
        "txt_loose": np.round(loose, 4),
        "txt_cnt20": cnt20,
        "txt_gap": gap,
    })
    out_path = OUT_DIR / "text_features_daily.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"交易日: {n}, 事件数: {len(ev)}, 半衰期: {HALF_LIFE}日")
    print(f"输出: {out_path}")
    print(out[out["txt_cnt20"] > 0].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
