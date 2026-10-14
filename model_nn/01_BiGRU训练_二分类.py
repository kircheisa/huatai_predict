# -*- coding: utf-8 -*-
"""
BiGRU + Inception + Self-Attention 神经网络版（仿 资料2.md 架构）
预测10年期国债收益率未来5天方向（二分类：下行=1）

与 XGB 主流程（model_new/03_XGB训练_二分类.py）的对齐：
  1. 特征: 复用 model_new/data/xgb_train_features_bin.csv（280列，已含STL/遗传特征）
  2. 标签: 复用 model_new/data/y_bin.csv（与XGB同一套标签，含超趋势版本）
  3. 评估: 逐fold对齐 model_new/data/xgb_fold_metrics_bin.csv 的28个fold边界
  4. 死区: |rate_delta_5d|<2bp 样本权重0.3（与XGB一致）

相对 资料2.md 原架构的改动（小样本防过拟合）：
  - input_size: 10 -> XGB重要性Top-K（默认30，可--full-features用全部280）
  - output_size: 3 -> 2（二分类）
  - Block数: 3 -> 默认1（--n-blocks可调，样本仅~2000个序列撑不起3层堆叠）
  - hidden_size: 64 -> 默认32
  - 新增: 早停（验证集loss, patience=8）、训练窗口内标准化（防泄漏）、时间序切分
"""

import sys
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

BASE = Path(__file__).resolve().parent.parent
XGB_DATA = BASE / "model_new" / "data"
OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(exist_ok=True)


# =====================================================================
# 参数
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser()
    # 数据对齐
    p.add_argument("--start-date", default="2018-01-01")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--top-k", type=int, default=30,
                   help="按XGB重要性取Top-K特征（--full-features则忽略）")
    p.add_argument("--full-features", action="store_true",
                   help="使用全部280列特征")
    p.add_argument("--deadzone-bp", type=float, default=2.0,
                   help="标签死区(bp)，死区内样本权重0.3，0=禁用")
    # 模型结构（资料2架构的降配版）
    p.add_argument("--seq-len", type=int, default=60, help="滑动窗口长度")
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--gru-layers", type=int, default=2)
    p.add_argument("--n-blocks", type=int, default=1, choices=[1, 2, 3],
                   help="BiGRU+Inception+Attention 堆叠块数（资料2为3，小样本建议1）")
    p.add_argument("--dropout", type=float, default=0.5)
    # 训练
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8, help="早停patience")
    p.add_argument("--val-rate", type=float, default=0.2, help="训练窗口尾部验证占比")
    p.add_argument("--max-train-days", type=int, default=1260,
                   help="训练窗口上限（交易日），0=expanding全量")
    p.add_argument("--seed", type=int, default=42)
    # 运行
    p.add_argument("--folds-file", default="xgb_fold_metrics_bin.csv",
                   help="XGB的fold边界文件（保证两模型测试区间完全一致）")
    # 文本标签特征（方案A：事件日历日度特征）
    p.add_argument("--text-features", default="",
                   help="日度文本特征CSV路径（02_文本特征构建.py 输出），空=不用")
    p.add_argument("--tag", default="", help="实验标签，输出文件名后缀（如 _TXT_A）")
    return p.parse_args()


# =====================================================================
# 模型结构（照搬 资料2.md，超参外部传入）
# =====================================================================
class InceptionModule(nn.Module):
    """4分支多尺度1D卷积 + 1x1残差 + BN + MaxPool（资料2.md 同款）"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels), nn.ReLU())
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 5, padding=2),
            nn.BatchNorm1d(out_channels), nn.ReLU())
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 7, padding=3),
            nn.BatchNorm1d(out_channels), nn.ReLU())
        self.branch4 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 9, padding=4),
            nn.BatchNorm1d(out_channels), nn.ReLU())
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels * 4, 1),
            nn.BatchNorm1d(out_channels * 4))
        self.pool = nn.MaxPool1d(2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = torch.cat([self.branch1(x), self.branch2(x),
                         self.branch3(x), self.branch4(x)], dim=1)
        out = out + self.residual(x)
        out = self.relu(out)
        return self.pool(out)


class SelfAttention(nn.Module):
    """单头自注意力 Q/K/V（资料2.md 同款，未除 sqrt(d)）"""

    def __init__(self, embed_size):
        super().__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # x: (batch, seq, embed)
        Q, K, V = self.query(x), self.key(x), self.value(x)
        scores = self.softmax(torch.bmm(Q, K.transpose(1, 2)))
        return torch.bmm(scores, V)


class BiGRUInceptionAttentionBlock(nn.Module):
    """BiGRU -> Inception -> SelfAttention -> Dropout"""

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=True)
        self.inception = InceptionModule(hidden_size * 2, hidden_size // 2)
        self.attention = SelfAttention((hidden_size // 2) * 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)              # (B, L, hidden*2)
        out = out.permute(0, 2, 1)            # (B, hidden*2, L)
        out = self.inception(out)             # (B, hidden*2, L//2)
        out = out.permute(0, 2, 1)            # (B, L//2, hidden*2)
        out = self.attention(out)             # (B, L//2, hidden*2)
        return self.dropout(out)


class BiGRUInceptionAttentionModel(nn.Module):
    """N块堆叠 + 全局平均池化 + 全连接二分类"""

    def __init__(self, input_size, hidden_size, num_layers, output_size=2,
                 dropout=0.5, n_blocks=1):
        super().__init__()
        blocks = []
        ch_in = input_size
        for _ in range(n_blocks):
            blocks.append(BiGRUInceptionAttentionBlock(
                ch_in, hidden_size, num_layers, dropout))
            ch_in = (hidden_size // 2) * 4
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(ch_in, output_size)

    def forward(self, x):
        out = self.blocks(x)                  # (B, L//n, ch)
        out = torch.mean(out, dim=1)          # 全局平均池化
        return self.fc(out)                   # (B, 2)


# =====================================================================
# 数据对齐（复用XGB的特征/标签/fold边界）
# =====================================================================
def load_aligned_data(args):
    # 1) 特征（1955×280，已按日期升序、已过滤2018+）
    feats = pd.read_csv(XGB_DATA / "xgb_train_features_bin.csv", index_col=0)

    # 2) 标签：XGB保存的对齐标签（1955行，与特征逐行对应）
    y = pd.read_csv(XGB_DATA / "xgb_labels_bin.csv")["label"].values.astype(int)
    assert len(y) == len(feats), \
        f"标签{len(y)}行 与 特征{len(feats)}行 不对齐"

    # 日期：复刻XGB的merged构造（data.csv ∩ rate.csv 内连接），
    # 去掉标签NaN尾部行（shift(-H)产生），再过滤起始日期
    df_data = pd.read_csv(XGB_DATA / "data.csv", usecols=["Unnamed: 0"])
    df_rate = pd.read_csv(XGB_DATA / "rate.csv")[["time", "rate"]]
    df_data["Unnamed: 0"] = pd.to_datetime(df_data["Unnamed: 0"])
    df_rate["time"] = pd.to_datetime(df_rate["time"])
    merged = pd.merge(df_data, df_rate, left_on="Unnamed: 0",
                      right_on="time", how="inner").sort_values("Unnamed: 0")
    H = args.horizon
    valid = merged["rate"].shift(-H).notna()   # 标签可定义的行
    merged = merged.loc[valid].reset_index(drop=True)

    cutoff = pd.to_datetime(args.start_date)
    mask = merged["Unnamed: 0"].values >= np.datetime64(cutoff)
    merged_f = merged.loc[mask].reset_index(drop=True)
    assert len(merged_f) == len(feats), \
        f"日期对齐失败: merged过滤后{len(merged_f)}行 != 特征{len(feats)}行，" \
        f"检查 --start-date（XGB运行时用的边界）"
    dates = merged_f["Unnamed: 0"]

    # 3) 死区权重：|rate_delta_5d|<2bp -> 0.3（与XGB口径一致，用原始delta）
    r = merged_f["rate"].values
    delta = np.full(len(r), np.nan)
    delta[:-H] = r[H:] - r[:-H]
    if args.deadzone_bp > 0:
        w = np.where(np.abs(delta) * 100 < args.deadzone_bp, 0.3, 1.0)
        w = np.where(np.isnan(delta), 1.0, w)
    else:
        w = np.ones(len(r))
    n_dead = int((w < 1).sum())

    # 4) 特征筛选：XGB重要性Top-K
    if not args.full_features:
        imp = pd.read_csv(XGB_DATA / "xgb_feature_importance_bin.csv")
        imp_col = "feature" if "feature" in imp.columns else imp.columns[0]
        top_cols = imp.sort_values("importance", ascending=False)[imp_col].tolist()
        top_cols = [c for c in top_cols if c in feats.columns][:args.top_k]
        feats = feats[top_cols]
        print(f"  特征筛选: XGB重要性Top{len(top_cols)}")

    print(f"  样本: {len(feats)}, 特征: {feats.shape[1]}, "
          f"死区降权: {n_dead}/{len(feats)}")
    return feats, y.astype(int), w, dates


def load_folds(args):
    fm = pd.read_csv(XGB_DATA / args.folds_file)
    fm["date_start"] = pd.to_datetime(fm["date_start"])
    fm["date_end"] = pd.to_datetime(fm["date_end"])
    return fm


# =====================================================================
# 滑动窗口序列（张量索引方式，绕过 torch.numpy 桥不兼容问题）
# =====================================================================
def gather_windows(X_t, indices, seq_len):
    """X_t: (N, F) tensor；返回 (n, seq_len, F)，窗口 = 行 [i-L+1, i]"""
    rows = [list(range(i - seq_len + 1, i + 1)) for i in indices]
    idx_t = torch.tensor(rows, dtype=torch.long)
    return X_t[idx_t]


def seq_indices(y_len, end_lo, end_hi, seq_len):
    return [i for i in range(max(seq_len - 1, end_lo), end_hi + 1)
            if i < y_len]


# =====================================================================
# 训练与预测
# =====================================================================
def train_one_fold(args, X_t, y, w, tr_lo, tr_hi, device, input_size):
    """训练窗口 [tr_lo, tr_hi]（end索引），返回模型与训练信息"""
    idx = seq_indices(len(y), tr_lo, tr_hi, args.seq_len)
    if len(idx) < 20:
        return None, {}

    # 时间序切分：尾部 val_rate 做验证（早停用）
    n_val = max(8, int(len(idx) * args.val_rate))
    tr_idx, va_idx = idx[:-n_val], idx[-n_val:]
    Xtr = gather_windows(X_t, tr_idx, args.seq_len)
    Xv = gather_windows(X_t, va_idx, args.seq_len)
    ytr = torch.tensor([int(y[i]) for i in tr_idx])
    yva = torch.tensor([int(y[i]) for i in va_idx])
    wtr = torch.tensor([float(w[i]) for i in tr_idx], dtype=torch.float32)

    torch.manual_seed(args.seed)
    model = BiGRUInceptionAttentionModel(
        input_size=input_size, hidden_size=args.hidden_size,
        num_layers=args.gru_layers, dropout=args.dropout,
        n_blocks=args.n_blocks).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss(reduction="none")

    tr_loader = DataLoader(
        TensorDataset(Xtr, ytr, wtr),
        batch_size=args.batch_size, shuffle=True)

    Xv_d, yv_d = Xv.to(device), yva.to(device)
    best_val, best_state, bad, best_ep = np.inf, None, 0, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb, wb in tr_loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            loss = (ce(model(xb), yb) * wb).mean()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            lv = ce(model(Xv_d), yv_d).mean().item()
        if lv < best_val - 1e-4:
            best_val, best_state, bad, best_ep = lv, \
                {k: v.clone() for k, v in model.state_dict().items()}, 0, ep
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, {"best_ep": best_ep, "val_loss": best_val,
                   "n_train_seq": len(idx)}


@torch.no_grad()
def predict_proba(model, X_t, end_idx, args, device, bs=256):
    """对 end_idx 逐个出预测概率"""
    model.eval()
    arr = gather_windows(X_t, list(end_idx), args.seq_len)
    out = []
    for s in range(0, len(arr), bs):
        xb = arr[s: s + bs].to(device)
        p = torch.softmax(model(xb), dim=1)
        out.append(p[:, 1].cpu().tolist())
    return np.concatenate(out)


# =====================================================================
# 主流程：滚动fold训练评估
# =====================================================================
def main():
    args = parse_args()
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 56)
    print("BiGRU+Inception+Attention 神经网络版（滚动fold评估）")
    print(f"device={device} | blocks={args.n_blocks} | hidden={args.hidden_size}"
          f" | seq={args.seq_len} | epochs<={args.epochs}")
    print("=" * 56)

    print("[1/4] 加载数据（对齐XGB特征/标签/fold）...")
    feats, y, w, dates = load_aligned_data(args)
    folds = load_folds(args)

    # 文本标签特征：按日期对齐合并为额外输入列（不参与Top-K筛选，直接追加）
    if args.text_features:
        tf = pd.read_csv(args.text_features)
        tf["date"] = pd.to_datetime(tf["date"])
        tcols = [c for c in tf.columns if c != "date"]
        date_df = pd.DataFrame({"date": pd.to_datetime(dates.values)})
        merged = date_df.merge(tf, on="date", how="left")
        n_miss = int(merged[tcols].isna().any(axis=1).sum())
        merged[tcols] = merged[tcols].fillna(0.0)
        for c in tcols:
            feats[c] = merged[c].values
        print(f"  文本特征: +{len(tcols)}列 {tcols} (缺失填0: {n_miss}日)")

    # 行索引 <-> 日期映射
    date_arr = dates.values
    X_raw = feats.values

    print(f"[2/4] 滚动训练: {len(folds)} folds ...")
    rows, preds = [], []
    for _, fd in folds.iterrows():
        s = int(np.searchsorted(date_arr, np.datetime64(fd["date_start"])))
        e = int(np.searchsorted(date_arr, np.datetime64(fd["date_end"]), "right")) - 1
        if s >= len(y) or e < s:
            continue
        # 训练窗口（expanding + 上限）
        tr_lo = 0 if args.max_train_days <= 0 else max(0, s - args.max_train_days)

        # 标准化：只用训练段统计（防泄漏）；NaN填0(=均值)，极端值裁剪
        mu = np.nanmean(X_raw[tr_lo:s], axis=0)
        sd = np.nanstd(X_raw[tr_lo:s], axis=0) + 1e-8
        Xs = (X_raw - mu) / sd
        Xs = np.clip(np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0),
                     -8.0, 8.0).astype(np.float32)
        # 矩阵一次性转tensor（tolist绕过torch/numpy ABI不兼容）
        X_t = torch.tensor(Xs.tolist(), dtype=torch.float32)

        model, info = train_one_fold(args, X_t, y, w, tr_lo, s - 1, device,
                                     input_size=X_raw.shape[1])
        if model is None:
            print(f"  fold起点{s}训练样本不足，跳过")
            continue

        end_idx = np.arange(s, e + 1)
        proba = predict_proba(model, X_t, end_idx, args, device)
        yp = (proba >= 0.5).astype(int)
        yt = y[end_idx]

        # 指标（与XGB同口径）
        try:
            auc = roc_auc_score(yt, proba)
        except ValueError:
            auc = np.nan
        acc = accuracy_score(yt, yp)
        f1m = f1_score(yt, yp, average="macro", zero_division=0)
        rows.append({"fold": fd["fold"], "date_start": str(fd["date_start"].date()),
                     "date_end": str(fd["date_end"].date()), "n_test": len(yt),
                     "accuracy": acc, "macro_f1": f1m, "auc": auc,
                     "best_ep": info.get("best_ep"), "n_train_seq": info.get("n_train_seq")})
        for i, pi, yi, di in zip(end_idx, proba, yt, date_arr[end_idx]):
            preds.append({"date": str(pd.Timestamp(di).date()), "row": int(i),
                          "y_true": int(yi), "proba_down": float(pi),
                          "pred": int(pi >= 0.5)})
        print(f"  fold {fd['fold']:>2} [{fd['date_start'].date()}~"
              f"{fd['date_end'].date()}] acc={acc:.4f} auc={auc:.4f} "
              f"ep={info.get('best_ep')}", flush=True)

    fm = pd.DataFrame(rows)
    pr = pd.DataFrame(preds)
    fm.to_csv(OUT_DIR / f"nn_fold_metrics{args.tag}.csv", index=False, encoding="utf-8-sig")
    pr.to_csv(OUT_DIR / f"nn_test_predictions{args.tag}.csv", index=False, encoding="utf-8-sig")

    print("[3/4] 全局指标 ...")
    if len(pr):
        yt, yp, pb = pr["y_true"], pr["pred"], pr["proba_down"]
        g_acc = accuracy_score(yt, yp)
        g_f1 = f1_score(yt, yp, average="macro", zero_division=0)
        g_auc = roc_auc_score(yt, pb)

        print("[4/4] 分年指标 ...")
        pr["year"] = pd.to_datetime(pr["date"]).dt.year
        year_stats = pr.groupby("year").apply(
            lambda g: pd.Series({
                "n": len(g),
                "acc": accuracy_score(g["y_true"], g["pred"]),
                "auc": roc_auc_score(g["y_true"], g["proba_down"])
                if g["y_true"].nunique() > 1 else np.nan,
            }), include_groups=False)

        lines = [
            "=" * 56,
            f"BiGRU+Inception+Attention ({args.n_blocks}块, hidden={args.hidden_size},"
            f" top-k={'all' if args.full_features else args.top_k})"
            f"{' [文本特征]' if args.text_features else ''}{args.tag}",
            f"device={device}",
            "-" * 56,
            f"全局: Acc={g_acc:.4f}  MacroF1={g_f1:.4f}  AUC={g_auc:.4f}",
            "XGB基线: Acc=0.5635  MacroF1=0.5627  AUC=0.5822",
            "-" * 56,
            "分年:",
            year_stats.round(4).to_string(),
            "-" * 56,
            f"folds={len(fm)}  预测样本={len(pr)}  耗时={time.time()-t0:.0f}s",
            "=" * 56,
        ]
        report = "\n".join(lines)
        print(report)
        (OUT_DIR / f"nn_metrics{args.tag}.txt").write_text(report, encoding="utf-8")
        print(f"\n输出: {OUT_DIR / f'nn_metrics{args.tag}.txt'}")


if __name__ == "__main__":
    import traceback
    try:
        print("MAIN START", flush=True)
        main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        raise
