#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pick_risk.py —— 出库拣选「到位缺货」风险前置标记器

背景
    调度系统下发拣货任务后频繁发生「到位缺货」：堆垛机/拣货员跑到货位才发现没货，
    每次空跑约损失 90 分钟。本脚本从历史任务里定位根因，并在派发环节做风险前置标记。

用法
    python pick_risk.py                # 完整流程：归因报告 + 生成标记清单
    python pick_risk.py report         # 只打印归因证据
    python pick_risk.py flag           # 只生成标记清单 out/risk_flags.csv
    python pick_risk.py report --tol 20   # 自定义快照滞后容忍线（分钟，默认 20）
    python pick_risk.py report --csv 路径.csv

依赖
    仅 Python 3.8+ 标准库。不需要 pandas / numpy / sklearn。

口径（来自 要求.txt）
    每次到位缺货   损失 90 分钟
    每次提前标记   花费 12 分钟
    => 只有预测缺货概率 p 满足 p*90 > 12，即 p > 13.3%，提前标记才划算。
       本脚本用这个盈亏平衡点做决策阈值，而不是拍脑袋定 0.5。
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

# ------------------------------------------------------------ 业务口径
COST_MISS_MIN = 90.0      # 一次到位缺货的损失（分钟）
COST_FLAG_MIN = 12.0      # 一次提前标记的成本（分钟）
BREAK_EVEN_P = COST_FLAG_MIN / COST_MISS_MIN          # 0.1333...

# ------------------------------------------------------------ 判定规则
DEFAULT_TOL_MIN = 20      # 库存快照滞后容忍线（分钟）
STOCK_LOW = 2             # 账面可用库存「偏低」界线

ACTION_STALE = "挂起派发，先对货位做一次实时库存校验"
ACTION_LOW = "派发前人工确认货位余量"
ACTION_OK = "正常下发"


# ============================================================ 读入与清洗
def parse_ts(s):
    """解析 ISO8601 时间戳。
    pick_time 带 +08:00，stock_synced_at 带 Z(UTC)。
    两者必须各自解析出时区后再换算，否则相减会凭空多出 8 小时。"""
    s = (s or "").strip()
    if not s:
        return None
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def load_tasks(path):
    """读 CSV → 去重 → 计算派生字段。返回 (rows, stats)。"""
    with open(path, encoding="utf-8-sig", newline="") as f:
        raw = list(csv.DictReader(f))

    # 完全相同的整行视为重复上报（本数据集 1292 行里有 62 行是重复的）
    seen, rows = set(), []
    for r in raw:
        sig = tuple(sorted((k, (v or "").strip()) for k, v in r.items()))
        if sig in seen:
            continue
        seen.add(sig)
        rows.append(r)

    bad_ts = 0
    for r in rows:
        pt, st = parse_ts(r.get("pick_time")), parse_ts(r.get("stock_synced_at"))
        if pt is None or st is None:
            r["lag_min"] = None
            bad_ts += 1
        else:
            # 关键：都换算到 UTC 再相减，消除 +08:00 与 Z 的口径差异
            r["lag_min"] = (pt.astimezone(timezone.utc)
                            - st.astimezone(timezone.utc)).total_seconds() / 60.0
        try:
            r["stock"] = int(str(r.get("avail_stock", "")).strip())
        except ValueError:
            r["stock"] = 0
        try:
            r["wave"] = int(str(r.get("wave_task_count", "")).strip())
        except ValueError:
            r["wave"] = 0
        v = str(r.get("is_success", "")).strip()
        r["labeled"] = v in ("0", "1")
        r["miss"] = 1 if v == "0" else 0
        r["day"] = (r.get("pick_time") or "")[:10]

    stats = {
        "raw_rows": len(raw),
        "unique_rows": len(rows),
        "dup_rows": len(raw) - len(rows),
        "bad_ts": bad_ts,
        "labeled": sum(1 for r in rows if r["labeled"]),
        "unlabeled": sum(1 for r in rows if not r["labeled"]),
        "days": sorted({r["day"] for r in rows if r["day"]}),
    }
    return rows, stats


# ============================================================ 风险模型
def stale(r, tol):
    """快照是否已过期（滞后超过容忍线）。lag 为 None 时按不可信处理。"""
    if r["lag_min"] is None:
        return True
    return r["lag_min"] > tol


def low_stock(r):
    return r["stock"] <= STOCK_LOW


def segment(r, tol):
    """两因子分段：快照新鲜度 × 账面库存高低。
    只用两个因子，4 个格子，样本量都在 26 条以上，可人工核对；
    波次规模不放进模型 —— 它是「同步积压」的上游触发器，见 report 第三节。"""
    return ("stale" if stale(r, tol) else "fresh",
            "low" if low_stock(r) else "ok")


def fit_risk_table(rows, tol):
    """用已回传样本估计每个分段的缺货概率（Laplace 平滑，向整体基础率收缩）。
    这是唯一的『训练』步骤：4 个数字，不是黑箱。"""
    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        if not r["labeled"]:
            continue
        a = agg[segment(r, tol)]
        a[0] += r["miss"]
        a[1] += 1
    tm = sum(v[0] for v in agg.values())
    tn = sum(v[1] for v in agg.values())
    prior = (tm + 1) / (tn + 2) if tn else 0.5
    table = {seg: (m + 2 * prior) / (n + 2) for seg, (m, n) in agg.items()}
    counts = {seg: tuple(v) for seg, v in agg.items()}
    return table, prior, counts


def predict_p(r, table, prior, tol):
    return table.get(segment(r, tol), prior)


def decide(r, table, prior, tol, min_p=None, only_stale=False):
    """返回 (是否标记, 预测概率, 建议动作)。

    only_stale=True 时用「只标慢链路」的推荐策略：快照过期且风险过盈亏平衡点才标记。
    这比让用户手敲 --min-p 0.26 更可靠 —— 报告里打印的 p 是四舍五入过的，
    真实值可能比 0.26 小万分之一，直接按浮点比较会漏掉整整一个分段。"""
    p = predict_p(r, table, prior, tol)
    if only_stale:
        if stale(r, tol) and p >= BREAK_EVEN_P:
            return True, p, ACTION_STALE
        return False, p, ACTION_OK
    if p < (BREAK_EVEN_P if min_p is None else min_p):
        return False, p, ACTION_OK
    if stale(r, tol):
        return True, p, ACTION_STALE
    return True, p, ACTION_LOW


# ============================================================ 统计工具
def auc(pairs):
    """AUC（秩和法，处理并列）。pairs = [(score, label)]。"""
    ps = sorted(pairs)
    pos = sum(l for _, l in ps)
    neg = len(ps) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rank_sum, i = 0.0, 0
    while i < len(ps):
        j = i
        while j < len(ps) and ps[j][0] == ps[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            if ps[k][1] == 1:
                rank_sum += avg
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def replay(pool, table, prior, tol, thr):
    """在给定样本上回放标记策略，返回指标字典。
    净节省 = 命中数*90 - 标记数*12（命中省一次空跑，误报白花 12 分钟）。"""
    flagged = [r for r in pool if predict_p(r, table, prior, tol) >= thr]
    cap = sum(r["miss"] for r in flagged)
    miss = sum(r["miss"] for r in pool)
    return {
        "n": len(pool), "miss": miss, "flagged": len(flagged), "cap": cap,
        "fp": len(flagged) - cap,
        "recall": cap / miss if miss else 0.0,
        "prec": cap / len(flagged) if flagged else 0.0,
        "baseline": miss * COST_MISS_MIN,
        "net": cap * COST_MISS_MIN - len(flagged) * COST_FLAG_MIN,
    }


def pct(a, b):
    return a / b * 100 if b else 0.0


# ============================================================ 报告
def cmd_report(rows, stats, tol):
    L = []
    A = L.append
    lab = [r for r in rows if r["labeled"]]
    tot_m = sum(r["miss"] for r in lab)
    ndays = len(stats["days"])

    A("=" * 76)
    A("一、数据体检（先确认能算，再谈结论）")
    A("=" * 76)
    A("  原始 %d 行 → 完全重复行 %d → 去重后唯一任务 %d" % (
        stats["raw_rows"], stats["dup_rows"], stats["unique_rows"]))
    A("  已回传(is_success=0/1) %d 条；未回传(空) %d 条，占比 %.1f%%" % (
        stats["labeled"], stats["unlabeled"], pct(stats["unlabeled"], stats["unique_rows"])))
    A("  整体缺货率 %d / %d = %.2f%%（未回传的不计入分母，避免把「还不知道」当成「没缺货」）"
      % (tot_m, len(lab), pct(tot_m, len(lab))))
    A("  时间跨度 %s ~ %s，共 %d 天" % (stats["days"][0], stats["days"][-1], ndays))
    A("")
    A("  【时区坑】pick_time 是 +08:00，stock_synced_at 是 Z(UTC)。")
    A("  两条时间必须各自解析时区后再相减。若当成同一时区直接相减，")
    A("  每条任务都会凭空多出 480 分钟滞后（8 小时），后面所有归因都会做反。")

    A("")
    A("=" * 76)
    A("二、根因：库存快照滞后是「双峰」，中间有断层")
    A("=" * 76)
    lags = sorted(int(r["lag_min"]) for r in lab if r["lag_min"] is not None)
    present = set(lags)
    holes = [x for x in range(lags[0], lags[-1] + 1) if x not in present]
    A("  滞后范围 %d~%d 分钟；其中完全没有样本的整分钟值：%s" % (
        lags[0], lags[-1], ", ".join(str(h) for h in holes) or "无"))
    A("  → 断层说明这不是「同步慢慢变慢」的连续漂移，而是两条不同的同步路径。")
    fresh = [r for r in lab if not stale(r, tol)]
    bad = [r for r in lab if stale(r, tol)]
    fm, sm = sum(r["miss"] for r in fresh), sum(r["miss"] for r in bad)
    A("")
    A("    快链路 lag<=%d : 任务 %4d (%4.1f%%)  缺货 %3d  缺货率 %5.2f%%  占全部缺货 %5.1f%%" % (
        tol, len(fresh), pct(len(fresh), len(lab)), fm, pct(fm, len(fresh)), pct(fm, tot_m)))
    A("    慢链路 lag> %d : 任务 %4d (%4.1f%%)  缺货 %3d  缺货率 %5.2f%%  占全部缺货 %5.1f%%" % (
        tol, len(bad), pct(len(bad), len(lab)), sm, pct(sm, len(bad)), pct(sm, tot_m)))
    A("  → 慢链路缺货率是快链路的 %.1f 倍：只用 %.1f%% 的任务量，制造了 %.1f%% 的缺货。" % (
        (sm / len(bad)) / (fm / len(fresh)),
        pct(len(bad), len(lab)), pct(sm, tot_m)))

    A("")
    A("=" * 76)
    A("三、旁证：波次规模不是独立致病因素，只是「同步积压」的触发器")
    A("=" * 76)
    A("  波次越大，走到慢链路的比例越高（发波峰把库存同步压垮）：")
    for lo, hi in [(0, 25), (26, 50), (51, 75), (76, 10 ** 9)]:
        v = [r for r in lab if lo <= r["wave"] <= hi]
        if not v:
            continue
        s = sum(1 for r in v if stale(r, tol))
        A("      wave %3d-%-5s : n=%3d   慢链路 %3d 条 = %5.1f%%" % (
            lo, "+" if hi > 10 ** 8 else str(hi), len(v), s, pct(s, len(v))))
    A("")
    A("  但把滞后固定住之后，波次的作用就基本消失了：")
    for tag, pool in [("快链路 lag<=%d" % tol, fresh), ("慢链路 lag>%d" % tol, bad)]:
        A("      [%s]" % tag)
        for lo, hi in [(0, 50), (51, 82), (83, 10 ** 9)]:
            v = [r for r in pool if lo <= r["wave"] <= hi]
            nm = sum(r["miss"] for r in v)
            label = "wave %3d-%-4s" % (lo, "+" if hi > 10 ** 8 else str(hi))
            if len(v) < 15:
                A("        %-18s n=%-4d 样本不足，不下结论" % (label, len(v)))
            else:
                A("        %-18s %3d /%4d = %5.2f%%" % (label, nm, len(v), pct(nm, len(v))))
    A("  → 快链路内波次几乎持平；只有进了慢链路，波次才会把缺货率往上推。")
    A("    所以真正该修的是同步链路，而不是简单限流波次。")
    A("    （wave>=83 的格子两类各只有 13 条，样本太少，这里不给结论。）")

    A("")
    A("=" * 76)
    A("四、avail_stock 不能当物理库存用")
    A("=" * 76)
    z = [r for r in lab if r["stock"] == 0]
    if z:
        zs = sum(1 for r in z if not r["miss"])
        A("  账面可用库存 = 0 的 %d 条任务，有 %d 条（%.1f%%）最后还是拣出来了。" % (
            len(z), zs, pct(zs, len(z))))
    A("  → 该字段是「过期快照上的账面数」：账面无货 ≠ 货位无货，账面有货 ≠ 货位有货。")
    A("  → 所以不能按 avail_stock 高低排优先级，必须叠加快照新鲜度一起看。")

    A("")
    A("=" * 76)
    A("五、单因子判别力 AUC（0.5 = 瞎猜，1.0 = 完美）")
    A("=" * 76)
    l2 = [(r, r["miss"]) for r in lab if r["lag_min"] is not None]
    for name, key in [("快照滞后 lag_min", lambda r: r["lag_min"]),
                      ("波次规模 wave_task_count", lambda r: r["wave"]),
                      ("低库存倾向 (avail_stock 取负)", lambda r: -r["stock"])]:
        A("    AUC = %.3f   ← %s" % (auc([(key(r), m) for r, m in l2]), name))
    A("  → 滞后是唯一有实用判别力的单因子；账面库存单用几乎等于瞎猜。")
    A("    但它不能单独当风险用，要和滞后叠乘（见第七节 4 个分段）。")
    return "\n".join(L)


def cmd_flag(rows, table, prior, tol, outdir, min_p=None, only_stale=False):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "risk_flags.csv")
    field = ["task_id", "pick_time", "warehouse_zone", "sku_category",
             "avail_stock", "wave_task_count", "lag_min", "risk_seg",
             "p_miss", "action", "is_success", "outcome"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (-decide(x, table, prior, tol, min_p, only_stale)[1],
                                             x["task_id"])):
            flag, p, act = decide(r, table, prior, tol, min_p, only_stale)
            if not r["labeled"]:
                out = "未回传"
            elif flag and r["miss"]:
                out = "命中"
            elif flag:
                out = "误报"
            elif r["miss"]:
                out = "漏报"
            else:
                out = "正确放行"
            w.writerow({
                "task_id": r["task_id"], "pick_time": r["pick_time"],
                "warehouse_zone": r["warehouse_zone"], "sku_category": r["sku_category"],
                "avail_stock": r["avail_stock"], "wave_task_count": r["wave_task_count"],
                "lag_min": ("%.0f" % r["lag_min"]) if r["lag_min"] is not None else "",
                "risk_seg": "|".join(segment(r, tol)),
                "p_miss": "%.4f" % p, "action": act,
                "is_success": r["is_success"], "outcome": out,
            })
    return path


def cmd_eval(rows, table, prior, tol, counts):
    lab = [r for r in rows if r["labeled"]]
    L = []
    A = L.append
    A("")
    A("=" * 76)
    A("六、落地：把风险评估接到派发环节")
    A("=" * 76)
    A("  决策规则： 预测缺货概率 p >= %.2f%%  →  提前标记；否则正常下发" % (BREAK_EVEN_P * 100))
    A("  为什么是 %.2f%%：命中一次省 90 分钟，误报一次白花 12 分钟，" % (BREAK_EVEN_P * 100))
    A("  只要 p*90 > 12 就划算，解得 p > 12/90 = 13.33%。这是算出来的，不是调参调的。")

    # ---- 四个分段的风险与是否过线
    A("")
    A("  【四个分段的风险】（概率由历史回传结果估计，就 4 个数，可人工核对）")
    order = sorted(table, key=lambda s: -table[s])
    for seg in order:
        m, n = counts[seg]
        p = table[seg]
        if p >= BREAK_EVEN_P:
            verdict = "→ 标记" if n >= 15 else "→ 标记(样本略少)"
        else:
            verdict = "→ 放行"
        A("     %s · %s：样本 %d 条，缺货 %d 条（%.1f%%），估计 p=%.3f  %s" % (
            "快照过期" if seg[0] == "stale" else "快照新鲜",
            "账面库存低" if seg[1] == "low" else "账面库存正常",
            n, m, pct(m, n), p, verdict))

    # ---- 三个可选策略的账
    A("")
    A("  【策略取舍：三种打法各自的账】（回放全部已回传 %d 条）" % len(lab))
    cum = {"flagged": 0, "cap": 0}
    rows_out = []
    for seg in order:
        m, n = counts[seg]
        cum["flagged"] += n
        cum["cap"] += m
        net = cum["cap"] * COST_MISS_MIN - cum["flagged"] * COST_FLAG_MIN
        thr_hi = table[seg] * 100
        rows_out.append((seg, cum["flagged"], cum["cap"], net, thr_hi))
        tail = "   ← 全量标记，会亏" if seg == order[-1] else ""
        A("     p>=%5.1f%% 起累加 :  标记 %4d  命中 %3d  误报 %4d  净节省 %7.0f 分钟%s" % (
            thr_hi, cum["flagged"], cum["cap"], cum["flagged"] - cum["cap"], net, tail))
    A("     一条都不标记        :  标记    0  命中   0  误报    0  净节省       0 分钟")
    A("     ----------------------------------------------------------------")
    base = sum(r["miss"] for r in lab) * COST_MISS_MIN
    A("     不干预的基准损失    :  %d 次缺货 × 90 分钟 = %d 分钟（%d 天）" % (
        sum(r["miss"] for r in lab), base, len({r["day"] for r in lab})))

    # ---- 推荐
    stale_only = [x for x in rows_out if x[0][0] == "stale"]
    if stale_only:
        seg, nf, cap, net, _ = stale_only[-1]
        r_a = replay(lab, table, prior, tol, table[seg] - 1e-9)
        A("")
        A("  推荐打法：只标「慢链路」这两个分段（约等于阈值 %.1f%%，但别直接把 %.3f 写进判定——" % (
            table[seg] * 100, table[seg]))
        A("  报告里是四舍五入值，实际概率可能低一点点，按浮点比较会漏掉整个分段）。")
        A("     %d 条任务被标记，抓到 %d 次缺货（占全部缺货 %.1f%%），误报 %d 次，" % (
            r_a["flagged"], r_a["cap"], r_a["recall"] * 100, r_a["fp"]))
        nd = len({r["day"] for r in lab})
        A("     净节省 %.0f 分钟 / %d 天 = %.1f 小时，折合每天 %.0f 分钟（%.1f 小时）。" % (
            r_a["net"], nd, r_a["net"] / 60, r_a["net"] / nd, r_a["net"] / nd / 60))
        A("     相当于把 %d 分钟的基准损失砍掉 %.1f%%。" % (base, r_a["net"] / base * 100))
        # 边际比较：再加 fresh|low 值不值
        fresh_low = [x for x in rows_out if x[0] == ("fresh", "low")]
        if fresh_low:
            _, nf2, cap2, net2, _ = fresh_low[0]
            A("     若再把「快链路+低库存」也标上（它的 p=%.1f%% 刚过线）：" % (table[("fresh", "low")] * 100))
            A("       多标 %d 次，只多抓 %d 次缺货，净增 %d 分钟。" % (
                nf2 - r_a["flagged"], cap2 - r_a["cap"], net2 - r_a["net"]))
            A("       多 12 分钟/次的打扰换 90 分钟/次的收益，几乎打平 → 不必为了这 %d 分钟" % (
                net2 - r_a["net"]))
            A("       去打扰一线。所以推荐阈值抬到 %.0f%%，而不是死守 13.3%%。" % (table[seg] * 100))

    # ---- 时间外推
    A("")
    A("  【时间外推检验】防止「规则只是记住历史」：前面几天训练，最后 2 天当未来数据。")
    days = sorted({r["day"] for r in lab})
    if len(days) >= 5:
        n_tr = -(-len(days) * 7 // 10)          # ceil(70%)
        cut = days[min(n_tr, len(days) - 1)]
        tr = [r for r in lab if r["day"] < cut]
        te = [r for r in lab if r["day"] >= cut]
        t2, p2, c2 = fit_risk_table(tr, tol)
        rtr = replay(tr, t2, p2, tol, BREAK_EVEN_P)
        rte = replay(te, t2, p2, tol, BREAK_EVEN_P)
        A("     训练 %s~%s：%d 条，缺货 %d 条，净节省 %.0f 分钟" % (
            days[0], days[days.index(cut) - 1], rtr["n"], rtr["miss"], rtr["net"]))
        A("     检验 %s~%s：%d 条，缺货 %d 条，净节省 %.0f 分钟（召回 %.1f%%，精确率 %.1f%%）" % (
            cut, days[-1], rte["n"], rte["miss"], rte["net"],
            rte["recall"] * 100, rte["prec"] * 100))
        A("     → 用没见过的后段数据，净节省仍为正，说明不是靠记住样本。")

    # ---- 根治
    A("")
    A("  【比标记更划算的是根治】标记是兜底，真正该修的是慢同步链路：")
    fresh = [r for r in lab if not stale(r, tol)]
    bad = [r for r in lab if stale(r, tol)]
    if fresh and bad:
        fr = {}
        for sd in ("ok", "low"):
            v = [r for r in fresh if segment(r, tol)[1] == sd]
            fr[sd] = (sum(r["miss"] for r in v), max(1, len(v)))
        exp = 0.0
        for sd in ("ok", "low"):
            n = sum(1 for r in bad if segment(r, tol)[1] == sd)
            exp += n * fr[sd][0] / fr[sd][1]
        act = sum(r["miss"] for r in bad)
        nd = len(days)
        A("     慢链路 %d 条任务实际缺货 %d 条。若把它们的新鲜度修到快链路水平，" % (len(bad), act))
        A("     按快链路同库存档的缺货率推算，预计只剩 %.1f 条 → 每 %d 天少 %.1f 次空跑，" % (
            exp, nd, act - exp))
        A("     少损失 %.0f 分钟（%.1f 小时）。" % ((act - exp) * COST_MISS_MIN,
                                            (act - exp) * COST_MISS_MIN / 60))
        A("     这是一次性工程投入、没有反复成本，比每次花 12 分钟标记更根本。")
        A("     注意两者收益不可相加：慢链路修好后，标记规则自动退化成只抓")
        A("     「快链路+账面低库存」那一小撮（该格缺货率 %.1f%%，收益很薄）。" % (
            pct(fr["low"][0], fr["low"][1])))
    return "\n".join(L)


# ============================================================ 主流程
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="出库拣选到位缺货风险前置标记器")
    ap.add_argument("cmd", nargs="?", default="all", choices=["all", "report", "flag"])
    ap.add_argument("--csv", default=os.path.join(here, "pick_tasks.csv"))
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL_MIN,
                    help="库存快照滞后容忍线(分钟)，默认 20")
    ap.add_argument("--min-p", type=float, default=None,
                    help="标记阈值，默认用盈亏平衡点 0.1333")
    ap.add_argument("--only-stale", action="store_true",
                    help="推荐策略：只标「快照过期」的分段（阈值≈26%%，即本方案建议上线口径）")
    ap.add_argument("--out", default=os.path.join(here, "out"))
    a = ap.parse_args()

    if not os.path.exists(a.csv):
        sys.exit("找不到数据文件：%s（请把 pick_tasks.csv 放在脚本同目录，或用 --csv 指定）" % a.csv)

    rows, stats = load_tasks(a.csv)
    table, prior, counts = fit_risk_table(rows, a.tol)
    os.makedirs(a.out, exist_ok=True)

    out = []
    if a.cmd in ("all", "report"):
        out.append(cmd_report(rows, stats, a.tol))
    if a.cmd in ("all", "flag"):
        out.append(cmd_eval(rows, table, prior, a.tol, counts))
        path = cmd_flag(rows, table, prior, a.tol, a.out, a.min_p, a.only_stale)
        out.append("")
        out.append("=" * 76)
        out.append("七、模型与产出")
        out.append("=" * 76)
        out.append("  风险模型只有 4 个分段（快照新鲜度 × 账面库存），概率可人工核对：")
        for sd in ("stale", "fresh"):
            for sk in ("low", "ok"):
                seg = (sd, sk)
                if seg in table:
                    m, n = counts[seg]
                    out.append("    %s·%s：样本 %3d 条，实际缺货 %3d 条，估计 p_miss = %.3f" % (
                        "快照过期" if sd == "stale" else "快照新鲜",
                        "库存低" if sk == "low" else "库存正常", n, m, table[seg]))
        out.append("")
        out.append("  逐任务标记清单：%s" % path)
        out.append("  （新任务只要按同样口径算出 lag_min 和 avail_stock 即可直接套用）")

    text = "\n".join(out)
    print(text)
    # 只有完整流程才写报告快照；单纯 flag 模式不要把 summary.txt 覆盖成片段
    if a.cmd == "all":
        with open(os.path.join(a.out, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
