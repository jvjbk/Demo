#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify.py —— 独立复算 README 里引用的每一个关键数字。

故意不 import pick_risk.py：另写一套读数据和算滞后的逻辑（手工解析时间戳），
这样如果主程序哪里写错，这里会报 FAIL 而不是跟着一起错。

用法：  python verify.py
退出码：全部通过 0，有任一不符 1
"""

import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "pick_tasks.csv")

# 期望值（与 README 正文一致）
EXPECT = {
    "raw_rows": 1292,
    "dup_rows": 62,
    "unique_rows": 1230,
    "labeled": 1179,
    "blank": 51,
    "total_miss": 105,
    "lag_min": 0,
    "lag_max": 70,
    "fresh_n": 931, "fresh_miss": 36,
    "stale_n": 248, "stale_miss": 69,
    "stock0_n": 36, "stock0_ok": 28,
    "net_slow": 3234,      # 69*90 - 248*12
    "net_all": 3366,       # 83*90 - 342*12
}

ok = 0
bad = 0


def check(name, got, want, tol=0):
    global ok, bad
    good = abs(got - want) <= tol if isinstance(want, (int, float)) else got == want
    if good:
        ok += 1
        print("  [PASS] %-34s = %s" % (name, got))
    else:
        bad += 1
        print("  [FAIL] %-34s = %s   期望 %s" % (name, got, want))


TS = re.compile(r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})(Z|[+-]\d{2}:\d{2})$")


def to_epoch(s):
    """手工把时间戳换成"自 1970-01-01 起的分钟数"，不借助 datetime.fromisoformat。
    即：把本地时间按 UTC 解释得到一个基准，再按后缀补上偏移。"""
    m = TS.match(s.strip())
    if not m:
        raise ValueError("时间戳格式不认识: %r" % s)
    y, mo, d, h, mi, se, tz = m.groups()
    # 简易日序（本数据只跨 2026 年 9 月，不用管闰年）
    import calendar
    days = calendar.timegm((int(y), int(mo), int(d), 0, 0, 0, 0, 0, 0)) // 86400
    minutes = days * 1440 + int(h) * 60 + int(mi) + int(se) / 60.0
    if tz == "Z":
        off = 0
    else:
        sign = 1 if tz[0] == "+" else -1
        off = sign * (int(tz[1:3]) * 60 + int(tz[4:6]))
    return minutes - off


def main():
    print("=" * 66)
    print("verify.py —— 独立复算（不 import 主程序）")
    print("=" * 66)

    with open(CSV, encoding="utf-8-sig", newline="") as f:
        raw = list(csv.DictReader(f))

    print("\n[A] 清洗")
    check("原始行数", len(raw), EXPECT["raw_rows"])

    seen, rows = set(), []
    for r in raw:
        k = tuple(sorted((a, (b or "").strip()) for a, b in r.items()))
        if k in seen:
            continue
        seen.add(k)
        rows.append(r)
    check("完全重复行", len(raw) - len(rows), EXPECT["dup_rows"])
    check("去重后唯一任务", len(rows), EXPECT["unique_rows"])

    for r in rows:
        v = (r["is_success"] or "").strip()
        r["_lab"] = v in ("0", "1")
        r["_miss"] = 1 if v == "0" else 0
        r["_lag"] = to_epoch(r["pick_time"]) - to_epoch(r["stock_synced_at"])
        r["_stock"] = int(r["avail_stock"])

    lab = [r for r in rows if r["_lab"]]
    check("已回传条数", len(lab), EXPECT["labeled"])
    check("未回传条数", len(rows) - len(lab), EXPECT["blank"])
    check("总缺货条数", sum(r["_miss"] for r in lab), EXPECT["total_miss"])

    print("\n[B] 时区口径")
    lags = [r["_lag"] for r in lab]
    check("滞后最小值(分钟)", round(min(lags)), EXPECT["lag_min"])
    check("滞后最大值(分钟)", round(max(lags)), EXPECT["lag_max"])
    neg = sum(1 for x in lags if x < 0)
    check("滞后为负的条数", neg, 0)
    holes = [m for m in range(int(min(lags)), int(max(lags)) + 1)
             if not any(abs(x - m) < 0.5 for x in lags)]
    check("中间断层的分钟值", holes, [19, 20, 21])
    # 若忽略时区直接相减，会恒定多出 480 分钟
    naive = [to_epoch(r["pick_time"].replace("+08:00", "Z")) - to_epoch(r["stock_synced_at"]) for r in lab]
    check("忽略时区后的恒定偏差", round(min(naive)), 480)

    print("\n[C] 两簇对照")
    fresh = [r for r in lab if r["_lag"] <= 20]
    stale = [r for r in lab if r["_lag"] > 20]
    check("快链路(lag<=20) 任务数", len(fresh), EXPECT["fresh_n"])
    check("快链路 缺货数", sum(r["_miss"] for r in fresh), EXPECT["fresh_miss"])
    check("慢链路(lag>20) 任务数", len(stale), EXPECT["stale_n"])
    check("慢链路 缺货数", sum(r["_miss"] for r in stale), EXPECT["stale_miss"])
    fr = sum(r["_miss"] for r in fresh) / len(fresh) * 100
    sr = sum(r["_miss"] for r in stale) / len(stale) * 100
    check("快链路缺货率(%)", round(fr, 2), 3.87, 0.01)
    check("慢链路缺货率(%)", round(sr, 2), 27.82, 0.01)
    share = sum(r["_miss"] for r in stale) / sum(r["_miss"] for r in lab) * 100
    check("慢链路占总缺货(%)", round(share, 1), 65.7, 0.05)
    check("慢链路/快链路 倍数", round(sr / fr, 1), 7.2, 0.05)

    print("\n[D] 账面库存可信度")
    z = [r for r in lab if r["_stock"] == 0]
    check("账面库存=0 的任务数", len(z), EXPECT["stock0_n"])
    check("其中拣货成功数", sum(1 for r in z if not r["_miss"]), EXPECT["stock0_ok"])

    print("\n[E] 收益算式（口径：缺货 90 分钟 / 标记 12 分钟）")
    nf, cap = len(stale), sum(r["_miss"] for r in stale)
    check("只标慢链路 净节省(分钟)", cap * 90 - nf * 12, EXPECT["net_slow"])
    # 盈亏平衡点
    be = 12 / 90 * 100
    check("盈亏平衡概率(%)", round(be, 2), 13.33, 0.01)
    # 全量标记应当亏钱
    allnet = sum(r["_miss"] for r in lab) * 90 - len(lab) * 12
    check("全量标记 净节省(分钟)", allnet, -4698)

    print("\n" + "=" * 66)
    print("通过 %d 项，失败 %d 项" % (ok, bad))
    print("=" * 66)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
