#!/usr/bin/env bash
# 一键运行：先自检，再出报告
set -e
cd "$(dirname "$0")"

echo "============================================"
echo " 1/2 自检（独立复算关键数字）"
echo "============================================"
python3 verify.py 2>/dev/null || python verify.py

echo
echo "============================================"
echo " 2/2 归因报告 + 生成标记清单"
echo "============================================"
python3 pick_risk.py 2>/dev/null || python pick_risk.py

echo
echo "============================================"
echo " 3/3 按推荐口径重新生成下发清单（只标慢链路）"
echo "============================================"
python3 pick_risk.py flag --only-stale 2>/dev/null || python pick_risk.py flag --only-stale

echo
echo "完成。"
echo "  报告快照：$(pwd)/out/summary.txt"
echo "  下发清单：$(pwd)/out/risk_flags.csv（259 条待标记）"
