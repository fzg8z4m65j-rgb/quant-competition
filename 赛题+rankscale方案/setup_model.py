# -*- coding: utf-8 -*-
"""从 Release 权重包恢复 model/ 目录下的 40 个 LightGBM booster。

用法：
    python setup_model.py --weights model_weights.zip
    python setup_model.py --weights model_weights.zip --force

权重包结构要求：
    model_weights.zip
    └── model_g{1..4}_s{seed}.txt        （40 个文件）

恢复后会与 model/meta.json 中的模型清单逐一校验。
"""
import argparse
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")


def main() -> int:
    ap = argparse.ArgumentParser(description="Restore model weights from a release archive.")
    ap.add_argument("--weights", required=True, help="path to model_weights.zip")
    ap.add_argument("--force", action="store_true", help="overwrite existing booster files")
    args = ap.parse_args()

    meta_path = os.path.join(MODEL_DIR, "meta.json")
    if not os.path.exists(meta_path):
        print("[ERR] 找不到 %s" % meta_path, file=sys.stderr)
        print("      请确认在仓库根目录运行本脚本。", file=sys.stderr)
        return 2
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    expected = [spec["file"] for spec in meta["models"]]
    print("meta.json 声明 %d 个模型，%d 个特征" % (len(expected), meta["n_features"]))

    if not os.path.exists(args.weights):
        print("[ERR] 找不到权重包 %s" % args.weights, file=sys.stderr)
        print("      请从 GitHub Releases 下载 model_weights.zip。", file=sys.stderr)
        return 2

    have = {f for f in os.listdir(MODEL_DIR) if f.endswith(".txt")}
    missing = [f for f in expected if f not in have]
    if not missing and not args.force:
        print("[OK] 40 个 booster 已就位，无需恢复。")
        return 0
    print("待恢复 %d 个文件" % len(missing))

    os.makedirs(MODEL_DIR, exist_ok=True)
    written = 0
    with zipfile.ZipFile(args.weights) as z:
        names = z.namelist()
        for fname in expected:
            # 兼容 zip 内带目录前缀的情况
            hit = [n for n in names if os.path.basename(n) == fname]
            if not hit:
                print("[ERR] 权重包中缺少 %s" % fname, file=sys.stderr)
                return 1
            dest = os.path.join(MODEL_DIR, fname)
            with z.open(hit[0]) as src, open(dest, "wb") as out:
                out.write(src.read())
            written += 1
    print("已写入 %d 个 booster 到 %s" % (written, MODEL_DIR))

    # 校验
    after = {f for f in os.listdir(MODEL_DIR) if f.endswith(".txt")}
    still = [f for f in expected if f not in after]
    if still:
        print("[ERR] 恢复后仍缺失 %d 个：%s" % (len(still), still[:5]), file=sys.stderr)
        return 1

    print("[OK] 全部 %d 个 booster 就位。" % len(expected))
    print()
    print("接下来运行自检（不需要原始数据）：")
    print("    python selfcheck.py")
    print("预期：max|Δ| = 0.000e+00  判定 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
